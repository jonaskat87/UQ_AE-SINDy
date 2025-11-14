import os
import sys
import argparse
import pickle
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from joblib import Parallel, delayed

# --- MPI / threading commands must come before any torch.* or numpy.* calls ---
# Force single‐threading in BLAS libs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Use spawn instead of fork to avoid inheriting unwanted state
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from pathlib import Path
from contextlib import redirect_stdout
from sklearn.model_selection import KFold

# project imports (data_utils, training, thresholding, etc.)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src import data_utils as du
from training_scripts import train_ae_sindy as train
from src import thresholding, training


def _init_worker():
    """Initializer for worker processes: set environment and torch thread limits early.

    This runs once in each worker process before any tasks are executed, avoiding
    calls to set_num_interop_threads after parallel runtime has started.
    """
    try:
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    except Exception:
        pass
    try:
        # Import torch locally inside worker initializer; if unavailable, skip thread tuning
        import importlib

        torch = importlib.import_module("torch")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        # Torch might not be importable in some lightweight test environments; ignore
        pass

parser = argparse.ArgumentParser(description="cv+ conformal for AE-SINDy")
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-t", "--test_size", type=float, default=0.2, help="test split proportion [0,1]"
)
parser.add_argument("-k", "--folds", type=int, default=5, help="number of CV folds")
parser.add_argument(
    "-c", "--cpus", type=int, default=None, help="override number of CPUs to use"
)
parser.add_argument("-e", "--epochs", type=int, default=100, help="training epochs")
parser.add_argument("-b", "--batches", type=int, default=200, help="batch size")
parser.add_argument(
    "-a",
    "--alpha",
    nargs="+",
    type=float,
    default=[0.1],
    help="miscoverage rate(s); list of values in (0,1)",
)
parser.add_argument(
    "-n", "--n_jobs", type=int, default=1, help="number of parallel jobs for Mahalanobis computation (default 1)"
)
args = parser.parse_args()

params = {
    "data_src": args.data_name,
    "random_seed": 1952,
    "num_epochs": args.epochs,
    "batch_size": args.batches,
    "learning_rate": 0.004204813405972317,
    "latent_dim": 3,
    "poly_order": 2,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_metaweight": 0.500989969537634,
    "CNN": False,
    "sequential_threshold_method": None,  # None, bimodal_gmm, knee_detection
    "sequential_thresholding_interval": None,  # None
    "print_frequency": 1,
}

# load data once
outputs = du.open_mass_dataset(
    name=args.data_name,
    data_dir=Path(__file__).parent.parent.parent / "data",
    sample_time=None,
    test_size=args.test_size,
    calib_size=None,
    random_state=1952,
)

# Compute & set weights based on Champion et al recs
lambda1, lambda2, _ = du.champion_calculate_weights(
    du.NormedBinDatasetDzDt(
        outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
    ),
    lambda1_metaweight=params["lambda1_metaweight"],
)
params["loss_weight_recon"] = 1.0
params["loss_weight_sindy_x"] = lambda1
params["loss_weight_sindy_z"] = lambda2

alphas = args.alpha
alpha_lows = [a / 2 for a in alphas]
alpha_ups = alpha_lows.copy()


def init_model(device, params, outputs):
    torch.manual_seed(params["random_seed"])
    model = train.AESINDy(
        n_channels=1,
        n_bins=outputs["n_bins"],
        n_latent=params["latent_dim"],
        poly_order=params["poly_order"],
        CNN=params["CNN"],
        sequential_thresholding=bool(params["sequential_thresholding_interval"]),
    )
    optimal_path = os.path.join(
        "results",
        "Optuna",
        "ERF Dataset",
        "AE-SINDy_LimParams",
        "erf_FFNN_latent3_order2_tr1000_lr0.004204813405972317_bs25_weights1.0-561.064697265625-56106.47265625_46d657b7ac094414a37843315fdeebbc",
    )
    ae_sindy_checkpoint = torch.load(
        os.path.join(
            optimal_path,
            "erf_FFNN_latent3_order2_tr1000_lr0.004204813405972317_bs25_weights1.0-561.064697265625-56106.47265625_46d657b7ac094414a37843315fdeebbc.pth",
        ),
        weights_only=True,
    )
    model.load_state_dict(ae_sindy_checkpoint)
    return model.to(device)


def init_optimizer(model, params):
    opt = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    if params["sequential_threshold_method"] is not None:
        opt.thresholder = thresholding.AdaptiveSequentialThresholdingSINDy(
            model.dzdt,
            thresholding.AdaptiveThresholdAnalyzer(
                method=params["sequential_threshold_method"],
                min_epochs_between=params["sequential_thresholding_interval"],
            ),
        )
    return opt


def init_scheduler(opt):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")


def compute_scores_and_inv(res_t):
    """Compute Mahalanobis distance scores and inverse covariance matrix.
    
    Args:
        res_t: residuals at a fixed time t, shape (m, d) where m=samples, d=dimensions
    
    Returns:
        scores_t: Mahalanobis distance scores, shape (m,)
        Sigma_inv: Inverse covariance matrix, shape (d, d)
        mu_t: Mean of residuals, shape (d,)
    """
    mu_t = res_t.mean(axis=0, keepdims=True)  # (1, d)
    res_c = res_t - mu_t  # center residuals
    lw = LedoitWolf().fit(res_c)  # covariance of centered residuals
    Sigma_inv = lw.precision_
    scores_t = np.einsum("mi,ij,mi->m", res_c, Sigma_inv, res_c)
    return scores_t, Sigma_inv, mu_t.squeeze(0)


def one_sided_quantiles(residuals, alpha_lows, alpha_ups):
    lows = np.array(alpha_lows)
    ups = 1.0 - np.array(alpha_ups)
    all_q = np.concatenate([lows, ups])
    sort_idx = np.argsort(all_q)
    q_sorted = all_q[sort_idx]
    qs = np.quantile(residuals, q_sorted, axis=0)
    qs_unsorted = np.empty_like(qs)
    qs_unsorted[sort_idx] = qs
    m = len(alpha_lows)
    return qs_unsorted[:m], qs_unsorted[m:]


# this helper utility encodes and then decodes each DSD to test how close the predictions are to the actual values
def encode_decode_ae(x, model):
    # for DSD data, decoded (predictions from network)
    DSD_all = (
        model.decoder(model.encoder(torch.Tensor(x))).detach().numpy()
    )  # DSD pred t+0
    return DSD_all


# run latent space dynamics
def run_ae_X_latent(x, m, model):
    # for DSD_mass data, not decoded (so still within the latent space)
    latents_all = np.empty(
        (x.shape[0], x.shape[1], params["latent_dim"] + 1),
        dtype=float,
    )
    z_enc_train = model.encoder(torch.Tensor(x)).detach().numpy()
    zlim = np.zeros((3 + 1, 2))  # DSD bins
    for il in range(3):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m.min()
    zlim[-1][1] = m.max()

    for id in range(len(x)):  # loop through each initial condition/sample/gridbox
        z0 = np.concatenate((z_enc_train[id, 0, :], np.array([m[id, 0]])), axis=-1)
        latents_all[id] = du.simulate(z0, outputs["dsd_time"], model.dzdt, zlim)
    return latents_all


# run full network
# computes the predicted DSD and mass trajectories
def run_ae_X(x, m, model):
    # DSD data, decoded (predictions from network)
    DSD_all = np.empty(x.shape, dtype=float)
    # mass data
    M_all = np.empty(m.shape, dtype=float)
    z_enc_train = model.encoder(torch.Tensor(x)).detach().numpy()
    zlim = np.zeros((3 + 1, 2))  # DSD bins
    for il in range(3):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m.min()
    zlim[-1][1] = m.max()

    for id in range(len(x)):  # loop through each initial condition/sample/gridbox
        z0 = np.concatenate((z_enc_train[id, 0, :], np.array([m[id, 0]])), axis=-1)
        latents_pred = du.simulate(z0, outputs["dsd_time"], model.dzdt, zlim)
        DSD_all[id] = (
            model.decoder(torch.Tensor(latents_pred[:, :-1])).detach().numpy()
        )  # add DSD predictions
        M_all[id] = latents_pred[:, -1]  # add mass predictions
    return DSD_all, M_all


def run_all(x, m, model, params):
    DSD_all = [
        np.empty(x.shape, dtype=float),  # decoder only
        np.empty(
            (x.shape[0], x.shape[1], params["latent_dim"]), dtype=float
        ),  # latent only
        np.empty(x.shape, dtype=float),  # full architecture
    ]
    M_all = np.empty(
        m.shape,
        dtype=float,
    )
    DSD_all[0] = encode_decode_ae(x, model=model)
    DSD_all[1] = run_ae_X_latent(x, m, model=model)[..., :-1]
    DSD_all[2], M_all = run_ae_X(x, m, model=model)
    return DSD_all, M_all


def _fold_worker(args):
    """
    This runs in its own process.  Make absolutely sure no
    DataLoader inside here uses num_workers>0.
    """
    fold_id, train_idx, val_idx, outputs, params, device = args

    # pin torch to 1 thread where possible (some platforms raise if set after parallel work)
    try:
        torch.set_num_threads(1)
    except Exception:
        # If threads have already been configured in this process, ignore
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # On some PyTorch builds you cannot set interop threads after
        # parallel runtime has been initialized; ignore and continue.
        pass
    except Exception:
        # Catch-all for unexpected issues setting interop threads
        pass

    # initialize model/opt/sched
    model = init_model(device, params, outputs)
    optimizer = init_optimizer(model, params)
    scheduler = init_scheduler(optimizer)
    early_stop = training.EarlyStopping(patience=params["patience"])

    # build _single‐worker_ dataloaders with num_workers=0
    train_ds = du.NormedBinDatasetDzDt(
        outputs["x_train"][train_idx],
        outputs["dsd_time"],
        outputs["m_train"][train_idx],
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=0
    )

    # use the *full* test set here
    test_ds = du.NormedBinDatasetDzDt(
        outputs["x_test"], outputs["dsd_time"], outputs["m_test"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=len(test_ds), shuffle=False, num_workers=0
    )

    # silently train
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull):
            best_model, *_ = train.train_and_eval(
                params["num_epochs"],
                model,
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                params,
                early_stopping=early_stop,
                print_flag=False,
                device=device,
            )

    # compute residuals & predictions on *validation* slice
    DSD_val_all, M_val_all = run_all(
        outputs["x_train"][val_idx],
        outputs["m_train"][val_idx],
        best_model,
        params,
    )
    # compute predictions on test‐set
    DSD_test_all, M_test_all = run_all(
        outputs["x_test"], outputs["m_test"], best_model, params
    )

    # assemble fold‐wise arrays
    D_resid, D_pred = [], []
    for i in range(3):
        if i == 1:
            enc_val = (
                best_model.encoder(torch.Tensor(outputs["x_train"][val_idx]).to(device))
                .detach()
                .cpu()
                .numpy()
            )
            resid_i = DSD_val_all[i] - enc_val
        else:
            resid_i = DSD_val_all[i] - outputs["x_train"][val_idx]
        D_resid.append(resid_i)
        D_pred.append(DSD_test_all[i])

    M_resid = M_val_all - outputs["m_train"][val_idx]
    M_pred = M_test_all

    print(f"[Fold {fold_id+1}] done.", flush=True)
    return D_resid, D_pred, M_resid, M_pred


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if torch.backends.mps.is_available() and params["batch_size"] > 1000
            else "cpu"
        )
    )

    # decide on CPU‐count
    total_cpus = (
        args.cpus or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count()
    )
    print(f"→ launching {args.folds}-fold evaluation on {total_cpus} {device} cores")

    # prepare CV splits (on train set)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=params["random_seed"])
    splits = list(kf.split(outputs["x_train"]))

    # pack arguments for each fold
    fold_args = [
        (fold_id, tr_idx, val_idx, outputs, params, device)
        for fold_id, (tr_idx, val_idx) in enumerate(splits)
    ]

    # spawn the pool with "spawn" and initialize workers to set thread limits early
    ctx = mp.get_context("spawn")
    n_procs = min(total_cpus, args.folds)
    print(f"Using {n_procs} worker processes for {args.folds} folds")
    with ctx.Pool(processes=n_procs, initializer=_init_worker) as pool:
        all_results = pool.map(_fold_worker, fold_args)

    # unpack and compute global quantiles just like before
    DSD_resid = [[] for _ in range(3)]
    DSD_pred = [[] for _ in range(3)]
    M_resid, M_pred = [], []
    for D_res, D_pred, m_res, m_pred in all_results:
        for i in range(3):
            DSD_resid[i].append(D_res[i])
            DSD_pred[i].append(D_pred[i])
        M_resid.append(m_res)
        M_pred.append(m_pred)

    # compute global quantiles & medians exactly as before
    rep_DSD = [None] * 3
    lower = [
        np.empty((len(alphas),) + outputs["x_test"].shape, dtype=float),  # decoder only
        np.empty(
            (
                len(alphas),
                outputs["x_test"].shape[0],
                outputs["x_test"].shape[1],
                params["latent_dim"],
            ),
            dtype=float,
        ),  # latent only
        np.empty(
            (len(alphas),) + outputs["x_test"].shape, dtype=float
        ),  # full architecture
    ]
    upper = lower.copy()
    latent_maha_cv = None  # Will be set if processing latent space
    for i in range(3):
        resid_pool = np.concatenate(DSD_resid[i], axis=0)
        ql, qh = one_sided_quantiles(
            resid_pool,
            alpha_lows,
            alpha_ups,
        )
        rep_DSD[i] = np.median(np.stack(DSD_pred[i], axis=0), axis=0)

        lower[i] = rep_DSD[i][np.newaxis, ...] - qh[:, np.newaxis, ...]
        upper[i] = rep_DSD[i][np.newaxis, ...] - ql[:, np.newaxis, ...]
        
        # For latent space (i==1), also compute Mahalanobis metrics
        if i == 1:
            print("Computing Mahalanobis distance metrics for latent space...")
            m, n, d = resid_pool.shape  # (total_samples, n_time, latent_dim)
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(compute_scores_and_inv)(resid_pool[:, t, :]) for t in range(n)
            )
            scores_list, Sigma_inv_list, mu_list = zip(*results)
            scores = np.stack(scores_list, axis=1)  # (m, n)
            Sigma_inv = np.stack(Sigma_inv_list, axis=0)  # (n, d, d)
            mu = np.stack(mu_list, axis=0)  # (n, d)
            # Compute quantiles of Mahalanobis scores for each alpha and time
            taus = {}
            for alpha in alphas:
                taus[alpha] = np.quantile(scores, 1 - alpha, axis=0)  # (n,)
            latent_maha_cv = {
                "Sigma_inv": Sigma_inv,
                "mu": mu,
                "taus": taus,
            }

    # mass‐trajectory
    resid_pool = np.concatenate(M_resid, axis=0)
    ql, qh = one_sided_quantiles(
        resid_pool,
        alpha_lows,
        alpha_ups,
    )
    rep_m = np.median(np.stack(M_pred, axis=0), axis=0)
    lower_m = rep_m[np.newaxis, ...] - qh[:, np.newaxis, ...]
    upper_m = rep_m[np.newaxis, ...] - ql[:, np.newaxis, ...]

    # finally: pickle lower/upper/representative bands
    outdir = Path("UQ/conformal/results/ae_SINDy")
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"{args.data_name}_cv+{args.folds}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(
            [
                alphas,
                args.test_size,
                outputs["idx_test"],
                (lower, upper, rep_DSD),
                (lower_m, upper_m, rep_m),
                latent_maha_cv,  # NEW: Include Mahalanobis metrics for latent space
            ],
            f,
        )
    print(f"Saved conformal results to {fname}.")


if __name__ == "__main__":
    main()

"""
Script takes in data filepath and does conformal predictions on AE-SINDy model.
"""
import os
import sys
import numpy as np
import argparse
import torch
import time
from pathlib import Path
import pickle
from sklearn.covariance import LedoitWolf
from joblib import Parallel, delayed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du
from src import training, thresholding
from training_scripts import train_ae_sindy as train

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-t",
    "--test_size",
    type=float,
    default=0.2,
    help="testing set proportion, must be between 0 and 1, default is 0.2",
)
parser.add_argument(
    "-m",
    "--method",
    default="split30",
    help="conformal predictions method to use (split[p], full), default is split30. \
                        The p in split indicates what *percent* you want to dedicate out of the full data for calibration.",
)
parser.add_argument(
    "-e", "--epochs", type=int, default=100, help="number of epochs (default is 100)"
)
parser.add_argument(
    "-b", "--batches", type=int, default=200, help="batch size (default is 200)"
)
parser.add_argument(
    "-a",
    "--alpha",
    nargs="+",
    type=float,
    default=0.1,
    help="miscoverage rate(s), must be between 0 and 1, default is 0.1",
)
parser.add_argument(
    "-n", "--n_jobs", type=int, default=1, help="number of parallel jobs for Mahalanobis computation (default is 1)"
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

# Global variables and settings
# Criterion and divergence need to be outside train function to be available in other scripts
torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])
sample_time = None  # for setting times to sample, as indices
criterion = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

# Setting inputted variables
test_size = args.test_size
method = args.method
calib_size = None  # default is no calibration data
# isolate calib_size or k in the situation where you're using split-conformal or cv+, respectively
if method[:5] == "split":
    calib_size = float(method[5:])
    if (calib_size <= 0) | (calib_size >= 100 * (1 - test_size)):
        raise ValueError(
            "Calibration size for split must be a percent strictly between 0 and 100 * (1 - test_size)."
        )
    method = "split"
if method[:3] == "cv+":
    k = int(method[3:])
    method = "cv+"
if method not in [
    "split",
    "full",
    "cv+",
]:  # raise error if method is not one of the list above
    raise ValueError("Conformal predictions method specified has not been implemented.")

alphas = args.alpha if isinstance(args.alpha, (list, tuple)) else [args.alpha]

if (test_size <= 0) | (test_size >= 1):
    raise ValueError("Test set proportion must be between 0 and 1")
if any(a <= 0 or a >= 1 for a in alphas):
    raise ValueError("Coverage rate (alpha) must be between 0 and 1 for all values")

# split each alpha into two equal parts for lower/upper tails
alpha_lows = [a / 2 for a in alphas]
alpha_ups = alpha_lows.copy()

# Set device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if torch.backends.mps.is_available() and params["batch_size"] > 1000
        else "cpu"
    )
)
# torch.backends.cudnn.benchmark = True
print(f"Using {device} device")

if calib_size is not None:
    calib_size *= 0.01
start_time = time.time()
# Open dataset
outputs = du.open_mass_dataset(
    name=params["data_src"],
    data_dir=Path(__file__).parent.parent.parent / "data",
    sample_time=sample_time,
    test_size=test_size,
    calib_size=calib_size,
    random_state=params["random_seed"],
)

# initialize model using optimal weights from results/Optuna
def init_model(device=device):
    # Initialize the model
    model = train.AESINDy(
        n_channels=1,
        n_bins=outputs["n_bins"],
        n_latent=params["latent_dim"],
        poly_order=params["poly_order"],
        CNN=params["CNN"],
        sequential_thresholding=(
            True if params["sequential_thresholding_interval"] is not None else False
        ),
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
    # total_params = sum(p.numel() for p in model.parameters())
    # total_coeffs = sum(p.numel() for p in model.dzdt.parameters())
    # print(
    #     f"Total number of parameters: {total_params}, {total_coeffs} are SINDy coefficients"
    # )
    return model.to(device)


# initialize optimizer
def init_optimizer(model):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    if params["sequential_threshold_method"] is not None:
        params["thresholder"] = thresholding.AdaptiveSequentialThresholdingSINDy(
            model.dzdt,
            thresholding.AdaptiveThresholdAnalyzer(
                method=params["sequential_threshold_method"],
                min_epochs_between=params["sequential_thresholding_interval"],
            ),
        )
    return optimizer


# initialize scheduling
def init_scheduler(optimizer):
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    return sched


early_stopping = training.EarlyStopping(patience=params["patience"])

# Compute & set weights based on Champion et al recs
lambda1, lambda2, lambda3 = du.champion_calculate_weights(
    du.NormedBinDatasetDzDt(
        outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
    ),
    lambda1_metaweight=params["lambda1_metaweight"],
)
print(f"lambda: 1.0, {lambda1}, {lambda2}")
params["loss_weight_recon"] = 1.0
params["loss_weight_sindy_x"] = lambda1
params["loss_weight_sindy_z"] = lambda2

"""
Helper utility functions
"""


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


# runs all three parts of the architecture above and returns data
def run_all(x, m, model):
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


def one_sided_quantiles(residuals, alpha_lows, alpha_ups):
    """
    Compute all lower- and upper-tail quantiles in one shot.

    residuals : array_like, shape (N, T, D)
    alpha_lows: list of alpha/2 levels (e.g. [0.125, 0.025] for 75% & 95%)
    alpha_ups : same as alpha_lows
    Returns
    -------
    q_low, q_high : arrays of shape (len(alpha_lows), T, D)
    """

    # 1) build the full list of levels
    lows = np.array(alpha_lows)
    ups = 1.0 - np.array(alpha_ups)
    all_q = np.concatenate([lows, ups])  # e.g. [0.125, 0.025, 0.875, 0.975]

    # 2) sort levels and remember how to invert
    sort_idx = np.argsort(all_q)
    q_sorted = all_q[sort_idx]

    # 3) single quantile call
    qs = np.quantile(residuals, q_sorted, axis=0)

    # 4) invert the sort
    qs_unsorted = np.empty_like(qs)
    qs_unsorted[sort_idx] = qs

    # 5) split into lows / highs
    m = len(alpha_lows)
    q_low = qs_unsorted[:m]
    q_high = qs_unsorted[m:]
    return q_low, q_high


"""
Run conformal predictions. 
The data splits used are different for each method, so we will configure those separately in each case.
"""

if method == "full":
    # 0) configure data and dataloaders
    train_data = du.NormedBinDatasetDzDt(
        outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    test_data = du.NormedBinDatasetDzDt(
        outputs["x_test"], outputs["dsd_time"], outputs["m_test"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )

    # 1) fit & predict on training data
    DSD_lower_full = [
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
    DSD_upper_full = DSD_lower_full.copy()
    M_lower_full = np.empty(
        (len(alphas),) + outputs["m_test"].shape,
        dtype=float,
    )
    M_upper_full = M_lower_full.copy()

    model = init_model(device)
    optimizer = init_optimizer(model)
    sched = init_scheduler(optimizer)

    # suppress output of train_and_eval
    (
        best_model,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = train.train_and_eval(
        params["num_epochs"],
        model,
        train_loader,
        test_loader,
        optimizer,
        sched,
        params,
        early_stopping=early_stopping,
        print_flag=True,
        device=device,
    )
    print("Predicting the training data.")
    DSD_train_all, M_train_all = run_all(
        x=outputs["x_train"], m=outputs["m_train"], model=best_model
    )
    print("Predicting the testing data.")
    DSD_test_all, M_test_all = run_all(
        x=outputs["x_test"], m=outputs["m_test"], model=best_model
    )
    print("Running vanilla conformal predictions.")
    latent_maha_full = None  # Will be set if using latent space with Mahalanobis
    for i in range(3):  # loop through different subsets of the architecture
        if (
            i == 1
        ):  # in latent case, you should do it relative to latent space. Otherwise, not
            DSD_res_signed = (
                DSD_train_all[i]
                - best_model.encoder(torch.Tensor(outputs["x_train"])).detach().numpy()
            )
            # Compute Mahalanobis distance-based conformal scores for latent space
            z_train_enc = best_model.encoder(torch.Tensor(outputs["x_train"])).detach().numpy()
            residuals_lat = DSD_train_all[i] - z_train_enc  # (m, n, d)
            m, n, d = residuals_lat.shape
            print("Computing Mahalanobis distance metrics for latent space...")
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(compute_scores_and_inv)(residuals_lat[:, t, :]) for t in range(n)
            )
            scores_list, Sigma_inv_list, mu_list = zip(*results)
            scores = np.stack(scores_list, axis=1)  # (m, n)
            Sigma_inv = np.stack(Sigma_inv_list, axis=0)  # (n, d, d)
            mu = np.stack(mu_list, axis=0)  # (n, d)
            # Compute quantiles of Mahalanobis scores for each alpha and time
            taus = {}
            for alpha in alphas:
                taus[alpha] = np.quantile(scores, 1 - alpha, axis=0)  # (n,)
            latent_maha_full = {
                "z_enc_test_pred": DSD_test_all[i],  # (N_test, n, latent_dim)
                "Sigma_inv": Sigma_inv,
                "mu": mu,
                "taus": taus,
            }
        else:
            DSD_res_signed = DSD_train_all[i] - outputs["x_train"]
        DSD_q_low, DSD_q_high = one_sided_quantiles(
            DSD_res_signed, alpha_lows, alpha_ups
        )
        DSD_lower_full[i] = (
            DSD_test_all[i][np.newaxis, ...] - DSD_q_high[:, np.newaxis, ...]
        )
        DSD_upper_full[i] = (
            DSD_test_all[i][np.newaxis, ...] - DSD_q_low[:, np.newaxis, ...]
        )
        if i == 2:  # also check mass in full architecture case
            M_res_signed = M_train_all - outputs["m_train"]
            M_q_low, M_q_high = one_sided_quantiles(M_res_signed, alpha_lows, alpha_ups)
            M_lower_full = M_test_all[np.newaxis, ...] - M_q_high[:, np.newaxis, ...]
            M_upper_full = M_test_all[np.newaxis, ...] - M_q_low[:, np.newaxis, ...]
    lower = DSD_lower_full
    upper = DSD_upper_full
    rep_DSD = DSD_test_all
    lower_m = M_lower_full
    upper_m = M_upper_full
    rep_m = M_test_all
    latent_maha = latent_maha_full

if method == "split":
    # 0) configure data and dataloaders
    train_data = du.NormedBinDatasetDzDt(
        outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    calib_data = du.NormedBinDatasetDzDt(
        outputs["x_calib"], outputs["dsd_time"], outputs["m_calib"]
    )
    calib_loader = torch.utils.data.DataLoader(
        calib_data, batch_size=len(calib_data), shuffle=True
    )
    test_data = du.NormedBinDatasetDzDt(
        outputs["x_test"], outputs["dsd_time"], outputs["m_test"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )

    # 1) fit & predict on training data
    DSD_lower_full = [
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
    DSD_upper_full = DSD_lower_full.copy()
    M_lower_full = np.empty(
        (len(alphas),) + outputs["m_test"].shape,
        dtype=float,
    )
    M_upper_full = M_lower_full.copy()

    model = init_model(device)
    optimizer = init_optimizer(model)
    sched = init_scheduler(optimizer)

    (
        best_model,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = train.train_and_eval(
        params["num_epochs"],
        model,
        train_loader,
        test_loader,
        optimizer,
        sched,
        params,
        early_stopping=early_stopping,
        print_flag=True,
        device=device,
    )
    print("Predicting the calibration data.")
    DSD_calib_all, M_calib_all = run_all(
        x=outputs["x_calib"], m=outputs["m_calib"], model=best_model
    )
    print("Predicting the testing data.")
    DSD_test_all, M_test_all = run_all(
        x=outputs["x_test"], m=outputs["m_test"], model=best_model
    )
    print("Running split conformal predictions.")
    latent_maha_split = None  # Will be set if using latent space with Mahalanobis
    for i in range(3):  # loop through different subsets of the architecture
        if (
            i == 1
        ):  # in latent case, you should do it relative to latent space. Otherwise, not
            DSD_res_signed = (
                DSD_calib_all[i]
                - best_model.encoder(torch.Tensor(outputs["x_calib"])).detach().numpy()
            )
            # Compute Mahalanobis distance-based conformal scores for latent space
            z_calib_enc = best_model.encoder(torch.Tensor(outputs["x_calib"])).detach().numpy()
            residuals_lat = DSD_calib_all[i] - z_calib_enc  # (m, n, d)
            m, n, d = residuals_lat.shape
            print("Computing Mahalanobis distance metrics for latent space...")
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(compute_scores_and_inv)(residuals_lat[:, t, :]) for t in range(n)
            )
            scores_list, Sigma_inv_list, mu_list = zip(*results)
            scores = np.stack(scores_list, axis=1)  # (m, n)
            Sigma_inv = np.stack(Sigma_inv_list, axis=0)  # (n, d, d)
            mu = np.stack(mu_list, axis=0)  # (n, d)
            # Compute quantiles of Mahalanobis scores for each alpha and time
            taus = {}
            for alpha in alphas:
                taus[alpha] = np.quantile(scores, 1 - alpha, axis=0)  # (n,)
            latent_maha_split = {
                "z_enc_test_pred": DSD_test_all[i],  # (N_test, n, latent_dim)
                "Sigma_inv": Sigma_inv,
                "mu": mu,
                "taus": taus,
            }
        else:
            DSD_res_signed = DSD_calib_all[i] - outputs["x_calib"]
        DSD_q_low, DSD_q_high = one_sided_quantiles(
            DSD_res_signed, alpha_lows, alpha_ups
        )
        DSD_lower_full[i] = (
            DSD_test_all[i][np.newaxis, ...] - DSD_q_high[:, np.newaxis, ...]
        )
        DSD_upper_full[i] = (
            DSD_test_all[i][np.newaxis, ...] - DSD_q_low[:, np.newaxis, ...]
        )
        if i == 2:  # also check mass in full architecture case
            M_res_signed = M_calib_all - outputs["m_calib"]
            M_q_low, M_q_high = one_sided_quantiles(M_res_signed, alpha_lows, alpha_ups)
            M_lower_full = M_test_all[np.newaxis, ...] - M_q_high[:, np.newaxis, ...]
            M_upper_full = M_test_all[np.newaxis, ...] - M_q_low[:, np.newaxis, ...]
    lower = DSD_lower_full
    upper = DSD_upper_full
    rep_DSD = DSD_test_all
    lower_m = M_lower_full
    upper_m = M_upper_full
    rep_m = M_test_all
    latent_maha = latent_maha_split

"""
Save:
-alpha values,
-indices for test data, 
-(lower and upper interval DSD values, and representative ("center") DSDs), and
-(lower and upper interval mass values, and representative ("center") mass),
in that order.
"""

if method == "split":  # add split percent if needed
    method += str(int(100 * calib_size))
with open(
    os.path.join(
        "UQ", "conformal", "results", "ae_SINDy", args.data_name + "_" + method + ".pkl"
    ),
    "wb",
) as f:
    pickle.dump(
        [
            alphas,
            test_size,
            outputs["idx_test"],
            (lower, upper, rep_DSD),
            (lower_m, upper_m, rep_m),
            latent_maha,  # NEW: Include Mahalanobis metrics for latent space
        ],
        f,
    )

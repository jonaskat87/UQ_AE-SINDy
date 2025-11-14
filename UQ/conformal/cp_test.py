"""
Script for testing how accurate the conformal predictions actually are in terms of coverage
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(parent_directory)  # parent_directory = ~/mphys-surrogate-model

from src import data_utils as du

params = {
    "random_seed": 1952,
    "latent_dim": 3,
}

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-s",
    "--subset",
    type=str,
    required=True,
    choices=["decoder", "latent", "full", "mass"],
    help="which subset of the network you want to test coverage for conformal predictions on: decoder, latent, full, or mass"
    "-decoder: the reconstruction/autoencoder only"
    "-latent: the dynamics in the latent space only"
    "-full: the entire network architecture (reconstruction+dynamics)"
    "-mass: the (normalized) mass as a function of time",
)
parser.add_argument(
    "-a",
    "--model",
    type=str,
    required=True,
    choices=["AR", "NNdzdt", "SINDy"],
    help="the dynamic model/architecture used (required): AR, NNdzdt, or SINDy",
)
parser.add_argument(
    "-m",
    "--method",
    type=str,
    required=True,
    help="which conformal predictions to plot (required): full, split[p], or cv+[k]",
)
parser.add_argument(
    "-e", "--epochs", type=int, default=100, help="training epochs (default 100)"
)
parser.add_argument(
    "-b", "--batches", type=int, default=200, help="batch size (default 200)"
)
args = parser.parse_args()

# Respect CLI overrides for training size
params.update({
    "num_epochs": args.epochs,
    "batch_size": args.batches,
})

method = args.method
calib_size = None
if method[:5] == "split":
    calib_size = float(method[5:])
    method = "split"
if method[:3] == "cv+":
    method = "cv+"
if method not in [
    "split",
    "full",
    "cv+",
]:  # raise error if method is not one of the list above
    raise ValueError("Conformal predictions method specified has not been implemented.")

# load results from conformal predictions
cp_results_file = args.data_name + "_" + args.method + ".pkl"
pickle_path = os.path.join(
    parent_directory,
    "UQ",
    "conformal",
    "results",
    "ae_" + args.model,
    cp_results_file,
)
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Support old format (5 elements) and new format with latent Mahalanobis info (6 elements)
if isinstance(data, (list, tuple)) and len(data) == 5:
    alphas, test_size, _, DSD_bands, m_bands = data
    latent_maha = None
elif isinstance(data, (list, tuple)) and len(data) >= 6:
    alphas, test_size, _, DSD_bands, m_bands, latent_maha = data[:6]
else:
    raise ValueError("Unrecognized conformal results file format")

# load data
if calib_size:
    outputs = du.open_mass_dataset(
        name=args.data_name,
        data_dir=Path(parent_directory) / "data",
        sample_time=None,
        test_size=test_size,
        calib_size=0.01 * calib_size,
        random_state=params["random_seed"],
    )
else:
    outputs = du.open_mass_dataset(
        name=args.data_name,
        data_dir=Path(parent_directory) / "data",
        sample_time=None,
        test_size=test_size,
        random_state=params["random_seed"],
    )

# get prediction bands
subset = args.subset
if subset == "decoder":
    lower = DSD_bands[0][0]
    upper = DSD_bands[1][0]
    rep = DSD_bands[2][0]
elif subset == "latent":
    lower = DSD_bands[0][1]
    upper = DSD_bands[1][1]
    rep = DSD_bands[2][1]
elif subset == "full":
    lower = DSD_bands[0][2]
    upper = DSD_bands[1][2]
    rep = DSD_bands[2][2]
elif subset == "mass":
    lower = m_bands[0]
    upper = m_bands[1]
    rep = m_bands[2]
else:
    raise KeyError("Subset of architecture indicated (via -s) has not been implemented")

# if we are testing conformal predictions on the latent space, we need to load the model
if subset == "latent":
    import torch
    from src import training

    params.update({"num_epochs": 200, "batch_size": 200})

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

    if args.model == "AR":
        from training_scripts import train_ae_ar as train

        params.update(
            {
                "n_lag": 1,
                "layer_size": (63, 98, 30),
                "CNN": False,
                "learning_rate": 0.002482884780966882,
                "wd": 1e-3,
                "patience": 50,
                "tol": 1e-8,
                "w_dx": 0.22816989332325596,
                "w_dz": 0.6719555656053005,
                "w_recon": 1,
                "lr_sched": True,
                "print_frequency": 1,
            }
        )

        init_model = train.AEAutoregressor(
            n_channels=1,
            n_bins=outputs["n_bins"],
            n_latent=params["latent_dim"],
            n_lag=params["n_lag"],
            layer_size=params["layer_size"],
            CNN=params["CNN"],
        )
        optimal_path = os.path.join(
            "results",
            "Optuna",
            "ERF Dataset",
            "AE-AR_2025-07-20T22:45:33_605d8b8697694137a65cab3b1012fffc",
            "erf_FFNN_latent3_order(63, 98, 30)_tr1000_lr0.002482884780966882_bs4_weights0.22816989332325596-0.6719555656053005_7c43ff3e659b47358fa327f690871ed7",
        )
        ae_ar_checkpoint = torch.load(
            os.path.join(
                optimal_path,
                "erf_FFNN_latent3_order(63, 98, 30)_tr1000_lr0.002482884780966882_bs4_weights0.22816989332325596-0.6719555656053005_7c43ff3e659b47358fa327f690871ed7.pth",
            ),
            weights_only=True,
        )
        init_model.load_state_dict(ae_ar_checkpoint)
        init_model.to(device)

        optimizer = torch.optim.AdamW(
            init_model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["wd"],
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        early_stopping = training.EarlyStopping(patience=params["patience"])

        train_data = du.NormedBinDatasetAR(
            outputs["x_train"], outputs["m_train"], lag=params["n_lag"]
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=params["batch_size"], shuffle=True
        )
        test_data = du.NormedBinDatasetAR(
            outputs["x_test"], outputs["m_test"], lag=params["n_lag"]
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=len(test_data), shuffle=True
        )
        (
            model,
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
            init_model,
            train_loader,
            test_loader,
            optimizer,
            sched,
            params,
            early_stopping=early_stopping,
            print_flag=False,
            device=device,
        )
    if args.model == "NNdzdt":
        from training_scripts import train_ae_NNdzdt as train

        params.update(
            {
                "layer_size": (42, 36, 46),
                "CNN": False,
                "learning_rate": 0.00314227212817401,
                "wd": 1e-3,
                "patience": 50,
                "tol": 1e-8,
                "lr_sched": True,
                "print_frequency": 1,
            }
        )

        init_model = train.AENNdzdt(
            n_channels=1,
            n_bins=outputs["n_bins"],
            n_latent=params["latent_dim"],
            layer_size=params["layer_size"],
            CNN=params["CNN"],
        )
        optimal_path = os.path.join(
            "results",
            "Optuna",
            "ERF Dataset",
            "NNdzdt_2025-07-20T23:31:20_3a400c596947422389559813cd41dfe6",
            "erf_FFNN_latent3_layers(42, 36, 46)_tr1000_lr0.00314227212817401_bs4_weights1.0-599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3",
        )
        ae_NNdzdt_checkpoint = torch.load(
            os.path.join(
                optimal_path,
                "erf_FFNN_latent3_layers(42, 36, 46)_tr1000_lr0.00314227212817401_bs4_weights1.0-599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3.pth",
            ),
            weights_only=True,
        )
        init_model.load_state_dict(ae_NNdzdt_checkpoint)
        init_model.to(device)

        optimizer = torch.optim.AdamW(
            init_model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["wd"],
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        early_stopping = training.EarlyStopping(patience=params["patience"])

        # Compute & set weights based on Champion et al recs
        lambda1, lambda2, lambda3 = du.champion_calculate_weights(
            du.NormedBinDatasetDzDt(
                outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
            ),
            lambda1_metaweight=0.5353139650038768,
        )
        params["loss_weight_recon"] = 1.0
        params["loss_weight_sindy_x"] = lambda1
        params["loss_weight_sindy_z"] = lambda2

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

        (
            model,
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
            init_model,
            train_loader,
            test_loader,
            optimizer,
            sched,
            params,
            early_stopping=early_stopping,
            print_flag=False,
            device=device,
        )
    if args.model == "SINDy":
        from training_scripts import train_ae_sindy as train
        from src import thresholding

        params.update(
            {
                "poly_order": 2,
                "CNN": False,
                "sequential_thresholding_interval": None,
                "learning_rate": 0.004204813405972317,
                "wd": 1e-3,
                "sequential_threshold_method": None,  # None, bimodal_gmm, knee_detection
                "patience": 50,
                "lambda1_metaweight": 0.500989969537634,
                "tol": 1e-8,
                "lr_sched": True,
                "print_frequency": 1,
            }
        )

        init_model = train.AESINDy(
            n_channels=1,
            n_bins=outputs["n_bins"],
            n_latent=params["latent_dim"],
            poly_order=params["poly_order"],
            CNN=params["CNN"],
            sequential_thresholding=(
                True
                if params["sequential_thresholding_interval"] is not None
                else False
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
        init_model.load_state_dict(ae_sindy_checkpoint)
        init_model.to(device)

        optimizer = torch.optim.AdamW(
            init_model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["wd"],
        )
        if params["sequential_threshold_method"] is not None:
            params["thresholder"] = thresholding.AdaptiveSequentialThresholdingSINDy(
                init_model.dzdt,
                thresholding.AdaptiveThresholdAnalyzer(
                    method=params["sequential_threshold_method"],
                    min_epochs_between=params["sequential_thresholding_interval"],
                ),
            )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        early_stopping = training.EarlyStopping(patience=params["patience"])

        # Compute & set weights based on Champion et al recs
        lambda1, lambda2, lambda3 = du.champion_calculate_weights(
            du.NormedBinDatasetDzDt(
                outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
            ),
            lambda1_metaweight=params["lambda1_metaweight"],
        )
        params["loss_weight_recon"] = 1.0
        params["loss_weight_sindy_x"] = lambda1
        params["loss_weight_sindy_z"] = lambda2

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

        (
            model,
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
            init_model,
            train_loader,
            test_loader,
            optimizer,
            sched,
            params,
            early_stopping=early_stopping,
            print_flag=False,
            device=device,
        )
    z_enc_test = model.encoder(torch.Tensor(outputs["x_test"])).detach().numpy()

"""
Now we test the conformal prediction coverage!
"""
N = len(outputs["x_test"])  # number of samples/initial conditions
# loop across alphas
for i, alpha in enumerate(alphas):
    print(f"testing for coverage 1-alpha={100*(1-alpha)}%")
    # test if it falls within the bands
    if subset == "latent":
        # If latent Mahalanobis info is present, evaluate ellipsoidal coverage
        if latent_maha is not None:
            # latent_maha expected keys: 'Sigma_inv' (n, d, d), 'mu' (n, d), and 'taus' (dict of alpha -> (n,))
            Sigma_inv = latent_maha.get("Sigma_inv")
            mu = latent_maha.get("mu")
            taus = latent_maha.get("taus")
            # rep latent predictions come from DSD_bands[2][1]
            rep = DSD_bands[2][1]
            # get alpha_index corresponding to this alpha value
            alpha_idx = list(alphas).index(alpha)
            q_thresh = taus[alpha]  # shape (n,)
            # compute Mahalanobis distance for each test sample and time
            n = rep.shape[1]  # number of time steps
            testing = np.zeros((N, n), dtype=bool)
            for t in range(n):
                Sinv_t = Sigma_inv[t]  # (d, d)
                # residuals: predicted - actual
                r_t = rep[:, t, :] - z_enc_test[:, t, :]  # (N, d)
                # mahalanobis squared: (r @ Sinv) * r summed over dims
                m2 = np.sum((r_t @ Sinv_t) * r_t, axis=1)
                m = np.sqrt(np.maximum(m2, 0.0))
                # For current alpha
                testing[:, t] = m <= q_thresh[t]
        else:
            testing = (z_enc_test >= lower[i]) & (z_enc_test <= upper[i])
    elif subset == "mass":
        testing = (outputs["m_test"] >= lower[i]) & (outputs["m_test"] <= upper[i])
    else:
        testing = (outputs["x_test"] >= lower[i]) & (outputs["x_test"] <= upper[i])
    # get fraction that fall within the bands
    counter = np.count_nonzero(testing, axis=0) / N
    if subset == "mass":
        print(
            f"Mean percent of real test trajectories that fall within: {100*np.mean(counter)}"
        )
        print(
            f"Median percent of real test trajectories that fall within: {100*np.median(counter)}"
        )
        print(
            f"Standard deviation of percent of real test trajectories that fall within: {100*np.std(counter)}"
        )
    else:
        print(
            f"Mean percent of real test trajectories that fall within: {100*np.mean(counter, axis=(0, 1))}"
        )
        print(
            f"Median percent of real test trajectories that fall within: {100*np.median(counter, axis=(0, 1))}"
        )
        print(
            f"Standard deviation of percent of real test trajectories that fall within: {100*np.std(counter, axis=(0, 1))}"
        )

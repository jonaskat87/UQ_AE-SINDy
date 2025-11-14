import os
import sys

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from src import data_utils as du

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


def get_latent_trajectories_AR(
    n_latent,
    model,
    dsd_time,
    x_test,
    m_test,
    n_lag=1,
):
    """
    Generate latent trajectories for the AE-AR model

    :param n_latent: Number of latent variables (excluding mass).
    :param model: Trained AE-AR model.
    :param list dsd_time: Time steps for DSD evolution.
    :param numpy.ndarray x_test: Test data array of shape (samples, timesteps, bins).
    :param numpy.ndarray m_test: Test mass array of shape (samples, timesteps).
    :param int n_lag: Number of lag steps for the AR model (default is 1).
    :returns: Tuple containing predicted latent trajectories, true latent trajectories, and predicted DSD.
    """
    z_pred = np.zeros((x_test.shape[0], len(dsd_time), n_latent + 1))
    z_data = np.zeros_like(z_pred)

    for j in range(x_test.shape[0]):
        x0 = x_test[j, :n_lag, :]
        mj = m_test[j, :]
        z0 = np.array(
            [
                model.encoder(torch.Tensor(x0[t]).reshape(1, -1)).detach().numpy()[0]
                for t in range(n_lag)
            ]
        )
        z_data[j, :, -1] = mj
        z_data[j, :n_lag, :-1] = z0
        z_pred[j, :n_lag, :-1] = z0
        z_pred[j, :n_lag, -1] = mj[0]
        for t in range(n_lag, x_test.shape[1]):
            lagged_input = torch.Tensor(z_pred[j, t - n_lag : t, :]).reshape(
                n_lag * (n_latent + 1)
            )
            z_pred[j, t, :] = model.autoregressor(lagged_input).detach().numpy()
            z_data[j, t, :-1] = (
                model.encoder(torch.Tensor(x_test[j, t, :]).reshape(1, -1))
                .detach()
                .numpy()[0]
            )
    x_pred = model.decoder(torch.Tensor(z_pred[:, :, :-1]))

    return z_pred, z_data, x_pred


def get_latent_trajectories_dzdt(
    n_latent,
    model,
    dsd_time,
    x_test,
    m_test,
    x_train,
    m_train,
):
    """
    Generate latent trajectories using the dz/dt (differential equation) approach for the DSD.

    :param int n_latent: Number of latent variables (excluding mass).
    :param model: Trained dz/dt model.
    :param list dsd_time: Time steps for DSD evolution.
    :param numpy.ndarray x_test: Test data array of shape (samples, timesteps, features).
    :param numpy.ndarray m_test: Test mass array of shape (samples, timesteps).
    :param numpy.ndarray x_train: Training data array
    :param numpy.ndarray m_train: Training mass array
    :returns: Tuple containing predicted latent trajectories, true latent trajectories, and predicted DSD.
    """

    # Compute limits
    z_enc_train = model.encoder(torch.Tensor(x_train)).detach().numpy()
    zlim = np.zeros((n_latent + 1, 2))
    for il in range(n_latent):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m_train.min()
    zlim[-1][1] = m_train.max()

    z_pred = np.zeros((x_test.shape[0], len(dsd_time), n_latent + 1))
    z_data = np.zeros_like(z_pred)

    z_data[:, :, :-1] = model.encoder(torch.Tensor(x_test)).detach().numpy()
    z_data[:, :, -1] = m_test

    for j in range(x_test.shape[0]):
        z0 = z_data[j, 0, :]
        latents_pred = du.simulate(z0, dsd_time, model.dzdt, zlim).squeeze()
        z_pred[j, :, :] = latents_pred
    x_pred = model.decoder(torch.Tensor(z_pred[:, :, :-1]))

    return z_pred, z_data, x_pred


def get_performance_metrics(x_test, m_test, z_pred, x_pred, tol=1e-8):
    """
    Compute performance metrics for model predictions, including KL divergence, Wasserstein distance,
    normalized mean squared error, and mass difference.

    :param numpy.ndarray x_test: Test data array of shape (samples, timesteps, bins).
    :param numpy.ndarray m_test: Test mass array of shape (samples, timesteps).
    :param numpy.ndarray z_pred: Predicted latent trajectories of shape (samples, timesteps, latents+1).
    :param torch.Tensor x_pred: Predicted DSD array of shape (samples, timesteps, bins).
    :param float tol: Tolerance value to avoid log(0)
    :returns: Tuple of arrays for KL divergence, Wasserstein distance, normalized MSE, and mass difference.
    """
    # Extra vars
    n_test = x_test.shape[0]
    n_timesteps = x_test.shape[1]
    divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    test_kl = np.zeros(x_test.shape[0:2])
    test_wass = np.zeros(x_test.shape[0:2])
    test_mse_un = np.zeros(x_test.shape[0:2])
    test_mass_diff = np.zeros(x_test.shape[0:2])
    for nm in range(n_test):
        for nt in range(n_timesteps):
            pred_dsd = x_pred[nm, nt]
            true_dsd = torch.Tensor(x_test[nm, nt]).reshape(1, 1, -1)
            test_kl[nm, nt] = divergence(
                torch.log(pred_dsd + tol),
                torch.log(true_dsd + tol),
            )
            test_wass[nm, nt] = wasserstein_distance(
                pred_dsd.detach().numpy().ravel(), true_dsd.detach().numpy().ravel()
            )
            test_mse_un[nm, nt] = np.linalg.norm(
                (pred_dsd - true_dsd).detach().numpy().ravel()
            ) / np.linalg.norm(true_dsd.detach().numpy().ravel())
            test_mass_diff[nm, nt] = z_pred[nm, nt, -1] - m_test[nm, nt]
    return test_kl, test_wass, test_mse_un, test_mass_diff


class EarlyStopping:

    def __init__(self, patience=5, verbose=False):
        """
        Early stopping utility to halt training when validation loss does not improve.

        :param int patience: Number of epochs to wait for improvement before stopping.
        :param bool verbose: If True, prints a message when early stopping is triggered.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call method to check if validation loss has improved, and update state.

        :param float val_loss: Current validation loss.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

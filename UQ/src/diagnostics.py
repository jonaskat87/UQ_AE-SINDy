import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from scipy.stats import wasserstein_distance

from src import data_utils as du
import plotly.graph_objects as go
import plotly.io as pio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from src import models


def get_latent_trajectories_AR(
    n_latent,
    model,
    dsd_time,
    x_test,
    m_test,
    n_lag=1,
):
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
            lagged_input = torch.cat(
                (
                    torch.Tensor(z_pred[j, t - n_lag : t, :-1]).reshape(
                        n_lag * n_latent
                    ),
                    torch.Tensor([mj[0]]),
                )
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
    # Extra vars
    n_test = x_test.shape[0]
    n_timesteps = x_test.shape[1]
    divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    test_kl = np.zeros(x_test.shape[0:2])
    test_wass = np.zeros(x_test.shape[0:2])
    test_wass_un = np.zeros(x_test.shape[0:2])
    for nm in range(n_test):
        for nt in range(n_timesteps):
            pred_dsd = x_pred[nm, nt]
            pred_dsd_un = pred_dsd * z_pred[nm, nt, -1]
            true_dsd = torch.Tensor(x_test[nm, nt]).reshape(1, 1, -1)
            true_dsd_un = true_dsd * m_test[nm, nt]
            test_kl[nm, nt] = divergence(
                torch.log(pred_dsd + tol),
                torch.log(true_dsd + tol),
            )
            test_wass[nm, nt] = wasserstein_distance(
                pred_dsd.detach().numpy().ravel(), true_dsd.detach().numpy().ravel()
            )
            test_wass_un[nm, nt] = wasserstein_distance(
                pred_dsd_un.detach().numpy().ravel(),
                true_dsd_un.detach().numpy().ravel(),
            )
    return test_kl, test_wass, test_wass_un

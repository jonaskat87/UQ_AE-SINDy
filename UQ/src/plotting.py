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


def plot_losses(
    losses,
    test_losses=None,
    sub_losses=None,
    labels=None,
    title="Training Loss",
    saveas=None,
):
    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="constrained")

    # Plot losses
    ax.plot(losses, label="total train")
    if test_losses is not None:
        ax.plot(test_losses, label="total test", ls="--")
    if sub_losses is not None:
        for j, loss in enumerate(sub_losses):
            ax.plot(loss, label=labels[j])

    # Accoutrements
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_reconstructions(
    model, test_ids, x_test, r_bins_edges, t_plt=[0, -1], saveas=None
):
    # Set up figure
    (fig, ax) = plt.subplots(
        nrows=len(t_plt),
        ncols=len(test_ids),
        figsize=(3 * len(test_ids), 2 * len(t_plt)),
        layout="constrained",
    )

    # Plot reconstruction for each test ID for multiple times
    for i, id in enumerate(test_ids):
        for j, t in enumerate(t_plt):
            ax[j][i].step(r_bins_edges, x_test[id, t])
            ax[j][i].step(
                r_bins_edges,
                model.decoder(
                    model.encoder(torch.Tensor(x_test[id, t]).reshape(1, 1, -1))
                )
                .detach()
                .numpy()[0, 0],
            )

            ax[j][i].set_xscale("log")
            ax[j][i].set_ylabel("PSD")
            ax[j][i].set_xlabel("r (m)")
        ax[0][i].set_title(f"Run #{id}")

    # Accoutrements
    ax[0][0].legend(["Data", "AE Reconstruction"])
    fig.suptitle("Reconstruction Demo: Out of Sample")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_predictions_AE_AR(
    model, test_ids, dsd_time, tplt, x_test, m_test, r_bins_edges, n_lag=1, saveas=None
):
    # Set up figure
    (fig, ax) = plt.subplots(
        ncols=len(test_ids),
        nrows=len(tplt),
        figsize=(3 * len(test_ids), 2 * len(tplt)),
        sharey=True,
        layout="constrained",
    )
    model.eval()

    # Plot predictions for AE-AR model for multiple time steps
    for i, id in enumerate(test_ids):
        x0 = x_test[id, :n_lag, :]
        m0 = m_test[id, 0]
        x_pred = np.zeros_like(x_test[id])
        x_pred[:n_lag, :] = (
            model.decoder(model.encoder(torch.Tensor(x0))).detach().numpy()
        )
        for t in range(n_lag, x_test.shape[1]):
            x_pred[t, :] = (
                model(
                    torch.Tensor(x_pred[t - n_lag : t, :]).reshape(
                        -1, n_lag, x_pred[t].shape[0]
                    ),
                    torch.Tensor([m0]).reshape(1, 1, 1),
                )
                .detach()
                .numpy()[0][0]
            )

        for j, t in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_test[id, t, :])
            ax[j][i].step(r_bins_edges, x_pred[t, :])

            ax[j][i].set_xscale("log")
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"Run #{id}")
        ax[-1][i].set_xlabel("radius (um)")

    # Accoutrements
    for j, t in enumerate(tplt):
        ax[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
    ax[1][0].legend(["Data", "Model"])
    fig.suptitle(
        f"VAE Autoregressive model, lag {n_lag}: Multi time step; out of sample"
    )

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_single_prediction_AE_AR(
    model, id, dsd_time, x_test, m_test, r_bins_edges, n_lag=1, saveas=None
):
    (fig, ax) = plt.subplots(
        1,
        1,
        figsize=(6, 3),
    )
    model.eval()
    x0 = x_test[id, :n_lag, :]
    m0 = m_test[id, 0]
    x_pred = np.zeros_like(x_test[id])
    x_pred[:n_lag, :] = model.decoder(model.encoder(torch.Tensor(x0))).detach().numpy()
    for t in range(n_lag, x_test.shape[1]):
        x_pred[t, :] = (
            model(
                torch.Tensor(x_pred[t - n_lag : t, :]).reshape(
                    -1, n_lag, x_pred[t].shape[0]
                ),
                torch.Tensor([m0]).reshape(1, 1, 1),
            )
            .detach()
            .numpy()[0][0]
        )

    l1 = ax.step(r_bins_edges, x_test[id, 0, :], label="t=0s, Data", color="grey")
    l2 = ax.step(r_bins_edges, x_test[id, -1, :], label=f"t={dsd_time[-1]}s Data")
    l3 = ax.step(
        r_bins_edges,
        x_pred[-1, :],
        ls="--",
        linewidth=3,
        label=f"t={dsd_time[-1]}s Model",
    )

    ax.set_xscale("log")
    ax.set_xlabel("radius (um)")

    ax.set_ylabel("PSD")
    ax.legend()
    plt.suptitle(
        f"VAE Autoregressive model, lag {n_lag}: Multi time step; out of sample"
    )
    if saveas is not None:
        plt.savefig(saveas)

    return fig


def plot_latent_trajectories(n_latent, dsd_time, z_pred, z_data, saveas=None):
    # Set up figure
    (fig, ax) = plt.subplots(
        nrows=2,
        ncols=n_latent + 1,
        figsize=(3 * (n_latent + 1), 6),
        sharey=False,
        sharex=True,
        layout="constrained",
    )
    colors = ["blue", "orange", "green", "pink", "purple", "gray"]

    for j in range(z_pred.shape[0]):
        for i in range(n_latent + 1):
            if i < n_latent:
                labeli = f"z{i}"
                color = colors[i]
            else:
                labeli = "M / dlnr"
                color = colors[-1]
            ax[0][i].plot(
                dsd_time,
                z_data[j, :, i],
                label=labeli,
                color=color,
                alpha=min(1, 150 / z_pred.shape[0]),
                lw=0.5,
            )
            ax[1][i].plot(
                dsd_time,
                z_pred[j, :, i],
                label=labeli,
                color=color,
                alpha=min(1, 150 / z_pred.shape[0]),
                lw=0.5,
            )
            ax[0][i].set_xlabel("Elapsed time")

    # Accoutrements
    for i in range(n_latent):
        ax[0][i].set_title(f"z{i + 1}")
        ax[0][i].set_xlim([0, dsd_time.max()])
    ax[0][-1].set_title("mass (rescaled)")
    ax[0][0].set_ylabel("Data")
    ax[1][0].set_ylabel("Model")
    fig.suptitle(f"Test set predicted Z(t)")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_single_latent_trajectory_AR(
    n_latent, model, dsd_time, x_test, m_test, j=0, n_lag=1, saveas=None
):
    (fig, ax) = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    colors = ["blue", "orange", "green", "pink", "purple", "gray"]
    x0 = x_test[j, :n_lag, :]
    mj = m_test[j, :]
    z0 = np.array(
        [
            model.encoder(torch.Tensor(x0[t]).reshape(1, -1)).detach().numpy()[0]
            for t in range(n_lag)
        ]
    )
    z_pred = np.zeros((x_test.shape[1], n_latent + 1))
    z_enc = np.zeros((x_test.shape[1], n_latent + 1))
    z_enc[:, -1] = mj
    z_enc[:n_lag, :-1] = z0
    z_pred[:n_lag, :-1] = z0
    z_pred[:n_lag, -1] = mj[0]
    for t in range(n_lag, x_test.shape[1]):
        lagged_input = torch.cat(
            (
                torch.Tensor(z_pred[t - n_lag : t, :-1]).reshape(n_lag * n_latent),
                torch.Tensor([mj[0]]),
            )
        )
        z_pred[t, :] = model.autoregressor(lagged_input).detach().numpy()
        z_enc[t, :-1] = (
            model.encoder(torch.Tensor(x_test[j, t, :]).reshape(1, -1))
            .detach()
            .numpy()[0]
        )

    for i in range(n_latent + 1):
        if i < n_latent:
            labeli = f"z{i}"
            color = colors[i]
        else:
            labeli = "M / dlnr"
            color = colors[-1]
        ax.plot(
            dsd_time,
            z_enc[:, i],
            label=labeli,
            color=color,
            lw=2,
        )
        ax.plot(
            dsd_time,
            z_pred[:, i],
            label=labeli + " pred",
            color=color,
            ls="--",
            lw=2,
        )
    ax.set_xlabel("Elapsed time")
    ax.set_ylabel("Latent variable value")
    ax.legend()
    ax.set_xlim([0, dsd_time.max()])
    plt.title(f"Autoregressive Z(t), lag {n_lag}")
    plt.show()


def plot_predictions_dzdt(
    test_ids,
    tplt,
    n_latent,
    model,
    dsd_time,
    x_test,
    m_test,
    x_train,
    m_train,
    r_bins_edges,
    saveas=None,
):
    # Set up figure
    (fig, ax) = plt.subplots(
        ncols=len(test_ids),
        nrows=len(tplt),
        figsize=(3 * len(test_ids), 2 * len(tplt)),
        sharey=True,
    )

    # Compute limits
    z_enc_train = model.encoder(torch.Tensor(x_train)).detach().numpy()
    zlim = np.zeros((n_latent + 1, 2))
    for il in range(n_latent):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m_train.min()
    zlim[-1][1] = m_train.max()

    # Compute all else
    z_encoded = model.encoder(torch.Tensor(x_test)).detach().numpy()
    for i, id in enumerate(test_ids):
        z0 = np.concatenate((z_encoded[id, 0, :], np.array([m_test[id, 0]])), axis=-1)
        latents_pred = du.simulate(z0, dsd_time[tplt], model.dzdt, zlim)
        x_pred = model.decoder(torch.Tensor(latents_pred[:, :-1])).detach().numpy()

        for j, t in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_test[id, t, :])
            ax[j][i].step(r_bins_edges, x_pred[j, :])

            ax[j][i].set_xscale("log")
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"Run #{id}")
        ax[-1][i].set_xlabel("radius (um)")

    # Accoutrements
    for j, t in enumerate(tplt):
        ax[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
    ax[1][0].legend(["Data", "Model"])
    fig.suptitle(f"Multi time step; out of sample")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def viz_3d_latent_space(model, x_test, time, saveas=None):
    # get latent space
    lsn = model.encoder(torch.Tensor(x_test)).detach().numpy()

    # Plot latent space
    pio.renderers.default = "browser"
    lsnp_data = lsn.reshape((-1, lsn.shape[-1]))
    x = lsnp_data[:, 0]
    if lsnp_data.shape[1] < 2:
        y = lsnp_data[:, 0] * 0.0
    else:
        y = lsnp_data[:, 1]
        if lsnp_data.shape[1] < 3:
            z = lsnp_data[:, 0] * 0.0
        else:
            z = lsnp_data[:, 2]
    color_values = (np.ones(lsn.shape[:2]) * time).ravel()
    tp_data = np.tile(np.arange(lsn.shape[1]), (lsn.shape[0], 1)).ravel()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                name="",
                marker=dict(
                    size=8,
                    color=color_values,
                    colorscale="Viridis",
                    opacity=0.7,
                    colorbar=dict(title="Time"),
                    line=dict(color="white", width=0.0),
                ),
                hovertemplate="LD1: %{x:.4f}<br>"
                + "LD2: %{y:.4f}<br>"
                + "LD3: %{z:.4f}<br>"
                + "Test Member: %{marker.color:d}<br>"
                + "Time Step: %{customdata}<br>",
                customdata=tp_data,
            )
        ]
    )
    fig.update_layout(
        title="Autoencoder Latent Space",
        width=1200,
        height=900,
        scene=dict(
            xaxis_title="Latent Dim 1",
            yaxis_title="Latent Dim 2",
            zaxis_title="Latent Dim 3",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    # Optional save
    if saveas is not None:
        fig.write_html(saveas)

    # Return fig for further manipulation
    return fig


def plot_full_testset_performance_recon(model, x_test, tol, saveas=None):
    # Extra vars
    n_test = x_test.shape[0]
    n_timesteps = x_test.shape[1]
    divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    # Determine best and worst performing members
    test_preds = model.decoder(model.encoder(torch.Tensor(x_test)))
    test_kl = np.zeros(x_test.shape[0:2])
    test_wass = np.zeros(x_test.shape[0:2])
    for nm in range(n_test):
        for nt in range(n_timesteps):
            pred_dist = test_preds[nm, nt]
            true_dist = torch.Tensor(x_test[nm, nt]).reshape(1, 1, -1)
            test_kl[nm, nt] = divergence(
                torch.log(pred_dist + tol),
                torch.log(true_dist + tol),
            )
            test_wass[nm, nt] = wasserstein_distance(
                pred_dist.detach().numpy().ravel(), true_dist.detach().numpy().ravel()
            )

    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(34, 5), layout="constrained")
    # ---
    ax = axes[0]
    klm = ax.matshow(np.log10(test_kl.T), vmin=-5, vmax=-2)
    fig.colorbar(
        klm,
        ax=ax,
        location="top",
        label=f"log10(KL Divergence) (Mean={np.mean(np.log10(test_kl)):.2f})",
        extend="both",
    )
    ax.set_ylabel(f"Time")
    # ---
    ax = axes[1]
    wsm = ax.matshow(test_wass.T, vmin=0.0005, vmax=0.008)
    fig.colorbar(
        wsm,
        ax=ax,
        location="top",
        label=f"Wasserstein Distance (Mean={np.mean(test_wass):.2e})",
        extend="both",
    )
    ax.set_xlabel(f"Test Member")
    ax.set_ylabel(f"Time")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_full_testset_performance_pred(test_kl, test_wass, test_wass_un, saveas=None):
    # Plot
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(34, 8), layout="constrained")
    # ---
    ax = axes[0]
    klm = ax.matshow(np.log10(test_kl.T), vmin=-5, vmax=-2)
    fig.colorbar(
        klm,
        ax=ax,
        location="top",
        label=f"log10(KL Divergence) (Mean={np.mean(np.log10(test_kl)):.2f})",
        extend="both",
    )
    ax.set_ylabel(f"Time")
    # ---
    ax = axes[1]
    wsm = ax.matshow(test_wass.T, vmin=0.0005, vmax=0.008)
    fig.colorbar(
        wsm,
        ax=ax,
        location="top",
        label=f"Wasserstein Distance (Mean={np.mean(test_wass):.2e})",
        extend="both",
    )
    ax.set_xlabel(f"Test Member")
    ax.set_ylabel(f"Time")
    # ---
    ax = axes[2]
    wsm = ax.matshow(test_wass_un.T, vmin=0.0005, vmax=0.008)
    fig.colorbar(
        wsm,
        ax=ax,
        location="top",
        label=f"Unnormalized Wasserstein Distance (Mean={np.mean(test_wass_un):.2e})",
        extend="both",
    )
    ax.set_xlabel(f"Test Member")
    ax.set_ylabel(f"Time")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_testset_quantiles_pred(
    x_test,
    test_preds,
    test_metric,
    tplt,
    dsd_time,
    r_bins_edges,
    qtiles=[0, 0.25, 0.5, 0.75, 0.9999],
    saveas=None,
):
    n_test = x_test.shape[0]
    test_metric_timemean = np.mean(test_metric, axis=1)
    tm_argsort = np.argsort(-test_metric_timemean)
    qtile_idx = (np.array(qtiles) * n_test).astype(int)
    qtile_mems = tm_argsort[qtile_idx]

    # Set up figure
    (fig, ax) = plt.subplots(
        ncols=len(qtile_mems),
        nrows=len(tplt),
        figsize=(3 * len(qtile_mems), 2 * len(tplt)),
        sharey=True,
    )

    for i, id in enumerate(qtile_mems):
        for j, t in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_test[id, t, :])
            ax[j][i].step(r_bins_edges, test_preds[id, t, :].detach().numpy())

            ax[j][i].set_xscale("log")
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"{qtiles[i] * 100}th percentile")
        ax[-1][i].set_xlabel("radius (um)")

    # Accoutrements
    for j, t in enumerate(tplt):
        ax[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
    ax[1][0].legend(["Data", "Model"])
    fig.suptitle(f"Multi time step; out of sample; percentiles")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig

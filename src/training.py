import torch
from src import data_utils as du, models


# import pysindy as ps


# Losses
def recon_loss(recon_x, x):
    mseloss = torch.nn.MSELoss()
    return mseloss(recon_x, x)


def l1_loss(x):
    tmp = torch.zeros_like(x, requires_grad=True)
    l1 = torch.nn.L1Loss()
    loss = l1(x, tmp)
    return loss


# Functions for training end-to-end network + SINDy
def train_network_e2e(
    train_dataloader, params, val_dataloader=None, device="cpu", X=None, T=None
):
    device = torch.device(device)

    if params["CNN"]:
        autoencoder_network = models.CNNAutoEncoder(
            n_channels=1, n_bins=params["input_dim"], n_latent=params["latent_dim"]
        )
    else:
        autoencoder_network = models.FFNNAutoEncoder(
            n_bins=params["input_dim"], n_latent=params["latent_dim"]
        )

    num_params = models.count_parameters(autoencoder_network)
    (encoder_weights, encoder_biases) = autoencoder_network.encoder.get_weights()
    (decoder_weights, decoder_biases) = autoencoder_network.decoder.get_weights()
    autoencoder_network.to(device)
    print(f"Autoencoder has {num_params} trainable parameters")

    library_size = du.library_size(params["latent_dim"], params["poly_order"])
    sindy_coeffs_tensor = torch.empty(
        (params["latent_dim"], library_size), requires_grad=True
    )
    # torch.nn.init.constant_(sindy_coeffs_tensor, 0.0)
    torch.nn.init.normal_(sindy_coeffs_tensor)
    print(f"SINDy has {torch.numel(sindy_coeffs_tensor)} trainable parameters")

    optimizer = torch.optim.Adam(
        [
            sindy_coeffs_tensor,
            *encoder_weights,
            *encoder_biases,
            *decoder_weights,
            *decoder_biases,
        ],
        lr=params["learning_rate"],
    )
    early_stopping = EarlyStopping(patience=params["patience"], verbose=True)

    train_loss = []
    train_losses = {"recon": [], "sindy_z": [], "sindy_x": [], "sindy_reg": []}
    if val_dataloader is not None:
        val_loss = []
        val_losses = {"recon": [], "sindy_z": [], "sindy_x": [], "sindy_reg": []}

    printerval = 10

    print("PRETRAINING")
    ref_params = params.copy()
    ref_params["loss_weight_sindy_reg"] = 0.0
    ref_params["loss_weight_sindy_x"] = 0.0
    ref_params["loss_weight_sindy_z"] = 0.0
    for epoch in range(params["pretraining_epochs"]):
        (epoch_loss, epoch_losses) = train_e2e(
            optimizer,
            train_dataloader,
            ref_params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            sindy_coeffs_tensor,
            autoencoder_network,
            device,
            cnn=params["CNN"],
        )
        train_loss.append(epoch_loss)
        for key in epoch_losses.keys():
            train_losses[key].append(epoch_losses[key])

        (val_epoch_loss, val_epoch_losses) = test_e2e(
            train_dataloader,
            ref_params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            sindy_coeffs_tensor,
            autoencoder_network,
            device,
            cnn=params["CNN"],
        )

        val_loss.append(val_epoch_loss)
        for key in val_epoch_losses.keys():
            val_losses[key].append(val_epoch_losses[key])

        if epoch % printerval == 0:
            print(
                f"\n Epoch: {epoch:03d}, \n Train MSE: {epoch_loss:.8f} | Val MSE: {val_epoch_loss:.8f}"
            )
            for key in epoch_losses.keys():
                print(f"{key}: {epoch_losses[key]} | {val_epoch_losses[key]}")

    ##########################
    if X is not None and T is not None:
        print("Re-initialized SINDy coefficients with provided data")
        sindy_coeffs_tensor = torch.tensor(
            initialize_sindy(autoencoder_network, params, X, T)
        ).float()

    printerval = 1 if params["CNN"] else 10
    reset_optimizer_state(optimizer)
    early_stopping = EarlyStopping(patience=params["patience"], verbose=True)

    print("\n TRAINING")
    for epoch in range(params["training_epochs"]):
        (epoch_loss, epoch_losses) = train_e2e(
            optimizer,
            train_dataloader,
            params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            sindy_coeffs_tensor,
            autoencoder_network,
            device,
            cnn=params["CNN"],
        )
        train_loss.append(epoch_loss)
        for key in epoch_losses.keys():
            train_losses[key].append(epoch_losses[key])

        (val_epoch_loss, val_epoch_losses) = test_e2e(
            train_dataloader,
            params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            sindy_coeffs_tensor,
            autoencoder_network,
            device,
            cnn=params["CNN"],
        )

        val_loss.append(val_epoch_loss)
        for key in val_epoch_losses.keys():
            val_losses[key].append(val_epoch_losses[key])

        if epoch % printerval == 0:
            print(
                f"\n Epoch: {epoch:03d}, \n Train MSE: {epoch_loss:.8f} | Val MSE: {val_epoch_loss:.8f}"
            )
            for key in epoch_losses.keys():
                print(f"{key}: {epoch_losses[key]} | {val_epoch_losses[key]}")

        # Check early stopping
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

    ##########################
    print("\n REFINEMENT")
    ref_params = params.copy()
    ref_params["loss_weight_sindy_reg"] = 0.0
    reset_optimizer_state(optimizer)
    early_stopping = EarlyStopping(patience=params["patience"], verbose=True)

    for epoch in range(params["refinement_epochs"]):
        (epoch_loss, epoch_losses) = train_e2e(
            optimizer,
            train_dataloader,
            ref_params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            sindy_coeffs_tensor,
            autoencoder_network,
            device,
            cnn=params["CNN"],
        )

        train_loss.append(epoch_loss)
        for key in epoch_losses.keys():
            train_losses[key].append(epoch_losses[key])

        (val_epoch_loss, val_epoch_losses) = test_e2e(
            train_dataloader,
            params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            sindy_coeffs_tensor,
            autoencoder_network,
            device,
            cnn=params["CNN"],
        )
        val_loss.append(val_epoch_loss)
        for key in val_epoch_losses.keys():
            val_losses[key].append(val_epoch_losses[key])

        if epoch % printerval == 0:
            print(
                f"\n Epoch: {epoch:03d}, \n Train MSE: {epoch_loss:.8f} | Val MSE: {val_epoch_loss:.8f}"
            )
            for key in epoch_losses.keys():
                print(f"{key}: {epoch_losses[key]} | {val_epoch_losses[key]}")

        # Check early stopping
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

    return (
        autoencoder_network,
        sindy_coeffs_tensor,
        train_loss,
        train_losses,
        val_loss,
        val_losses,
    )


# Functions for training end-to-end network + black box time tendencies
def train_network_e2e_bb(
    train_dataloader, params, val_dataloader=None, device="cpu", X=None, T=None
):
    device = torch.device(device)

    if params["CNN"]:
        autoencoder_network = models.CNNAutoEncoder(
            n_channels=1, n_bins=params["input_dim"], n_latent=params["latent_dim"]
        )
    else:
        autoencoder_network = models.FFNNAutoEncoder(
            n_bins=params["input_dim"], n_latent=params["latent_dim"]
        )

    derivative_network = models.NNDerivatives(
        n_latent=params["latent_dim"], layer_size=params["layers"]
    )

    num_params = models.count_parameters(autoencoder_network)
    np_deriv = models.count_parameters(derivative_network)
    (encoder_weights, encoder_biases) = autoencoder_network.encoder.get_weights()
    (decoder_weights, decoder_biases) = autoencoder_network.decoder.get_weights()
    (deriv_weights, deriv_biases) = derivative_network.get_weights()
    autoencoder_network.to(device)
    derivative_network.to(device)
    print(f"Autoencoder has {num_params} trainable parameters")
    print(f"Derivative network has {np_deriv} trainable parameters")

    optimizer = torch.optim.Adam(
        [
            *deriv_weights,
            *deriv_biases,
            *encoder_weights,
            *encoder_biases,
            *decoder_weights,
            *decoder_biases,
        ],
        lr=params["learning_rate"],
    )
    early_stopping = EarlyStopping(patience=params["patience"], verbose=True)

    train_loss = []
    train_losses = {"recon": [], "sindy_z": [], "sindy_x": [], "sindy_reg": []}
    if val_dataloader is not None:
        val_loss = []
        val_losses = {"recon": [], "sindy_z": [], "sindy_x": [], "sindy_reg": []}

    printerval = 10

    print("PRETRAINING")
    ref_params = params.copy()
    ref_params["loss_weight_sindy_reg"] = 0.0
    ref_params["loss_weight_sindy_x"] = 0.0
    ref_params["loss_weight_sindy_z"] = 0.0
    for epoch in range(params["pretraining_epochs"]):
        (epoch_loss, epoch_losses) = train_e2e_bb(
            optimizer,
            train_dataloader,
            ref_params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            deriv_weights,
            deriv_biases,
            autoencoder_network,
            derivative_network,
            device,
            cnn=params["CNN"],
        )
        train_loss.append(epoch_loss)
        for key in epoch_losses.keys():
            train_losses[key].append(epoch_losses[key])

        (val_epoch_loss, val_epoch_losses) = test_e2e_bb(
            train_dataloader,
            ref_params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            deriv_weights,
            deriv_biases,
            autoencoder_network,
            derivative_network,
            device,
            cnn=params["CNN"],
        )

        val_loss.append(val_epoch_loss)
        for key in val_epoch_losses.keys():
            val_losses[key].append(val_epoch_losses[key])

        if epoch % printerval == 0:
            print(
                f"\n Epoch: {epoch:03d}, \n Train MSE: {epoch_loss:.8f} | Val MSE: {val_epoch_loss:.8f}"
            )
            for key in epoch_losses.keys():
                print(f"{key}: {epoch_losses[key]} | {val_epoch_losses[key]}")

    printerval = 1 if params["CNN"] else 10
    early_stopping = EarlyStopping(patience=params["patience"], verbose=True)

    print("\n TRAINING")
    for epoch in range(params["training_epochs"]):
        (epoch_loss, epoch_losses) = train_e2e_bb(
            optimizer,
            train_dataloader,
            params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            deriv_weights,
            deriv_biases,
            autoencoder_network,
            derivative_network,
            device,
            cnn=params["CNN"],
        )
        train_loss.append(epoch_loss)
        for key in epoch_losses.keys():
            train_losses[key].append(epoch_losses[key])

        (val_epoch_loss, val_epoch_losses) = test_e2e_bb(
            train_dataloader,
            params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            deriv_weights,
            deriv_biases,
            autoencoder_network,
            derivative_network,
            device,
            cnn=params["CNN"],
        )

        val_loss.append(val_epoch_loss)
        for key in val_epoch_losses.keys():
            val_losses[key].append(val_epoch_losses[key])

        if epoch % printerval == 0:
            print(
                f"\n Epoch: {epoch:03d}, \n Train MSE: {epoch_loss:.8f} | Val MSE: {val_epoch_loss:.8f}"
            )
            for key in epoch_losses.keys():
                print(f"{key}: {epoch_losses[key]} | {val_epoch_losses[key]}")

        # Check early stopping
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

    return (
        autoencoder_network,
        derivative_network,
        train_loss,
        train_losses,
        val_loss,
        val_losses,
    )


def train_e2e(
    optimizer,
    train_dataloader,
    params,
    encoder_weights,
    encoder_biases,
    decoder_weights,
    decoder_biases,
    sindy_coeffs_tensor,
    autoencoder_network,
    device,
    cnn=False,
):
    # autoencoder_network.train()
    epoch_loss = 0.0
    epoch_losses = {"recon": 0.0, "sindy_z": 0.0, "sindy_x": 0.0, "sindy_reg": 0.0}

    for batch, train_data in enumerate(train_dataloader):
        if "erf_data" in params.keys():
            (x_data, _, _, dx_data, _) = train_data
        else:
            (x_data, dx_data) = train_data
        # (x_data, dx_data) = train_dataloader
        optimizer.zero_grad(set_to_none=True)
        x_data = x_data.to(device)
        dx_data = dx_data.to(device)

        loss, losses = loss_fn_e2e(
            x_data,
            dx_data,
            params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            sindy_coeffs_tensor,
            autoencoder_network,
            device,
            cnn=cnn,
        )

        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss.item()
        for key in losses.keys():
            epoch_losses[key] += losses[key].item()

    epoch_loss /= len(train_dataloader)
    for key in losses.keys():
        epoch_losses[key] /= len(train_dataloader)

    return (epoch_loss, epoch_losses)


def train_e2e_bb(
    optimizer,
    train_dataloader,
    params,
    encoder_weights,
    encoder_biases,
    decoder_weights,
    decoder_biases,
    deriv_weights,
    deriv_biases,
    autoencoder_network,
    derivative_network,
    device,
    cnn=False,
):
    # autoencoder_network.train()
    epoch_loss = 0.0
    epoch_losses = {"recon": 0.0, "sindy_z": 0.0, "sindy_x": 0.0, "sindy_reg": 0.0}

    for batch, (x_data, dx_data) in enumerate(train_dataloader):
        # (x_data, dx_data) = train_dataloader
        optimizer.zero_grad(set_to_none=True)
        x_data = x_data.to(device)
        dx_data = dx_data.to(device)

        loss, losses = loss_fn_e2e_bb(
            x_data,
            dx_data,
            params,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            deriv_weights,
            deriv_biases,
            autoencoder_network,
            derivative_network,
            device,
            cnn=cnn,
        )

        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss.item()
        for key in losses.keys():
            epoch_losses[key] += losses[key].item()

    epoch_loss /= len(train_dataloader)
    for key in losses.keys():
        epoch_losses[key] /= len(train_dataloader)

    return (epoch_loss, epoch_losses)


def test_e2e_bb(
    val_dataloader,
    params,
    encoder_weights,
    encoder_biases,
    decoder_weights,
    decoder_biases,
    deriv_weights,
    deriv_biases,
    autoencoder_network,
    derivative_network,
    device,
    cnn=False,
):
    autoencoder_network.eval()
    derivative_network.eval()
    epoch_loss = 0.0
    epoch_losses = {"recon": 0.0, "sindy_z": 0.0, "sindy_x": 0.0, "sindy_reg": 0.0}

    with torch.no_grad():
        for batch, (x_data, dx_data) in enumerate(val_dataloader):
            x_data = x_data.to(device)
            dx_data = dx_data.to(device)

            loss, losses = loss_fn_e2e_bb(
                x_data,
                dx_data,
                params,
                encoder_weights,
                encoder_biases,
                decoder_weights,
                decoder_biases,
                deriv_weights,
                deriv_biases,
                autoencoder_network,
                derivative_network,
                device,
                cnn=cnn,
            )

            epoch_loss += loss.item()
            for key in losses.keys():
                epoch_losses[key] += losses[key].item()

    epoch_loss /= len(val_dataloader)
    for key in losses.keys():
        epoch_losses[key] /= len(val_dataloader)

    return (epoch_loss, epoch_losses)


def test_e2e(
    val_dataloader,
    params,
    encoder_weights,
    encoder_biases,
    decoder_weights,
    decoder_biases,
    sindy_coeffs_tensor,
    autoencoder_network,
    device,
    cnn=False,
):
    autoencoder_network.eval()
    epoch_loss = 0.0
    epoch_losses = {"recon": 0.0, "sindy_z": 0.0, "sindy_x": 0.0, "sindy_reg": 0.0}

    with torch.no_grad():
        # (x_data, dx_data) = val_dataloader
        for batch, val_data in enumerate(val_dataloader):
            if "erf_data" in params.keys():
                (x_data, _, _, dx_data, _) = val_data
            else:
                (x_data, dx_data) = val_data
            x_data = x_data.to(device)
            dx_data = dx_data.to(device)

            loss, losses = loss_fn_e2e(
                x_data,
                dx_data,
                params,
                encoder_weights,
                encoder_biases,
                decoder_weights,
                decoder_biases,
                sindy_coeffs_tensor,
                autoencoder_network,
                device,
                cnn=cnn,
            )

            epoch_loss += loss.item()
            for key in losses.keys():
                epoch_losses[key] += losses[key].item()

    epoch_loss /= len(val_dataloader)
    for key in losses.keys():
        epoch_losses[key] /= len(val_dataloader)

    return (epoch_loss, epoch_losses)


# def initialize_sindy(vae, params, X, T):
#     optimizer = ps.SR3(
#         threshold=1e-1, thresholder="l1", max_iter=1000, normalize_columns=False, tol=1e-1
#         )
#     sindy_model = ps.SINDy(
#         optimizer=optimizer,
#         feature_library=ps.PolynomialLibrary(2)
#     )
#     z_encoded = vae.encoder(torch.tensor(X).reshape(-1, 1, params["input_dim"]))
#     z_encoded = z_encoded.reshape(params["n_runs"], -1, params["latent_dim"]).detach().numpy()
#     sindy_model.fit(z_encoded, t=T)
#     sindy_model.print()
#
#     return optimizer.coef_


def loss_fn_e2e(
    x_data,
    dx_data,
    params,
    encoder_weights,
    encoder_biases,
    decoder_weights,
    decoder_biases,
    sindy_coeffs,
    autoencoder_network,
    device="cpu",
    cnn=False,
):
    # first update the weights & coefficients
    autoencoder_network.encoder.set_weights(encoder_weights, encoder_biases)
    autoencoder_network.decoder.set_weights(decoder_weights, decoder_biases)

    # set up the loss function
    losses = {}
    x = x_data.to(device)
    dx = dx_data.to(device)

    # reconstruction loss
    x_recon = autoencoder_network(x)
    losses["recon"] = recon_loss(x, x_recon)

    if (params["loss_weight_sindy_z"] + params["loss_weight_sindy_x"]) > 0.0:
        # sindy_dz
        xx = x.clone().detach().requires_grad_()
        z = autoencoder_network.encoder(xx)
        if cnn:
            gradient_x = torch.func.vmap(
                torch.func.jacrev(autoencoder_network.encoder, chunk_size=20),
                chunk_size=20,
            )(xx)[:, 0, :, :, 0, :]
        else:
            gradient_x = torch.func.vmap(
                torch.func.vmap(
                    torch.func.jacrev(autoencoder_network.encoder, chunk_size=20),
                    chunk_size=20,
                ),
                chunk_size=20,
            )(xx)

        dz = torch.einsum("abcd, abd->abc", gradient_x, dx)
        del gradient_x

        sindy_library = du.sindy_library_tensor(
            z, params["latent_dim"], params["poly_order"]
        ).to(device)
        sindy_coeffs_data = sindy_coeffs.to(device)

        dz_sindy = torch.matmul(sindy_library, sindy_coeffs_data.T)
        losses["sindy_z"] = recon_loss(dz, dz_sindy)
        del sindy_library, dz

        if params["loss_weight_sindy_x"] > 0.0:
            # sindy_dx
            z = z.clone().detach().requires_grad_()
            if cnn:
                gradient_z = torch.func.vmap(
                    torch.func.vmap(
                        torch.func.jacrev(autoencoder_network.decoder, chunk_size=20),
                        chunk_size=20,
                    ),
                    chunk_size=20,
                )(z)[:, 0, 0, :, :, :]
            else:
                gradient_z = torch.func.vmap(
                    torch.func.vmap(torch.func.jacrev(autoencoder_network.decoder))
                )(z)

            dx_recon = torch.einsum("abcd, abd->abc", gradient_z, dz_sindy)
            del z, gradient_z

        losses["sindy_x"] = recon_loss(dx, dx_recon)
        del dx, dx_recon, dz_sindy

        # sindy reg
        losses["sindy_reg"] = l1_loss(sindy_coeffs_data)
        del sindy_coeffs_data

    loss = 0.0
    for i, key in enumerate(losses.keys()):
        loss += losses[key] * params["loss_weight_" + key]

    return loss, losses


def loss_fn_e2e_bb(
    x_data,
    dx_data,
    params,
    encoder_weights,
    encoder_biases,
    decoder_weights,
    decoder_biases,
    deriv_weights,
    deriv_biases,
    autoencoder_network,
    derivative_network,
    device="cpu",
    cnn=False,
):
    # first update the weights & coefficients
    autoencoder_network.encoder.set_weights(encoder_weights, encoder_biases)
    autoencoder_network.decoder.set_weights(decoder_weights, decoder_biases)
    derivative_network.set_weights(deriv_weights, deriv_biases)

    # set up the loss function
    losses = {}
    x = x_data.to(device)
    dx = dx_data.to(device)

    # reconstruction loss
    x_recon = autoencoder_network(x)
    losses["recon"] = recon_loss(x, x_recon)

    if (params["loss_weight_sindy_z"] + params["loss_weight_sindy_x"]) > 0.0:
        # sindy_dz
        xx = x.clone().detach().requires_grad_()
        z = autoencoder_network.encoder(xx)
        if cnn:
            gradient_x = torch.func.vmap(
                torch.func.jacrev(autoencoder_network.encoder, chunk_size=20),
                chunk_size=20,
            )(xx)[:, 0, :, :, 0, :]
        else:
            gradient_x = torch.func.vmap(
                torch.func.vmap(
                    torch.func.jacrev(autoencoder_network.encoder, chunk_size=20),
                    chunk_size=20,
                ),
                chunk_size=20,
            )(xx)

        dz = torch.einsum("abcd, abd->abc", gradient_x, dx)
        del gradient_x

        dzdt = derivative_network(z)
        losses["sindy_z"] = recon_loss(dz, dzdt)

        if params["loss_weight_sindy_x"] > 0.0:
            # sindy_dx
            z = z.clone().detach().requires_grad_()
            if cnn:
                gradient_z = torch.func.vmap(
                    torch.func.vmap(
                        torch.func.jacrev(autoencoder_network.decoder, chunk_size=20),
                        chunk_size=20,
                    ),
                    chunk_size=20,
                )(z)[:, 0, 0, :, :, :]
            else:
                gradient_z = torch.func.vmap(
                    torch.func.vmap(torch.func.jacrev(autoencoder_network.decoder))
                )(z)

            dx_recon = torch.einsum("abcd, abd->abc", gradient_z, dzdt)
            del z, gradient_z, dzdt

            losses["sindy_x"] = recon_loss(dx, dx_recon)
            del dx, dx_recon

    loss = 0.0
    for i, key in enumerate(losses.keys()):
        loss += losses[key] * params["loss_weight_" + key]

    return loss, losses


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")


def reset_optimizer_state(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


# Functions for standalone training of network (i.e. no timeseries, not end-to-end)
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    for bindata in dataloader:
        optimizer.zero_grad(set_to_none=True)
        bindata = bindata.to(device)
        recon = model(bindata.float())
        loss = loss_fn(recon, bindata.float())

        # backprop
        loss.backward()
        optimizer.step()


def test(model, dataloader, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for bindata in dataloader:
            bindata = bindata.to(device)
            recon = model(bindata.float())
            test_loss += loss_fn(recon.float(), bindata.float())

    test_loss /= num_batches
    return test_loss

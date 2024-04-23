import sys;

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import torch
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import src.utils.utils as util_general
from src.utils.utils import save_checkpoint
from torchvision.utils import save_image
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure
from sys import platform

# x = low energy
# y = recombined
def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, device, cfg_trainer):
    gen.train()
    disc.train()

    loop = tqdm(loader, leave=True)

    for idx, (x, y, id) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * cfg_trainer["L1_LAMBDA"]
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

    return disc, gen


def eval_fn(gen, loader, criterion, device, outputs_dir):
    gen.eval()

    loop = tqdm(loader, leave=True)
    running_loss = 0.0
    for idx, (x, y, file_names) in enumerate(loop):  # ImgDataset restituisce prima LE e poi RECO
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast():
            y_fake = gen(x)

            if outputs_dir:
                for output, file_name in zip(y_fake, file_names):
                    # SAVE OUTPUTS PNG
                    filename_output = "%s_output.png" % (file_name)
                    save_image(output, os.path.join(outputs_dir, filename_output))

        loss = criterion(y_fake, y)
        # statistics
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(loader)
    return epoch_loss


def train_pix2pix(gen, disc, data_loaders, early_stopping_criterion, opt_disc, opt_gen, L1, bce, model_fold_dir,
                  cfg_trainer, device):
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    best_gen_wts = copy.deepcopy(gen.state_dict())
    best_disc_wts = copy.deepcopy(disc.state_dict())
    best_loss = np.Inf

    history = {'val_loss': []}

    epochs_no_improve = 0
    early_stop = False
    for epoch in range(cfg_trainer["max_epochs"]):

        # Train one epoch
        disc, gen = train_fn(disc=disc, gen=gen, loader=data_loaders['train'], opt_disc=opt_disc, opt_gen=opt_gen,
                             l1_loss=L1,
                             bce=bce, g_scaler=g_scaler, d_scaler=d_scaler, device=device, cfg_trainer=cfg_trainer)

        # Val
        epoch_loss = eval_fn(gen=gen, loader=data_loaders['val'], criterion=early_stopping_criterion, device=device,
                             outputs_dir=False)

        history['val_loss'].append(epoch_loss)
        if epoch > cfg_trainer['warm_up']:
            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_gen_wts = copy.deepcopy(gen.state_dict())
                best_disc_wts = copy.deepcopy(disc.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= cfg_trainer["early_stopping"]:
                    print(f'\nEarly Stopping! Total epochs: {epoch}%')
                    early_stop = True
            if early_stop:
                break

    print('Training complete')
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Loss: {:4f}'.format(best_loss))

    gen.load_state_dict(best_gen_wts)
    disc.load_state_dict(best_disc_wts)

    # Save model
    save_checkpoint(gen, opt_gen, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN"]))
    save_checkpoint(disc, opt_disc, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_DISC"]))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return gen, disc, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)
    # Training results Loss function
    plt.figure(figsize=(8, 6))
    for c in ['val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(model_plot_dir, "Loss"))


def test_4metrics(gen, loader_train, loader_val, loader_test, mse_metric, psnr_metric, device, outputs_dir,
                  save_output):
    gen.eval()
    loop_test = tqdm(loader_test, leave=True)
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    running_loss4 = 0.0

    if loader_train:
        loop_train = tqdm(loader_train, leave=True)
        with torch.no_grad():
            for idx, (x, y, file_names) in enumerate(loop_train):  # ImgDataset restituisce prima LE e poi RECO
                x = x.to(device)
                with torch.cuda.amp.autocast():
                    y_fake = gen(x)
                    if save_output:
                        for output, file_name in zip(y_fake, file_names):
                            filename_output = "%s_output.png" % (file_name)
                            save_image(output, os.path.join(outputs_dir, filename_output))

    if loader_val:
        loop_val = tqdm(loader_val, leave=True)
        with torch.no_grad():
            for idx, (x, y, file_names) in enumerate(loop_val):
                x = x.to(device)
                with torch.cuda.amp.autocast():
                    y_fake = gen(x)
                    if save_output:
                        for output, file_name in zip(y_fake, file_names):
                            filename_output = "%s_output.png" % (file_name)
                            save_image(output, os.path.join(outputs_dir, filename_output))

    with torch.no_grad():
        for idx, (x, y, file_names) in enumerate(loop_test):
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                if save_output:
                    for output, file_name in zip(y_fake, file_names):  # SAVE OUTPUTS PNG
                        filename_output = "%s_output.png" % (file_name)
                        save_image(output, os.path.join(outputs_dir, filename_output))

            loss1 = mse_metric(y_fake, y)
            loss2 = psnr_metric(y_fake, y)
            if platform == 'win32':  # se sono su windows
                loss3 = vif_p(y_fake, y)
                loss4 = structural_similarity_index_measure(y_fake, y)
            else: # se sono su alvis
                loss3 = vif_p(y_fake.float(), y.float())
                loss4 = structural_similarity_index_measure(y_fake.to(torch.float16), y.to(torch.float16))

            # statistics
            running_loss1 += loss1.item() * x.size(0)
            running_loss2 += loss2.item() * x.size(0)
            running_loss3 += loss3.item() * x.size(0)
            running_loss4 += loss4.item() * x.size(0)

    epoch_loss1 = running_loss1 / len(loader_test)
    epoch_loss2 = running_loss2 / len(loader_test)
    epoch_loss3 = running_loss3 / len(loader_test)
    epoch_loss4 = running_loss4 / len(loader_test)

    return epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4

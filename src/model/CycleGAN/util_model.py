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
from torchvision.utils import save_image
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure
import src.utils.utils as util_general
from src.utils.utils import save_checkpoint
from sys import platform

def train_fn(disc_REC, disc_LE, gen_LE, gen_REC, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, device, cfg_trainer):
    disc_REC.train()
    disc_LE.train()
    gen_LE.train()
    gen_REC.train()
    REC_reals = 0
    REC_fakes = 0
    loop = tqdm(loader, leave=True)
    for idx, (lowenergy, recombined, id) in enumerate(loop):  # ImgDataset restituisce prima LE e poi RECO
        lowenergy = lowenergy.to(device)
        recombined = recombined.to(device)
        # Train Discriminators
        with torch.cuda.amp.autocast():
            fake_recombined = gen_REC(lowenergy)
            D_REC_real = disc_REC(recombined)
            D_REC_fake = disc_REC(fake_recombined.detach())
            REC_reals += D_REC_real.mean().item()
            REC_fakes += D_REC_fake.mean().item()
            D_REC_real_loss = mse(D_REC_real, torch.ones_like(D_REC_real))  # real = 1
            D_REC_fake_loss = mse(D_REC_fake, torch.zeros_like(D_REC_fake))  # fake = 0
            D_REC_loss = D_REC_real_loss + D_REC_fake_loss

            fake_lowenergy = gen_LE(recombined)
            D_LE_real = disc_LE(lowenergy)
            D_LE_fake = disc_LE(fake_lowenergy.detach())
            D_LE_real_loss = mse(D_LE_real, torch.ones_like(D_LE_real))
            D_LE_fake_loss = mse(D_LE_fake, torch.zeros_like(D_LE_fake))
            D_LE_loss = D_LE_real_loss + D_LE_fake_loss

            # put it together
            D_loss = (D_REC_loss + D_LE_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_REC_fake = disc_REC(fake_recombined)
            D_LE_fake = disc_LE(fake_lowenergy)
            loss_G_REC = mse(D_REC_fake, torch.ones_like(D_REC_fake))
            loss_G_LE = mse(D_LE_fake, torch.ones_like(D_LE_fake))

            # cycle loss
            cycle_lowenergy = gen_LE(fake_recombined)
            cycle_recombined = gen_REC(fake_lowenergy)
            cycle_lowenergy_loss = l1(lowenergy, cycle_lowenergy)
            cycle_recombined_loss = l1(recombined, cycle_recombined)

            # identity loss
            identity_lowenergy = gen_LE(lowenergy)
            identity_recombined = gen_REC(recombined)
            identity_lowenergy_loss = l1(lowenergy, identity_lowenergy)
            identity_recombined_loss = l1(recombined, identity_recombined)

            # add all together
            G_loss = (loss_G_LE + loss_G_REC
                      + cycle_lowenergy_loss * cfg_trainer["LAMBDA_CYCLE"] + cycle_recombined_loss * cfg_trainer["LAMBDA_CYCLE"]
                      + identity_recombined_loss * cfg_trainer["LAMBDA_IDENTITY"] + identity_lowenergy_loss * cfg_trainer["LAMBDA_IDENTITY"])

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(REC_real=REC_reals / (idx + 1), REC_fake=REC_fakes / (idx + 1))

    return disc_REC, disc_LE, gen_LE, gen_REC


def eval_fn(gen_REC, loader, criterion, device, outputs_dir):
    gen_REC.eval()
    loop = tqdm(loader, leave=True)
    running_loss = 0.0
    for idx, (lowenergy, recombined, file_names) in enumerate(loop):
        lowenergy = lowenergy.to(device)
        recombined = recombined.to(device)
        with torch.cuda.amp.autocast():
            fake_recombined = gen_REC(lowenergy.float())

            if outputs_dir:
                for output, file_name in zip(fake_recombined, file_names):
                    # SAVE OUTPUTS PNG
                    filename_output = "%s_output.png" % (file_name)
                    save_image(output, os.path.join(outputs_dir, filename_output))

        loss = criterion(fake_recombined, recombined)
        # statistics
        running_loss += loss.item() * lowenergy.size(0)
    epoch_loss = running_loss / len(loader)
    return epoch_loss


def train_cycle_gan(gen_REC, gen_LE, disc_REC, disc_LE, data_loaders, early_stopping_criterion, opt_disc, opt_gen, L1,
                    mse, model_fold_dir, cfg_trainer, device):
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    best_gen_REC_wts = copy.deepcopy(gen_REC.state_dict())
    best_gen_LE_wts = copy.deepcopy(gen_LE.state_dict())
    best_disc_REC_wts = copy.deepcopy(disc_REC.state_dict())
    best_disc_LE_wts = copy.deepcopy(disc_LE.state_dict())
    best_loss = np.Inf

    history = {'val_loss': []}

    epochs_no_improve = 0
    early_stop = False
    for epoch in range(cfg_trainer["max_epochs"]):
        # Train one epoch
        disc_REC, disc_LE, gen_LE, gen_REC = train_fn(disc_REC=disc_REC, disc_LE=disc_LE, gen_LE=gen_LE,
                                                      gen_REC=gen_REC,
                                                      loader=data_loaders['train'], opt_disc=opt_disc, opt_gen=opt_gen,
                                                      l1=L1,
                                                      mse=mse, d_scaler=d_scaler, g_scaler=g_scaler, device=device,
                                                      cfg_trainer=cfg_trainer)

        # Val
        epoch_loss = eval_fn(gen_REC=gen_REC, loader=data_loaders['val'], criterion=early_stopping_criterion, device=device, outputs_dir=False)

        history['val_loss'].append(epoch_loss)
        if epoch > cfg_trainer['warm_up']:
            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_gen_REC_wts = copy.deepcopy(gen_REC.state_dict())
                best_gen_LE_wts = copy.deepcopy(gen_LE.state_dict())
                best_disc_REC_wts = copy.deepcopy(disc_REC.state_dict())
                best_disc_LE_wts = copy.deepcopy(disc_LE.state_dict())
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

    gen_REC.load_state_dict(best_gen_REC_wts)
    gen_LE.load_state_dict(best_gen_LE_wts)
    disc_REC.load_state_dict(best_disc_REC_wts)
    disc_LE.load_state_dict(best_disc_LE_wts)

    # Save model
    save_checkpoint(gen_REC, opt_gen, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN_REC"]))
    save_checkpoint(gen_LE, opt_gen, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_GEN_LE"]))
    save_checkpoint(disc_REC, opt_disc, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_CRITIC_REC"]))
    save_checkpoint(disc_LE, opt_disc, filename=os.path.join(model_fold_dir, cfg_trainer["CHECKPOINT_CRITIC_LE"]))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return gen_REC, gen_LE, disc_REC, disc_LE, history


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


def test_4metrics(gen_REC, loader_train, loader_val, loader_test, mse_metric, psnr_metric, device, outputs_dir,
                  save_output):
    gen_REC.eval()
    loop_test = tqdm(loader_test, leave=True)
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    running_loss4 = 0.0

    if loader_train:
        loop_train = tqdm(loader_train, leave=True)
        with torch.no_grad():
            for idx, (lowenergy, recombined, file_names) in enumerate(loop_train):
                lowenergy = lowenergy.to(device)
                with torch.cuda.amp.autocast():
                    fake_recombined = gen_REC(lowenergy.float())
                    if save_output:
                        for output, file_name in zip(fake_recombined, file_names):  # SAVE OUTPUTS PNG
                            filename_output = "%s_output.png" % (file_name)
                            save_image(output, os.path.join(outputs_dir, filename_output))

    if loader_val:
        loop_val = tqdm(loader_val, leave=True)
        with torch.no_grad():
            for idx, (lowenergy, recombined, file_names) in enumerate(loop_val):
                lowenergy = lowenergy.to(device)
                with torch.cuda.amp.autocast():
                    fake_recombined = gen_REC(lowenergy.float())
                    if save_output:
                        for output, file_name in zip(fake_recombined, file_names):  # SAVE OUTPUTS PNG
                            filename_output = "%s_output.png" % (file_name)
                            save_image(output, os.path.join(outputs_dir, filename_output))

    with torch.no_grad():
        for idx, (lowenergy, recombined, file_names) in enumerate(loop_test):
            lowenergy = lowenergy.to(device)
            recombined = recombined.to(device)
            with torch.cuda.amp.autocast():
                fake_recombined = gen_REC(lowenergy.float())
                if save_output:
                    for output, file_name in zip(fake_recombined, file_names):  # SAVE OUTPUTS PNG
                        filename_output = "%s_output.png" % (file_name)
                        save_image(output, os.path.join(outputs_dir, filename_output))

            loss1 = mse_metric(fake_recombined, recombined)
            loss2 = psnr_metric(fake_recombined, recombined)
            loss3 = vif_p(fake_recombined.float(), recombined.float())
            if platform == 'win32': # se sono su windows
                loss4 = structural_similarity_index_measure(fake_recombined, recombined)
            else: # se sono su alvis
                loss4 = structural_similarity_index_measure(fake_recombined.to(torch.float16), recombined.to(torch.float16))

            # statistics
            running_loss1 += loss1.item() * lowenergy.size(0)
            running_loss2 += loss2.item() * lowenergy.size(0)
            running_loss3 += loss3.item() * lowenergy.size(0)
            running_loss4 += loss4.item() * lowenergy.size(0)

    epoch_loss1 = running_loss1 / len(loader_test)
    epoch_loss2 = running_loss2 / len(loader_test)
    epoch_loss3 = running_loss3 / len(loader_test)
    epoch_loss4 = running_loss4 / len(loader_test)
    return epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4





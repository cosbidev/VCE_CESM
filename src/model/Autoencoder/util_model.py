import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from piq import vif_p
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.utils import save_image
from tqdm import tqdm
from sys import platform

import src.model.Autoencoder.generator_model as generator_model
import src.utils.utils as util_general


def freeze_layer_parameters(model, freeze_layers):
    for name, param in model.named_parameters():
        if name.startswith(tuple(freeze_layers)):
            param.requires_grad = False


def initialize_model(model_name, cfg_model, device, state_dict=True):
    if model_name == "autoencoder":
        model = generator_model.Autoencoder()
    else:
        print("Invalid model name, exiting...")
        exit()
    return model


def train_model(model, criterion, optimizer, scheduler, model_name, data_loaders, model_dir, device, cfg_trainer):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': []}

    epochs_no_improve = 0
    early_stop = False

    for epoch in range(cfg_trainer["max_epochs"]):
        print('Epoch {}/{}'.format(epoch, cfg_trainer["max_epochs"] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{cfg_trainer["max_epochs"]}',
                      unit='img') as pbar:
                for inputs, truths, file_names in data_loaders[phase]:
                    inputs = inputs.to(device)
                    truths = truths.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())
                        loss = criterion(outputs, truths)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    pbar.update(inputs.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)

            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # Early stopping
            if phase == 'val':
                if epoch > cfg_trainer['warm_up']:
                    if epoch_loss < best_loss:
                        best_epoch = epoch
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= cfg_trainer["early_stopping"]:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(model, os.path.join(model_dir, "%s.pt" % model_name))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)
    # Training results Loss function
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(model_plot_dir, "Loss"))


def evaluate(model, data_loader, device, outputs_dir):
    # Metric
    test_results = {}

    mse = nn.MSELoss()

    # Test loop
    model.eval()
    with torch.no_grad():
        for inputs, truths, file_names in tqdm(data_loader):
            inputs = inputs.to(device)
            truths = truths.to(device)
            # Prediction
            outputs = model(inputs.float())

            for input, output, truth, file_name in zip(inputs, outputs, truths, file_names):

                # MSE
                metric = mse(output, truth).item()
                test_results[file_name] = metric

                # SAVE OUTPUTS
                filename_output = "%s_output.png" % (file_name)
                save_image(output, os.path.join(outputs_dir, filename_output))

    # Avg
    test_results = np.mean(list(test_results.values()))

    return test_results


def test_4metrics(model, loader_train, loader_val, loader_test, mse_metric, psnr_metric, device, outputs_dir, save_output):

    model.eval()
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
                    y_fake = model(x).to(device)
                    if save_output:
                        for output, file_name in zip(y_fake, file_names):  # SAVE OUTPUTS PNG
                            filename_output = "%s_output.png" % (file_name)
                            save_image(output, os.path.join(outputs_dir, filename_output))

    if loader_val:
        loop_val = tqdm(loader_val, leave=True)
        with torch.no_grad():
            for idx, (x, y, file_names) in enumerate(loop_val):  # ImgDataset restituisce prima LE e poi RECO
                x = x.to(device)
                with torch.cuda.amp.autocast():
                    y_fake = model(x).to(device)
                    if save_output:
                        for output, file_name in zip(y_fake, file_names):  # SAVE OUTPUTS PNG
                            filename_output = "%s_output.png" % (file_name)
                            save_image(output, os.path.join(outputs_dir, filename_output))

    with torch.no_grad():
        for idx, (x, y, file_names) in enumerate(loop_test):  # ImgDataset restituisce prima LE e poi RECO
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                y_fake = model(x).to(device)
                if save_output:
                    for output, file_name in zip(y_fake, file_names):  # SAVE OUTPUTS PNG
                        filename_output = "%s_output.png" % (file_name)
                        save_image(output, os.path.join(outputs_dir, filename_output))

            loss1 = mse_metric(y_fake, y)
            loss2 = psnr_metric(y_fake, y)
            if platform == 'win32':  # su windows
                loss3 = vif_p(y_fake, y)
                loss4 = structural_similarity_index_measure(y_fake, y)
            else: # su alvis
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


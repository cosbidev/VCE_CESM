import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import torch
import torch.nn as nn
import pandas as pd
import collections
import yaml
import ssl
from sys import platform

import src.utils.utils as util_general
import src.utils.util_data as util_data
import src.model.Autoencoder.util_model as util_model
from torchmetrics import PeakSignalNoiseRatio


torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = './model/cnn/pretrained'
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration file
if platform == 'win32': # su windows
    args = {}
    args['cfg_file'] = "./configs/autoencoder_test.yaml"
    with open(args['cfg_file']) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
else: # su alvis
    args = util_general.get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
exp_name = cfg['exp_name']
model_name = cfg['model']['model_name']
#lr = cfg['trainer']['optimizer']['lr']
acr = cfg['data']['acr']
save_img_output = cfg['output']['save_img_output']
cv = cfg['data']['cv']
fold_list = list(range(cv))

FPUCBM_train_val_test = cfg['dataset']['FPUCBM_train_val_test']
FPUCBM_complete = cfg['dataset']['FPUCBM_complete']
FPUCBM_test_acr = cfg['dataset']['FPUCBM_test_acr']
Public_train_val_test = cfg['dataset']['Public_train_val_test']

# Device
device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(device)

# Files and Directories
fold_dir = os.path.join(cfg['data']['fold_dir']) # FPUCBM Dataset
fold_public_dir = os.path.join(cfg['data']['fold_public_dir'])  # Public Dataset
fold_dir_acr = os.path.join(cfg['data']['fold_dir_acr']) # FPUCBM Dataset con test diviso per ACR

report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
util_general.create_dir(report_dir)

report_file_mse = os.path.join(report_dir, 'report_mse.xlsx')
report_file_psnr = os.path.join(report_dir, 'report_psnr.xlsx')
report_file_vif = os.path.join(report_dir, 'report_vif.xlsx')
report_file_ssim = os.path.join(report_dir, 'report_ssim.xlsx')

if save_img_output:
    outputs_dir = os.path.join(report_dir, "outputs")
    util_general.create_dir(outputs_dir)

# CV
results_mse = collections.defaultdict(lambda: [])
metric_cols_mse = []
results_psnr = collections.defaultdict(lambda: [])
metric_cols_psnr = []
results_vif = collections.defaultdict(lambda: [])
metric_cols_vif = []
results_ssim = collections.defaultdict(lambda: [])
metric_cols_ssim = []

for fold in fold_list:

    if save_img_output:
        outputs_fold_dir = os.path.join(outputs_dir, str(fold))
        util_general.create_dir(outputs_fold_dir)
    else:
        outputs_fold_dir = False

    # Results Frame
    metric_cols_mse.append("%s MSE" % str(fold))
    metric_cols_psnr.append("%s PSNR" % str(fold))
    metric_cols_vif.append("%s VIF" % str(fold))
    metric_cols_ssim.append("%s SSIM" % str(fold))

    # Load public dataset diviso in train, val e test
    if Public_train_val_test == True:
        fold_data = { step: pd.read_csv(os.path.join(fold_public_dir, str(fold), f'{step}.txt'), delimiter="\t", index_col='id') for step in ['train', 'val', 'test']}
        datasets = {step: util_data.ImgDataset_public_dataset(data=fold_data[step], cfg_data=cfg['data'], step=step, do_augmentation=False) for step in ['train', 'val', 'test']}
        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True,num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)}
        loader_test = data_loaders['test']

    # Load FPUCBM dataset diviso in train, val e test
    if FPUCBM_train_val_test == True:
        fold_data = {step: pd.read_csv(os.path.join(fold_dir, str(fold), f'{step}.txt'), delimiter="\t", index_col='id') for step in ['train', 'val', 'test']}
        datasets = {step: util_data.ImgDataset(data=fold_data[step], cfg_data=cfg['data'], do_augmentation=False, step=False) for step in ['train', 'val', 'test']}
        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True,num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False,num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)}
        loader_test = data_loaders['test']

    # Load campioni di Test di FPUCBM dataset con acr specificato nel file pix2pixtest.yaml
    if FPUCBM_test_acr == True:
        data_test = pd.read_csv(os.path.join(fold_dir_acr, str(fold), f'test_{acr}.txt'), delimiter="\t", index_col='id')
        dataset_test = util_data.ImgDataset(data=data_test, cfg_data=cfg['data'], step=False, do_augmentation=False)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker)

    # Load di FPUCBM dataset completo, senza divisione in train, val e test
    if FPUCBM_complete == True:
        all_data = pd.read_csv(os.path.join(fold_dir, 'all.txt'), delimiter="\t", index_col='id')
        all_datasets = util_data.ImgDataset(data=all_data, cfg_data=cfg['data'], step=False, do_augmentation=False)
        all_data_loaders = torch.utils.data.DataLoader(all_datasets, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker)
        loader_test = all_data_loaders

    # Model
    print("%s%s%s" % ("*" * 50, model_name, "*" * 50))

    # Carico modello addestrato
    pretrained_model = os.path.join(cfg['data']['model_dir'], cfg['model']['pretrained_model'])
    pretrained_model_dir = os.path.join(pretrained_model, str(fold))
    CHECKPOINT = os.path.join(pretrained_model_dir, "autoencoder_gong.pt")

    model = util_model.initialize_model(model_name=model_name, cfg_model=cfg['model'], device=device)
    checkpoint = torch.load(CHECKPOINT, map_location=device).module  # map_location = device
    model.load_state_dict(checkpoint.state_dict())

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Test model
    mse = nn.MSELoss()
    psnr = PeakSignalNoiseRatio().to(device)


    mse_test_results, psnr_test_results, vif_test_results, ssim_test_results = util_model.test_4metrics(model,loader_train=False, loader_val=False, loader_test=loader_test,
                                                                                                        mse_metric=mse, psnr_metric=psnr, device=device, outputs_dir=outputs_fold_dir, save_output=save_img_output)

    print("mse_test_results:", mse_test_results, "psnr_test_results:", psnr_test_results, "vif_test_results:", vif_test_results, "ssim_test_results:", ssim_test_results)

    # Update report
    results_mse["%s MSE" % str(fold)].append(mse_test_results)
    results_psnr["%s PSNR" % str(fold)].append(psnr_test_results)
    results_vif["%s VIF" % str(fold)].append(vif_test_results)
    results_ssim["%s SSIM" % str(fold)].append(ssim_test_results)

    # Save Results MSE
    results_mse_frame = pd.DataFrame(results_mse)
    results_mse_frame.insert(loc=0, column='std MSE', value=results_mse_frame[metric_cols_mse].std(axis=1))
    results_mse_frame.insert(loc=0, column='mean MSE', value=results_mse_frame[metric_cols_mse].mean(axis=1))
    results_mse_frame.insert(loc=0, column='model', value=model_name)
    results_mse_frame.to_excel(report_file_mse, index=False)

    # Save Results PSNR
    results_psnr_frame = pd.DataFrame(results_psnr)
    results_psnr_frame.insert(loc=0, column='std PSNR', value=results_psnr_frame[metric_cols_psnr].std(axis=1))
    results_psnr_frame.insert(loc=0, column='mean PSNR', value=results_psnr_frame[metric_cols_psnr].mean(axis=1))
    results_psnr_frame.insert(loc=0, column='model', value=model_name)
    results_psnr_frame.to_excel(report_file_psnr, index=False)

    # Save Results VIF
    results_vif_frame = pd.DataFrame(results_vif)
    results_vif_frame.insert(loc=0, column='std VIF', value=results_vif_frame[metric_cols_vif].std(axis=1))
    results_vif_frame.insert(loc=0, column='mean VIF', value=results_vif_frame[metric_cols_vif].mean(axis=1))
    results_vif_frame.insert(loc=0, column='model', value=model_name)
    results_vif_frame.to_excel(report_file_vif, index=False)

    # Save Results SSIM
    results_ssim_frame = pd.DataFrame(results_ssim)
    results_ssim_frame.insert(loc=0, column='std SSIM', value=results_ssim_frame[metric_cols_ssim].std(axis=1))
    results_ssim_frame.insert(loc=0, column='mean SSIM', value=results_ssim_frame[metric_cols_ssim].mean(axis=1))
    results_ssim_frame.insert(loc=0, column='model', value=model_name)
    results_ssim_frame.to_excel(report_file_ssim, index=False)


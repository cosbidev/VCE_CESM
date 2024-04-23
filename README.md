# A Deep Learning Approach for Virtual Contrast Enhancement in Contrast Enhanced Spectral Mammography
This repository contains the code used to run experiments related to Virtual Contrast Enhancement CESM.

## Installation

### Clone the repository
```bash
git clone
cd VCE_CESM
```

### Install dependencies

#### Using virtualenv
1) First, create a Python virtual environment (optional) using the following command:
```bash
python -m venv ve-vce_cesm
```

2) Activate the virtual environment using the following command:
```bash
source ve-vce_cesm/bin/activate
```

3) Install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

The code is organized into three folders:
1) "configs": contains YAML files used to set experiment parameters.
2) "data": contains "raw" and "processed" folders. The "processed" folder contains TXT files with paths to dataset images, while the "raw" folder should contain images without preprocessing.
3) "src": contains "model" and "utils" folders. The "model" folder includes subfolders for Autoencoder, CycleGAN, and PixPix, each containing specific model code. The "utils" folder includes the "utils.py" generic utility code and the "util_data.py" code for image processing.

For each model (Autoencoder, CycleGAN, and Pix2Pix), you can run training, transfer learning, or testing code.
These codes are located in the "src -> model -> Autoencoder/CycleGAN/Pix2Pix" folders.
1) Running the training code trains the model on the public dataset. Experiment parameters can be set using the configuration files "autoencoder_train.yaml", "cyclegan_train.yaml", or "pix2pix_train.yaml".
2) Running the transfer learning code loads a pre-trained model on the public dataset and retrains it on the FPUCBM dataset. Experiment parameters, including the pre-trained model to load, can be set using the configuration files "autoencoder_tran_learn.yaml", "cyclegan_tran_learn.yaml", or "pix2pix_tran_learn.yaml".
3) Running the test code tests the trained model on the desired dataset. Experiment parameters, including the test dataset, can be set using the configuration files "autoencoder_test.yaml", "cyclegan_test.yaml", or "pix2pix_test.yaml".

# Contact
For questions and comments, feel free to contact: aurora.rofena@unicampus.it, valerio.guarrasi@unicampus.it

# Citation
If you use our code in your study, please cite as:
[A Deep Learning Approach for Virtual Contrast Enhancement in Contrast Enhanced Spectral Mammography](https://arxiv.org/abs/2308.00471)

# License
The code is distributed under a [MIT License](https://github.com/cosbidev/VCE_CESM#MIT-1-ov-file)

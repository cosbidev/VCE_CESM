# VCE_CESM
This repository contains the code used to run experiments related to Virtual Contrast Enhancemnt CESM.

The code is organized into two folders for the main topics covered:

Folder "VCE": contains the code used for Virtual Contrast Enhancement.
Folder "Low_Dose": contains the code used to generate high-energy full-dose images from simulated high-energy low-dose images.
Both folders are divided into 4 subfolders:

"configs": contains YAML files used to set experiment parameters.
"data": contains "raw" and "processed" folders. The "processed" folder contains TXT files with paths to dataset images, while the "raw" folder should contain images without preprocessing.
"models": contains trained models.
"src/src_lowdose": contains "model" and "utils" folders. The "model" folder includes subfolders for Autoencoder, CycleGAN, and PixPix, each containing specific model code. The "utils" folder includes generic utility code and "util_data/util_data_lowdose" code for image processing.
Folder "VCE":

For each model (Autoencoder, CycleGAN, and Pix2Pix), you can run training, transfer learning, or testing code. These codes are located in the "src -> model -> Autoencoder/CycleGAN/Pix2Pix" folders.

Running the training code trains the model on the public dataset. Experiment parameters can be set using the configuration files "autoencoder_train.yaml", "cyclegan_train.yaml", or "pix2pix_train.yaml".
Running the transfer learning code loads a pre-trained model on the public dataset and retrains it on the FPUCBM dataset. Experiment parameters, including the pre-trained model to load, can be set using the configuration files "autoencoder_tran_learn.yaml", "cyclegan_tran_learn.yaml", or "pix2pix_tran_learn.yaml".
Running the test code tests the trained model on the desired dataset. Experiment parameters, including the test dataset, can be set using the configuration files "autoencoder_test.yaml", "cyclegan_test.yaml", or "pix2pix_test.yaml".
Folder "Low_Dose":

For each model (Autoencoder, CycleGAN, and Pix2Pix), you can run training code located in the "src_lowdose -> model -> Autoencoder/CycleGAN/Pix2Pix" folders. Experiment parameters can be set using the configuration files "autoencoder_lowdose.yaml", "cyclegan_lowdose.yaml", or "pix2pix_lowdose.yaml", including values for "k" and noise (reported as "perc").

# Contact
For questions and comments, feel free to contact me: aurora.rofena@unicampus.it

# Citation
If you use our code in your studt, please cite as:
(A Deep Learning Approach for Virtual Contrast Enhancement in Contrast Enhanced Spectral Mammography)[https://arxiv.org/abs/2308.00471]

# License
The code is distributed under a (MIT License)[https://github.com/cosbidev/VCE_CESM#MIT-1-ov-file]

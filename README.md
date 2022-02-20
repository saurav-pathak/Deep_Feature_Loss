# Deep_Feature_Loss
PyTorch Implementation of Deep Feature Loss based model

## Folder Structure
1) ./code_scripts/train_featurelossnet_without_reg.py: script for training featureloss network
2) ./code_scripts/train_denoisingnet_modi_lib.py: script for training denoising network
3) ./code_scripts/models_mod.py: script containing denoising and featureloss networks
4) ./code_scripts/load_data_lib.py: script for loading and preprocessing data
5) ./code_scripts/helper.py: script containing other useful functions
6) ./code_scripts/dataext.ipynb: jupyter notebook for extracting audio data from zip files and preprocess them
7) ./code_scripts/models/denosing_model_v1_lib.pth: trained denoising model
8) ./code_scripts/models/loss_model_without_red_mod_data.pth: trained loss model

## Featureoss Network Dataset folder structure

Following steps need to be followed for downloading and extracting dataset and perform required preprocessing of dataset for training featureloss network

1. Create a folder ./dataset_zip and download all the audio zip files of both dataset (TUT and Chime_home).
2. Create six empty folders - ./dataset1/asc1/trainset1_temp, ./dataset1/asc1/valset1_temp, ./dataset1/asc1/trainset1, ./dataset1/asc1/valset1, ./dataset1/dat1_temp, ./dataset1/dat1
3. Run all cells of dataext.ipynb to extract and preprocess the complete dataset

## Denoising Network Dataset folder structure

Following steps need to followed for using downloading and using denoising dataset

1. Create four folders ./NSDTSEA/trainset_clean, ./NSDTSEA/trainset_noisy, ./NSDTSEA/valset_clean, ./NSDTSEA/valset_noisy, and populate all the folders with Voice Bank corpus dataset






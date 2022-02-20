# Deep_Feature_Loss
PyTorch Implementation of Deep Feature Loss based model

## Folder Structure
./code_scripts/train_featurelossnet_without_reg.py: script for training featureloss network
./code_scripts/train_denoisingnet_modi_lib.py: script for training denoising network
./code_scripts/models_mod.py: script containing denoising and featureloss networks
./code_scripts/load_data_lib.py: script for loading and preprocessing data
./code_scripts/helper.py: script containing other useful functions
./code_scripts/dataext.ipynb: jupyter notebook for extracting audio data from zip files and preprocess them

## Featureoss Network Dataset folder structure

Following steps need to be followed for downloading and extracting dataset and perform required preprocessing of dataset for training featureloss network

1. Create a folder ./dataset_zip and download all the audio zip files of both dataset (TUT and Chime_home).
2. Create six empty folders - ./dataset1/asc1/trainset1_temp, ./dataset1/asc1/valset1_temp, ./dataset1/asc1/trainset1, ./dataset1/asc1/valset1, ./dataset1/dat1_temp, ./dataset1/dat1
3. Run all cells of dataext.ipynb to extract and preprocess the complete dataset

## Denoising Network Dataset folder structure








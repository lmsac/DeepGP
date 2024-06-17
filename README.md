# DeepGP
DeepGP is a deep learning framework for the prediction of MS/MS spectra and retention time of glycopeptides. 
# Tutorial
## User Guide
For detailed step-by-step instructions on how to get started with DeepGP, please refer to [User_guide.md]([main folder/User_guide.md](https://github.com/yuz2011/DeepGP/blob/main/User_guide.md)) available in the [main folder]().

## What's Inside User Guide

Package Requirements: A list of all required packages and software.

Package Installation: Step-by-step guide to installing necessary packages.

Demo Data: Information and access to demo data sets.

Demo Data Description: Detailed description of the demo data.

Step-by-Step Instructions: Comprehensive guide to help you run and understand DeepGP.

# Model
The model is available at the [Google Drive](https://drive.google.com/drive/folders/1J4CKnsrikETNgcLj9xL5eRjWqC0oQx8Q). 

Here are the trained DeepGP models. These models are organized into five files, each denoted by the following names: DeepFLR, human, mouse, human&mouse and mouse_rt.

DeepFLR: This is the base model.

mouse: This is the DeepGP model for spectra prediction trained with mouse datasets, built on top of the DeepFLR base model.

human: This is the DeepGP model for spectra prediction trained with human datasets, built on top of the DeepFLR base model.

human&mouse: This is the DeepGP model for spectra prediction trained with both human and mouse datasets, built on top of the DeepFLR base model.

mouse_rt: The is the DeepGP model for retention time prediction trained with mouse datasets, built on top of the DeepFLR base model.

For further details, please refer to the User Guide.

# Demo data 2
The demo data for iRT prediction is available at the [Google Drive](https://drive.google.com/drive/folders/1ysrME3schgZHtxB4JL114MrvyNnXx6_k). 

This demo data is for clear and comprehensive presentation of iRT pre-processing, calibration, model training, and prediction. It includes three relatively large mouse datasets.

For further details, please refer to the User Guide.

# Post analysis
We also present the post-analysis code for the re-identification in the [main folder/Post_analysis](). For detailed step-by-step instructions on how to perform post analysis, please refer to [User_guide_post_analysis.docx]() available in the [main main folder/Post_analysis]().

The demo data for post analysis is available at the [Google Drive](https://drive.google.com/drive/folders/1FGRUSyV-_pBYnTG8tqaY2594TSKY8e-9). 

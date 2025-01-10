# Welcome to the CLOUD-D RF Codebase!

This repo currently assumes you will run everything locally on sagemaker from the cloudd-rf directory. All data needs to be within the data folder. To download the data locally to your personal sagemaker studio, follow the data download instructions below carefully and name the folders exactly as shown.
    
## Overview of Summer Scripts

- plot_accuracy_vs_nuisance_params.ipynb

  This script generates accuracy vs nuisance parameters for each nuisance parameter and saves the plots to `data/fusion_plots/accuracy_vs_nuisance_params`. It’s currently set up with the fusion model trained on features extracted from the second to last layer of each model. To change this, run baseline_fusion to get a new fused model or load in an already trained model from the summer_models folder.

- baseline_fusion.ipynb

   This script trains a fused model on all features extracted from the individual models. The extracted features are saved to the fusion_data folder and the fused model is saved to the `data/summer_models` folder.

- feature_contribution_analysis.ipynb

   This script generates plots for average feature importance vs nuisance parameter and saves the plots to `data/fusion_plots/feature_importances`.

- rl_rfe_fusion_and_analysis.ipynb

   This script creates the RL and RFE fusion models, plots of accuracy vs. number of features used for fusion, and plots showing how much each individual model contributes to the fused model. The models are saved to `data/summer_models` and the plots are saved to `data/fusion_plots/fused_model_accuracy`. 

- validate_fused_models.ipynb

   This script generates confusion matrices for each of the different fused models as well as the individual team models. The plots are saved to `data/fusion_plots/confusion_matrices`.

- feature_extraction.ipynb

   This script performs feature extraction on each of the individual teams' models, using Principal Component Analysis to find the top 2 or 3 features and plot them on a 2D or 3D graph, respectively. The user can easily configure the script to create plots for any model, fully connected layer, and nuisance parameter.

- train_all_models_sagemaker.ipynb

   This script trains the four teams' models on the training dataset, and is intended to run on Amazon SageMaker.

## Linux Installation and Usage Instructions

1. Install necessary prerequisite software:
    
    `sudo apt install python3 python3-pip python3-venv python3-wheel`

2. Set up a Python virtual environment:
    
    `python3 -m venv .cloudd-rf-venv`
            
3. Activate the virtual environment (you will need to do this every time you begin working with the repository):
    
    `source .cloudd-rf-venv/bin/activate`

4. Install setuptools:
    
    `pip3 install setuptools`

5. From the project source directory, install the repository (you will need to do this any time you update files in the /src/ folder of this repository):
    
    `pip3 install --editable .`

    `pip3 install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

    `pip3 install torchinfo`

6. To verify correct execution of the codebase, run the following commands:
    
    `python3 examples/image_gen_test.py`
    
    `eog data/*.png`

7. To create an example dataset, run the following command:

    `python3 scripts/iq_dataset_create.py`

8. To clean the data folder between runs:

    `bash scripts/cleanup.sh`

## Windows Installation and Usage Instructions

To be written.

## MAC Installation and Usage Instructions

1. Install the necessary prerequisite software:

    `brew install python3 python3-pip python3-venv python3-wheel`

2. Set up a Python virtual environent:

    `python3 -m venv .cloudd-rf-venv`

3. Activate the virtual environment (you will need to do this every time you begin working with the repository):
    
    `source .cloudd-rf-venv/bin/activate`

4. Install setuptools:
    
    `pip3 install setuptools`

5. From the project source directory, install the repository (you will need to do this any time you update files in the /src/ folder of this repository):
    
    `pip3 install --editable .`

    `pip3 install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

    `pip3 install torchinfo`

6. To verify correct execution of the codebase, run the following commands:
    
    `python3 examples/image_gen_test.py`
    
    `open data/*.png`

7. To create an example dataset, run the following command:

    `python3 scripts/iq_dataset_create.py`

8. To clean the data folder between runs:

    `zsh scripts/cleanup.sh`

## How to retrieve data from our S3 bucket
1. (If not running on SageMaker but on a separate Jupyter Notebook instance): Install awscli on your machine. For Linux:
    `sudo apt install awscli`

2. Ensure you have an access key ID and secret access key generated.
    - Go to the Amazon Web Services console and click on the name of your account (it is located in the top right corner of the console). 
    - Then, in the expanded drop-down list, click Security Credentials. 
    - Scroll to the Access Keys section and select "create access key".
    - Select "local code" and proceed to follow instructions.
    - Take note of the generated access key ID and secret access key

3. Configure your AWS account within terminal: `aws configure`
    - It will prompt you for your access key ID, secret access key, region (put us-east-1), and default output format (put json).
    
4. Download data: `aws s3 sync s3://yourbucket /local/path`
    - For training, testing, validation datasets: `aws s3 sync s3://summer-team-bucket/06-04-24-data data/dataset`
    - For the trained summer models: `aws s3 sync s3://summer-team-bucket/summer_models data/summer_models`
    - For the fusion data: `aws s3 sync s3://summer-team-bucket/fusion_data data/fusion_data`

## How to generate data
Run `gen.py` within the scripts folder to generate data.

## How to run the tutorials
Follow ReadMe.md in the tutorials folder for instructions.

import argparse
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure

import torch.nn as nn
import torch.nn.functional as F

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

SIG_TYPES = [['2-ASK', ['ask', 2], 0],
             ['4-ASK', ['ask', 4], 1],
             ['8-ASK', ['ask', 8], 2],
             ['BPSK', ['psk', 2], 3],
             ['QPSK', ['psk', 4], 4],
             ['16-QAM', ['qam', 16], 5],
             ['Tone', ['constant'], 6],
             ['P-FMCW', ['p_fmcw'], 7]]
NUM_CLASSES = len(SIG_TYPES)
sig_names = [i[0] for i in SIG_TYPES]

OBS_INT = {
    'model': 2048,
    1: 2048,
    2: 1024,
    3: 512,
    4: 256    
}

FC_LAYERS = {
    1: 'fc3',
    2: 'fc1',
    3: 'fc3',
    4: 'fc2'
}

def float_list(arg):
    return list(map(float, arg.split(',')))

def int_list(arg):
    return list(map(int, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Number of samples per file.
    parser.add_argument("--num-sensors", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples-per-batch", type=int, default=1000) # CHUNK_SIZE
    parser.add_argument("--input-path", type=str, default=os.getenv("SM_CHANNEL_VAL"))
    parser.add_argument("--output-path", type=str, default=os.getenv("SM_OUTPUT_DIR"))
    parser.add_argument("--model-path", type=str, default=os.getenv("SM_BASE_MODEL_DIR"))
    
    return parser.parse_known_args()

def load_models(num_sensors, batch_size, samples_per_batch, input_path, model_path, device):
    print (f"Loading Models")
    import importlib
    models_config = {
        'team_models': {},
        'fused_models': {
            'baseline': {},
            'rl': {},
            'rfe': {}
        }
    }
    for sensor in range(1, num_sensors+1):
        models_config['team_models'][sensor] = {
           'model': None,
           'dataloader': None,
           'params': None,
           'features': None
        }    
    
        # Dynamically import each team model class and instantiate model from it
        module_name = f'team{sensor}_model'
        class_name = f'Team{sensor}Model'
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        team_model = class_(NUM_CLASSES)
        team_model.load_state_dict(torch.load(f'{model_path}/team/team{sensor}_model.pt', map_location=torch.device(device)))
        team_model.eval()
        team_model.to(device)
        print (f"Loaded Team {sensor} model")
        models_config['team_models'][sensor]['model'] = team_model

        # Get the number of trainable parameters in each of the teams' models
        team_params = sum(p.numel() for p in team_model.parameters() if p.requires_grad)
        print(f'# trainable params, Team {sensor}:', team_model)
        models_config['team_models'][sensor]['params'] = team_params
    
    # Load regular fused model
    reg_fused_model = xgb.XGBClassifier(tree_method="hist")
    reg_fused_model.load_model(f'{model_path}/baseline/baseline_fused_2tl.json')
    models_config['fused_models']['baseline']['model'] = reg_fused_model
    print('loaded baseline fused model')

    # Load RL fused model
    rl_fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
    rl_fused_model.load_model(f'{model_path}/rlrfe/fusion_data/rl_fused_2tl.json')
    with open(f'{model_path}/rlrfe/fusion_data/rl_feature_idxes_2tl.pkl', 'rb') as f:
        rl_feature_idxes = pickle.load(f)
    models_config['fused_models']['rl']['model'] = rl_fused_model
    models_config['fused_models']['rl']['feat_idxs'] = rl_feature_idxes
    print('loaded RL fused model')

    # Load RFE fused model
    rfe_fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
    rfe_fused_model.load_model(f'{model_path}/rlrfe/fusion_data/rfe_fused_2tl.json')
    with open(f'{model_path}/rlrfe/fusion_data/rfe_feature_idxes_2tl.pkl', 'rb') as f:
        rfe_feature_idxes = pickle.load(f)
    models_config['fused_models']['rfe']['model'] = rfe_fused_model
    models_config['fused_models']['rfe']['feat_idxs'] = rfe_feature_idxes
    print('loaded RFE fused model')
    

    return models_config

def get_num_samples(iq_input_path, samples_per_batch):
    joined_files = os.path.join(iq_input_path, "iqdata", "example_*.dat") 
    joined_list = glob.glob(joined_files)
    num_batches = len(joined_list)
    num_samples = num_batches * samples_per_batch
    return num_batches, num_samples
  
def load_data(channel_path, batch_size, num_batches, num_train_examples, data_obs_int):
    training_data = np.zeros((num_train_examples, 1, 2, OBS_INT['model']), dtype=np.float32)

    last_index = 0
    for k in range(num_batches):
        # This is used if we have a labeldata folder that stores class labels
        label_df = pd.read_csv(f"{channel_path}/labeldata/example_{k + 1}.csv")
        num_nans = 0
        iq_file_name = f"{channel_path}/iqdata/example_{k + 1}.dat"
        iq_data = np.fromfile(iq_file_name, np.csingle)
        iq_data = np.reshape(iq_data, (-1, data_obs_int))  # Turn the IQ data into chunks of (chunk size) x (data_obs_int)
        for j in range(iq_data.shape[0]):
            # Check if the current row contains NaN values
            if np.isnan(np.sum(iq_data[j][:])):    
                num_nans += 1
            else:
                iq_array_norm = iq_data[j][:] / np.max(np.abs(iq_data[j][:]))  # Normalize the observation
                iq_array = np.vstack((iq_array_norm.real, iq_array_norm.imag))  # Separate into 2 subarrays - 1 with only real (in-phase), the other only imaginary (quadrature)

                # Pad the iq array with zeros to meet the observation length requirement
                # This is needed because the CNN models have a fixed input size
                iq_array = np.pad(iq_array, ((0, 0), (0, OBS_INT['model'] - iq_array[0].size)), mode='constant', constant_values=0)

                training_data[last_index, 0, :, :] = iq_array
            last_index += 1
        
        if num_nans > 0:
            print(f'Found {num_nans} rows containing NaNs in {iq_file_name}')
    return torch.utils.data.DataLoader([training_data[i] for i in range(num_train_examples)], batch_size=batch_size, shuffle=False)

def get_sensor_param_dir(sensor, param, input_path):
    validation_dir = os.path.join(input_path, str(sensor), param)
    root_data_dir = os.listdir(validation_dir)
    return validation_dir, root_data_dir

def get_dataloader(sensor, param, param_value, input_path, samples_per_batch, batch_size):
    validation_dir = os.path.join(input_path, str(sensor), param, str(param_value))
    num_batches, num_samples = get_num_samples(validation_dir, samples_per_batch)
    dataloader = load_data(validation_dir, batch_size, num_batches, num_samples, OBS_INT[sensor])

    return num_batches, dataloader
    
def load_labels(validation_dir, num_batches):
    labels = torch.stack([torch.nn.functional.one_hot(torch.tensor(pd.read_csv(os.path.join(validation_dir, f'labeldata/example_1.csv')).iloc[:,0])) for i in range(num_batches)]).numpy()
    labels = labels.reshape((labels.shape[0] * labels.shape[1], labels.shape[2]))
    labels = np.argmax(labels, axis=1)
    return labels

def load_features(team_model, dataloader, layer, device):
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    selected_layer = getattr(team_model, layer)  #sometimes its just model or model.module
    input_features = selected_layer.in_features
    handle = selected_layer.register_forward_hook(get_features('feats'))

    feats_list = []

    # Feed the IQ data into the model
    for idx, inputs in tqdm(enumerate(dataloader)):
        with torch.inference_mode():
            preds = team_model(inputs.to(device))
        feats_list.append(features['feats'].cpu().numpy())

    feats_list = np.concatenate(feats_list)

    features = np.array(feats_list)
    features = torch.tensor(features)
    features = features.reshape(-1, features.shape[-1])
        
    handle.remove()
    
    return features

def get_team_model_accuracy(model, dataloader, labels):
    # Get model predictions
    outputs = []
    with torch.inference_mode():
        for idx, inputs in tqdm(enumerate(dataloader)):
            pred = np.argmax(model(inputs.to(device)).cpu())
            outputs.append(pred)
    outputs = np.array(outputs)

    # Get the overall accuracy
    num_correct = np.sum(outputs == labels)
    accuracy = num_correct / len(outputs)
    
    return accuracy


def get_fusion_model_accuracy(models_config, fusion_type, combined_tensor):
    print(f'Evaluating {fusion_type} fusion model accuracy')
    
    feats = combined_tensor if fusion_type == 'baseline' else combined_tensor[:, models_config['fused_models'][fusion_type]['feat_idxs']]
    
    preds = models_config['fused_models'][fusion_type]['model'].predict(feats)
    if fusion_type == 'baseline': 
        preds = np.argmax(preds, axis=1)

    num_correct = np.sum(preds == models_config['labels'])
    accuracy = num_correct / len(preds)
    print(f'{fusion_type} fused model validation accuracy:', accuracy)
    
    return accuracy

def plot_feature_contribution_over_param(models_config, param, num_sensors, batch_size, samples_per_batch, input_path, output_path, title):
    output_artifacts_dir = os.path.join(output_path, 'fusion_plots', 'accuracy_vs_nuisance_params')
    os.makedirs(output_artifacts_dir, exist_ok=True)
    
    # Load the root directory for the current param being evaluated (e.g. snr)
    param_dir, param_vals = get_sensor_param_dir(1, param, input_path)
    model_accuracies = {}
    processed_params = []
    for param_val in param_vals:        
        for sensor in range(1, num_sensors+1):
            print(f'Evaluating model {sensor} feature importance for {param}: {param_val}')
            
            if (sensor not in model_accuracies):
                model_accuracies[sensor] = []
            
            # Load Labels
            curr_sensor_data = os.path.join(param_dir, param_val)
            num_batches, num_samples = get_num_samples(curr_sensor_data, samples_per_batch)
            models_config['labels'] = load_labels(curr_sensor_data, num_batches)

            # Get Dataloader
            print (f"Loading Dataloader for Model {sensor}")
            num_batches, dataloader = get_dataloader(sensor, param, param_val, input_path, samples_per_batch, batch_size)
            models_config['team_models'][sensor]['dataloader'] = dataloader
        
            # Load Features
            print (f"Loading Features for Model {sensor}")
            models_config['team_models'][sensor]['features'] = load_features(models_config['team_models'][sensor]['model'], dataloader, FC_LAYERS[sensor], device)
            
            # Get Predictions
            print(f'Evaluating model {sensor} accuracy')
            accuracy = get_team_model_accuracy(models_config['team_models'][sensor]['model'], dataloader, models_config['labels'])
            model_accuracies[sensor].append(accuracy)
            print(f'Team {sensor} model accuracy for {param}={param_val}:', accuracy)
    
        combined_tensor = torch.cat(([models_config['team_models'][sensor]['features'] for sensor in range(1, num_sensors+1)]), dim=1)
        
        # Evaluate Fusion Model Accuracy
        for fusion_type in ['baseline','rl','rfe']:
            if (fusion_type not in model_accuracies):
                model_accuracies[fusion_type] = []
            model_accuracies[fusion_type].append(get_fusion_model_accuracy(models_config, fusion_type, combined_tensor))
        
        # Add current SNR as a processed SNR value
        processed_params.append(param_val)

    plot_data_dict = {}
    plot_data_dict[param] = processed_params
    for sensor in range(1, num_sensors+1):
        plot_data_dict[f'Team {sensor}']= model_accuracies[sensor]
    
    df = pd.DataFrame(plot_data_dict)
    df.sort_values(by=[param], inplace=True)
    df.plot.line(x=param)

    plt.ylabel('Average Probability of Correct Classification')
    plt.xlabel(title)
    plt.title(f'Accuracy vs. {param.upper()}')
    plt.savefig(f'{output_artifacts_dir}/{param}_with_fused.png')
    plt.show()
    
if __name__ == "__main__":
    
    args, _ = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device', device)
    
    # Load Models
    models_config = load_models(args.num_sensors, args.batch_size, args.samples_per_batch, args.input_path, args.model_path, device)
    
    # Plot Accuracies over Nuisance Params
    x_axis_labels = {
        'snr': 'Signal-to-Noise Ratio (dB)',
        'cent_freqs': 'Center Frequency'
    }
    
    for param in ['snr','cent_freqs']:
        plot_feature_contribution_over_param(models_config, param, args.num_sensors, args.batch_size, args.samples_per_batch, args.input_path, args.output_path, x_axis_labels[param])
    

import os
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import xgboost as xgb
import pickle
import math
from random import sample

from tqdm import tqdm
from gymnasium import spaces
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter 
import torch.nn.functional as F
import glob
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go

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

NUM_COMPONENTS = 2
NUM_SAMPLES = 700

MODEL_CONFIG = {
    'model': {
        'obs_int': 2048,
    },
    1: {
        'plot': True,
        'obs_int': 2048,
        'fc_layer': 'fc3',
        'params_to_plot': ['snr'],
        'bandwidths': [0.5],
        'center_freqs': [0.5],
        'sig_types': 'all',
        'snrs': [5, 10, 14]
    },
    2: {
        'plot': True,
        'obs_int': 1024,
        'fc_layer': 'fc1',
        'params_to_plot': ['bandwidth', 'cent_freq', 'obs_int', 'snr'],
        'bandwidths': [0.5],
        'center_freqs': [0.5],
        'sig_types': 'all',
        'snrs': [14]
    },
    3: {
        'plot': True,
        'obs_int': 512,
        'fc_layer': 'fc3',
        'params_to_plot': ['bandwidth', 'cent_freq', 'obs_int', 'snr'],
        'bandwidths': [0.5],
        'center_freqs': [0.5],
        'sig_types': [],
        'snrs': [14]
    },
    4: {
        'plot': True,
        'obs_int': 256,
        'fc_layer': 'fc2',
        'params_to_plot': ['bandwidth', 'cent_freq', 'obs_int', 'sig_type', 'snr'],
        'bandwidths': [0.05, 0.2525, 0.5],
        'center_freqs': [256, 1024, 2048],
        'sig_types': 'all',
        'snrs': [5, 10, 14]
    }
}

COLORS = ['Blue', 'Orange', 'Green', 'Red', 'Purple', 'Brown', px.colors.qualitative.Plotly[6], 'Gray']

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
    parser.add_argument("--input-path", type=str, default=os.getenv("SM_CHANNEL_TEST"))
    parser.add_argument("--output-path", type=str, default=os.getenv("SM_OUTPUT_DIR"))
    parser.add_argument("--model-path", type=str, default=os.getenv("SM_MODEL_DIR"))
    
    return parser.parse_known_args()

def load_models(num_sensors, batch_size, samples_per_batch, input_path, model_path, device):
    global NUM_CLASSES
    print (f"Loading Models")
    import importlib
    models_config = {
        'team_models': {}
    }
    for sensor in range(1, num_sensors+1):
        models_config['team_models'][sensor] = {
           'model': None,
           'dataloader': None,
           'params': None,
           'importance': None,
           'features': None
        }
        # Dynamically import each team model class and instantiate model from it
        module_name = f'team{sensor}_model'
        class_name = f'Team{sensor}Model'
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        team_model = class_(NUM_CLASSES)
        team_model.load_state_dict(torch.load(f'{model_path}/team{sensor}_model.pt', map_location=torch.device(device)))
        team_model.eval()
        team_model.to(device)
        print (f"Loaded Team {sensor} model")
        models_config['team_models'][sensor]['model'] = team_model

        # Get the number of trainable parameters in each of the teams' models
        team_params = sum(p.numel() for p in team_model.parameters() if p.requires_grad)
        models_config['team_models'][sensor]['params'] = team_params

    return models_config

def get_num_samples(iq_input_path, samples_per_batch):
    joined_files = os.path.join(iq_input_path, "iqdata", "example_*.dat") 
    joined_list = glob.glob(joined_files)
    num_batches = len(joined_list)
    num_samples = num_batches * samples_per_batch
    return num_batches, num_samples
  
def load_data(channel_path, batch_size, num_batches, num_train_examples, data_obs_int):
    training_data = np.zeros((num_train_examples, 1, 2, MODEL_CONFIG['model']['obs_int']), dtype=np.float32)
    training_labels = np.zeros((num_train_examples, NUM_CLASSES), dtype=np.float32)
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
                iq_array = np.pad(iq_array, ((0, 0), (0, MODEL_CONFIG['model']['obs_int'] - iq_array[0].size)), mode='constant', constant_values=0)

                training_data[last_index, 0, :, :] = iq_array
                training_labels[last_index, label_df.iloc[j]] = 1.0
            last_index += 1
        
        if num_nans > 0:
            print(f'Found {num_nans} rows containing NaNs in {iq_file_name}')
    return torch.utils.data.DataLoader([[training_data[i], training_labels[i]] for i in range(num_train_examples)], batch_size=batch_size, shuffle=False)

def get_sensor_param_dir(sensor, param, input_path):
    validation_dir = os.path.join(input_path, str(sensor), param)
    root_data_dir = os.listdir(validation_dir)
    return validation_dir, root_data_dir

def get_dataloader(sensor, param, param_value, input_path, samples_per_batch, batch_size):
    validation_dir = os.path.join(input_path, str(sensor), param, str(param_value))
    num_batches, num_samples = get_num_samples(validation_dir, samples_per_batch)
    dataloader = load_data(validation_dir, batch_size, num_batches, num_samples, MODEL_CONFIG[sensor]['obs_int'])

    return num_batches, dataloader
    
def load_labels(validation_dir, num_batches):
    labels = torch.stack([torch.nn.functional.one_hot(torch.tensor(pd.read_csv(os.path.join(validation_dir, f'labeldata/example_1.csv')).iloc[:,0])) for i in range(num_batches)]).numpy()
    labels = labels.reshape((labels.shape[0] * labels.shape[1], labels.shape[2]))
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
    labels_list = []

    # Feed the IQ data into the model
    for idx, (inputs, labels) in tqdm(enumerate(dataloader)):
        with torch.inference_mode():
            preds = team_model(inputs.to(device))
        feats_list.append(features['feats'].cpu().numpy())
        labels_list.append(labels.numpy())
        if idx == NUM_SAMPLES:  # Including too many samples can make the plot difficult to read
            break

    feats_list = np.concatenate(feats_list)
    labels_list = np.concatenate(labels_list)

    features = np.array(feats_list)
    features = torch.tensor(features)
    features = features.reshape(-1, features.shape[-1])
    #print("features size: ", features.size())
    #print (type(features))
        
    handle.remove()
    
    return features, labels_list
    
def plot_feature_extraction_over_param(models_config, param, num_sensors, batch_size, samples_per_batch, input_path, output_path, title):
    output_artifacts_dir = os.path.join(output_path, 'fusion_plots', 'feature_extraction')
    os.makedirs(output_artifacts_dir, exist_ok=True)
    
    # Load the root directory for the current param being evaluated (e.g. snr)
    param_dir, param_vals = get_sensor_param_dir(1, param, input_path)
    
    MARKER_MIN_SIZE = 5
    MARKER_MAX_SIZE = 15
    
    if param != 'sig_types':
        min_param_val = min([float(x) for x in param_vals])
        max_param_val = max([float(x) for x in param_vals])
        

    # Each row is (comp1, comp2, marker size)
    sig_points = [np.empty(shape=[0, NUM_COMPONENTS+1]) for i in SIG_TYPES]

    for param_val in param_vals:
        
        # Get padding value based on current param val
        constant_vals = np.interp(param_val, [min_param_val, max_param_val], [MARKER_MIN_SIZE, MARKER_MAX_SIZE]) if param != 'sig_types' else 15
        
        for sensor in range(1, num_sensors+1):
            print(f'Extracting features for Team {sensor} model for {param}: {param_val}')
            
            # Get Dataloader
            print (f"Loading Dataloader for Model {sensor}")
            num_batches, dataloader = get_dataloader(sensor, param, param_val, input_path, samples_per_batch, batch_size)
            models_config['team_models'][sensor]['dataloader'] = dataloader
        
            # Load Features
            print (f"Loading Features for Model {sensor}")
            features, labels_list = load_features(models_config['team_models'][sensor]['model'], dataloader, MODEL_CONFIG[sensor]['fc_layer'], device)
            models_config['team_models'][sensor]['features'] = features
                        
            # Extract features and get data needed for the scatterplot
            pca = PCA(n_components=NUM_COMPONENTS)
            X_pca = pca.fit_transform(features)
            y = np.argmax(labels_list, axis=1)  # Color markers based on ground truth label
            for idx in range(len(SIG_TYPES)):                
                sig_points[idx] = np.append(sig_points[idx], np.pad(X_pca[(y == idx)], pad_width=((0, 0), (0, 1)), mode='constant', constant_values=constant_vals), axis=0)
            
            add_signals_to_plot(sig_points, output_artifacts_dir, param, param_val, sensor, MODEL_CONFIG[sensor]['fc_layer'])

def add_signals_to_plot(sig_points, output_artifacts_dir, param, param_val, sensor, layer):
    title = f"PCA of Dataset With {param.upper()} {param_val} for Team {sensor} Model, Layer {layer.upper()}"
    # Add traces for each signal type
    data = []
    for idx in range(len(SIG_TYPES)):
        trace = {
            'type': 'scatter' if NUM_COMPONENTS == 2 else 'scatter3d',
            'mode': 'markers',
            'x': sig_points[idx][:,0],
            'y': sig_points[idx][:,1],
            'marker_symbol': 'circle',
            'marker': dict( color=COLORS[idx],
                            size=sig_points[idx][:,NUM_COMPONENTS],
                            opacity=0.3),
            'name':SIG_TYPES[idx][0],
            'showlegend': True
        }
        if NUM_COMPONENTS == 3:
            trace['z'] = sig_points[idx][:,2]

        data.append(trace)
    
    fig = go.Figure(data)
    
    width=800 
    height=600
    
    layout = {
        'title': dict(text=title),
        'xaxis_title':'Principal Component 1',
        'yaxis_title':'Principal Component 2',
        'autosize': False,
        'width': width,
        'height': height
    }
    if NUM_COMPONENTS == 3:
        layout['zaxis_title'] = 'Principal Component 3'
        width=1200 
        height=900
        layout['width'] = width
        layout['height'] = height

    fig.update_layout(layout)
    
    fig.write_image(file=f'{output_artifacts_dir}/pca_{param.upper()}_{param_val}_team_{sensor}_{layer.upper()}.png', format='png', width=width, height=height)

if __name__ == "__main__":
    
    args, _ = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device', device)
    
    # Load Models
    models_config = load_models(args.num_sensors, args.batch_size, args.samples_per_batch, args.input_path, args.model_path, device)
    
    x_axis_labels = {
        'snr': 'Signal-to-Noise Ratio (dB)',
        'cent_freqs': 'Center Frequency',
        'sig_types': 'Signal Type'
    }
    
    for param in ['snr','cent_freqs', 'sig_types']:
        plot_feature_extraction_over_param(models_config, param, args.num_sensors, args.batch_size, args.samples_per_batch, args.input_path, args.output_path, x_axis_labels[param])

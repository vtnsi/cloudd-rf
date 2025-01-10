import argparse
import os
import torch
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import glob
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    parser.add_argument("--model-path", type=str, default=os.getenv("SM_MODEL_DIR"))
    
    return parser.parse_known_args()

def load_models(num_sensors, batch_size, samples_per_batch, input_path, model_path, device):
    global NUM_CLASSES
    print (f"Loading Models")
    import importlib
    models_config = {
        'team_models': {},
        'reg_fused': {},
        'rl_fused': {},
        'rfe_fused': {}
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
        team_model.load_state_dict(torch.load(f'{model_path}/team{sensor}_model.pt', map_location=torch.device(device)))
        team_model.eval()
        team_model.to(device)
        print (f"Loaded Team {sensor} model")
        models_config['team_models'][sensor]['model'] = team_model

        # Get the number of trainable parameters in each of the teams' models
        team_params = sum(p.numel() for p in team_model.parameters() if p.requires_grad)
        print(f'# trainable params, Team {sensor}:', team_model)
        models_config['team_models'][sensor]['params'] = team_params
    
        # Get Dataloaders
        print (f"Loading Dataloader for Model {sensor}")
        num_batches, dataloader = get_dataloaders(sensor, input_path, samples_per_batch, batch_size)
        models_config['team_models'][sensor]['dataloader'] = dataloader
    
        # Load Features
        print (f"Loading Features for Model {sensor}")
        models_config['team_models'][sensor]['features'] = load_features(team_model, dataloader, FC_LAYERS[sensor], sensor, device)
    
    # Load Labels
    print ("Loading Labels")
    labels = load_labels(input_path, num_batches).numpy()
    labels = labels.reshape((labels.shape[0] * labels.shape[1], labels.shape[2]))
    labels = np.argmax(labels, axis=1)
    models_config['labels'] = labels
    models_config['num_batches'] = num_batches
    
    # # Load regular fused model
    # reg_fused_model = xgb.XGBClassifier(tree_method="hist")
    # reg_fused_model.load_model(f'{model_path}/baseline_fused_2tl.json')
    # print('loaded baseline fused model')
    #models_config['reg_fused']['model'] = reg_fused_model

    # # Load RL fused model
    # rl_fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
    # rl_fused_model.load_model(f'{model_path}/rl_fused_2tl.json')
    # with open(f'{model_path}/rl_feature_idxes_2tl.pkl', 'rb') as f:
    #     rl_feature_idxes = pickle.load(f)
    # print('loaded RL fused model')
    #models_config['rl_fused']['model'] = rl_fused_model

    # # Load RFE fused model
    # rfe_fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
    # rfe_fused_model.load_model(f'{model_path}/rfe_fused_2tl.json')
    # with open(f'{model_path}/rfe_feature_idxes_2tl.pkl', 'rb') as f:
    #     rfe_feature_idxes = pickle.load(f)
    # print('loaded RFE fused model')
    #models_config['rfe_fused']['model'] = rfe_fused_model
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

def get_dataloaders(sensor, input_path, samples_per_batch, batch_size):
    validation_dir = os.path.join(input_path, str(sensor))
    num_batches, num_samples = get_num_samples(validation_dir, samples_per_batch)
    dataloader = load_data(validation_dir, batch_size, num_batches, num_samples, OBS_INT[sensor])

    return num_batches, dataloader
    
def load_labels(input_path, num_batches):
    validation_dir = os.path.join(input_path, '1')
    labels = torch.stack([torch.nn.functional.one_hot(torch.tensor(pd.read_csv(os.path.join(validation_dir, f'labeldata/example_1.csv')).iloc[:,0])) for i in range(num_batches)])
    return labels

def load_features(team_model, dataloader, layer, sensor, device):
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    print('Attaching to Layer: ', layer)
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
    print("features size: ", features.size())
    print (type(features))
    
    handle.remove()
    
    return features

def get_model_accuracy(models_config, combined_tensor, input_path, output_path):
    global NUM_CLASSES
    
    labels_list = []

    testloader = load_labels(input_path, models_config['num_batches'])
    x = 0
    for batch in testloader:
        if torch.sum(batch) == 0:
            print(f'Found all zeros at index {x}')
        
        labels_list.append(batch)
        x += 1

    all_labels = torch.cat(labels_list, dim=0)
    labels= all_labels.numpy()

    output_artifacts_dir = os.path.join(output_path, 'fusion_data')
    X_train, X_test, y_train, y_test = train_test_split(combined_tensor.numpy(), labels, test_size=0.2, random_state=42)
    params = {
        'objective': 'multi:softmax',  
        'num_class': NUM_CLASSES,      
        'eval_metric': 'merror',       
        'max_depth': 6,                
        'min_child_weight': 1,         
        'subsample': 0.8,              
        'colsample_bytree': 0.8,       
        'learning_rate': 0.1,          
        'n_estimators': 100            
    }

    num_rounds = 15  


    model_fused = xgb.XGBClassifier(tree_method="hist")
    model_fused.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    preds = model_fused.predict(X_test).argmax(1)
    acc = (preds == y_test.argmax(1)).sum() / preds.shape[0]
    print(f"Accuracy: {acc}")
    model_fused.save_model(f'{output_artifacts_dir}/baseline_fused_2tl.json')

def save_features(models_config, num_sensors, output_path):
    # Save the extracted features and labels for fusion in other scripts
    fusion_data_folder = os.path.join(output_path, 'fusion_data')
    os.makedirs(fusion_data_folder, exist_ok=True)

    # Save features for the individual models
    for sensor in range(1, num_sensors+1):
        print(f'Saving model features for model {sensor}')
        with open(os.path.join(fusion_data_folder, f'team{sensor}_features_second_to_last.npy'), 'wb') as f:
            np.save(f, models_config['team_models'][sensor]['features'].numpy())
            
    # Save combined features
    combined_tensor = torch.cat(([models_config['team_models'][sensor]['features'] for sensor in range(1, num_sensors+1)]), dim=1)
    print(f"Combined_tensor size: {combined_tensor.size()}")
    with open(os.path.join(fusion_data_folder, 'combined_features_allsecond_to_last.npy'), 'wb') as f:
        np.save(f, combined_tensor.numpy())

    # Save labels
    with open(os.path.join(fusion_data_folder, 'labels.npy'), 'wb') as f:
        np.save(f, models_config['labels'])
        
    return combined_tensor

if __name__ == "__main__":
    
    args, _ = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device', device)
    
    # Load Models
    models_config = load_models(args.num_sensors, args.batch_size, args.samples_per_batch, args.input_path, args.model_path, device)

    # Save Model Features
    combined_tensor = save_features(models_config, args.num_sensors, args.output_path)
    
    # Evaluate Model Accuracy
    get_model_accuracy(models_config, combined_tensor, args.input_path, args.output_path)

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfg.models import Team1Model, Team2Model, Team3Model, Team4Model
from utils.iq_dataset import IQDataset


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def accuracy_plots(mode):
    accuracies = {}

    for team_idx, model in enumerate(models):
        accuracies[team_idx] = {}
        padding = OBS_INT - obs_ints[team_idx]

        model.load_state_dict(torch.load(f"ckpts/team{team_idx+1}_model.pt", weights_only=True))
        model.eval()
        model.to(device)

        for folder in tqdm(glob.glob(f"data/team{team_idx+1}/test/{mode}/*")):
            level = os.path.basename(folder)

            dataset = IQDataset(folder, obs_ints[team_idx], padding=padding, device=device)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            num_correct = 0
            for x, y in dataloader:
                # preds = torch.hstack([model(x).argmax(1) for x, _ in dataloader])
                num_correct += (model(x).argmax(1) == y).sum().item()

            accuracies[team_idx][level] = num_correct / len(dataset)

    print(accuracies)



if __name__ == "__main__":
    OBS_INT = 2048
    obs_ints = [2048, 1024, 512, 256]
    feature_extraction_layers = ["fc3", "fc1", "fc3", "fc2"]
    device = "cuda"
    batch_size = 1024

    sig_types = [
        ["2-ASK", ["ask", 2], 0],
        ["4-ASK", ["ask", 4], 1],
        ["8-ASK", ["ask", 8], 2],
        ["BPSK", ["psk", 2], 3],
        ["QPSK", ["psk", 4], 4],
        ["16-QAM", ["qam", 16], 5],
        ["Tone", ["constant"], 6],
        ["P-FMCW", ["p_fmcw"], 7],
    ]
    num_classes = len(sig_types)

    models = [Team1Model(num_classes), Team2Model(num_classes), Team3Model(num_classes), Team4Model(num_classes)]

    fused_model = xgb.XGBClassifier(tree_method="hist")
    fused_model.load_model('ckpts/baseline_fused.json')

    accuracy_plots("snr")
    accuracy_plots("cent_freqs")


# ## Plot accuracy vs SNR

# In[10]:


team1_snr_dir = os.path.join(team1_test_dir, 'snr')
team2_snr_dir = os.path.join(team2_test_dir, 'snr')
team3_snr_dir = os.path.join(team3_test_dir, 'snr')
team4_snr_dir = os.path.join(team4_test_dir, 'snr')

snr_folder_names = []
processed_snrs = []
team1_accuracies = []
team2_accuracies = []
team3_accuracies = []
team4_accuracies = []
fused_accuracies = []

# Iterate over every folder in the SNR directory
for path, subdirs, files in os.walk(team1_snr_dir):
    # If we're in the top-level SNR directory, get the values from all the folder names
    if os.path.basename(path) == 'snr':
        snr_folder_names = subdirs
    elif os.path.basename(path) in snr_folder_names:
        snr = int(os.path.basename(path))
        
        # Load the IQ and label data for this SNR
        labels = load_labels(path, 1, num_batches, num_examples, MODELS_OBS_INT).numpy()
        team1_dataloader = load_data(os.path.join(team1_snr_dir, str(snr)), 1, num_batches, num_examples, TEAM1_OBS_INT)
        team2_dataloader = load_data(os.path.join(team2_snr_dir, str(snr)), 1, num_batches, num_examples, TEAM2_OBS_INT)
        team3_dataloader = load_data(os.path.join(team3_snr_dir, str(snr)), 1, num_batches, num_examples, TEAM3_OBS_INT)
        team4_dataloader = load_data(os.path.join(team4_snr_dir, str(snr)), 1, num_batches, num_examples, TEAM4_OBS_INT)
        print(f'Loaded data for SNR {snr}')
        
        count_all_zeros = 0
        labels = np.squeeze(labels)
        for i in range(len(labels)):
            if np.sum(labels[i]) == 0:
                count_all_zeros += 1
        print("Number of missing labels:", count_all_zeros)
        labels = np.argmax(labels, axis=1)
        
        print('Processing data...')
        # Extract features from Team 1's model
        team1_selected_layer.register_forward_hook(get_features('feats'))
        team1_model.to(device)
        team1_feats_list = []
        team1_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team1_dataloader), desc='Team 1'):
            with torch.inference_mode():
                team1_preds_list.append(np.argmax(team1_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team1_feats_list.append(features['feats'].cpu().numpy())
        team1_feats_list = np.concatenate(team1_feats_list)
        features1 = np.array(team1_feats_list)
        features1 = torch.tensor(features1)
        features1 = features1.reshape(-1, features1.shape[-1])
        if features1.shape[1] != NUM_FEATS_T1:
            print(f'Error: Number of features extracted from team 1 model doesn\'t match expected value ({NUM_FEATS_T1})')
                
        # Extract features from Team 2's model
        team2_selected_layer.register_forward_hook(get_features('feats'))
        team2_model.to(device)
        team2_feats_list = []
        team2_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team2_dataloader), desc='Team 2'):
            with torch.inference_mode():
                team2_preds_list.append(np.argmax(team2_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team2_feats_list.append(features['feats'].cpu().numpy())
        team2_feats_list = np.concatenate(team2_feats_list)
        features2 = np.array(team2_feats_list)
        features2 = torch.tensor(features2)
        features2 = features2.reshape(-1, features2.shape[-1])
        if features2.shape[1] != NUM_FEATS_T2:
            print(f'Error: Number of features extracted from team 2 model doesn\'t match expected value ({NUM_FEATS_T2})')
        
        # Extract features from Team 3's model
        team3_selected_layer.register_forward_hook(get_features('feats'))
        team3_model.to(device)
        team3_feats_list = []
        team3_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team3_dataloader), desc='Team 3'):
            with torch.inference_mode():
                team3_preds_list.append(np.argmax(team3_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team3_feats_list.append(features['feats'].cpu().numpy())
        team3_feats_list = np.concatenate(team3_feats_list)
        features3 = np.array(team3_feats_list)
        features3 = torch.tensor(features3)
        features3 = features3.reshape(-1, features3.shape[-1])
        if features3.shape[1] != NUM_FEATS_T3:
            print(f'Error: Number of features extracted from team 3 model doesn\'t match expected value ({NUM_FEATS_T3})')
        
        # Extract features from team 4's model
        team4_selected_layer.register_forward_hook(get_features('feats'))
        team4_model.to(device)
        team4_feats_list = []
        team4_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team4_dataloader), desc='Team 4'):
            with torch.inference_mode():
                team4_preds_list.append(np.argmax(team4_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team4_feats_list.append(features['feats'].cpu().numpy())
        team4_feats_list = np.concatenate(team4_feats_list)
        features4 = np.array(team4_feats_list)
        features4 = torch.tensor(features4)
        features4 = features4.reshape(-1, features4.shape[-1])
        if features4.shape[1] != NUM_FEATS_T4:
            print(f'Error: Number of features extracted from team 4 model doesn\'t match expected value ({NUM_FEATS_T4})')
        
        combined_tensor = torch.cat((features1, features2, features3, features4), dim=1)
        
        # Create baseline fused model
        train_data, validation_data, train_labels, validation_labels = train_test_split(combined_tensor, labels, test_size=0.2)
        fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
        fused_model.fit(train_data, train_labels, eval_set=[(validation_data, validation_labels)], verbose=0)

        fused_model_out = fused_model.predict(validation_data)
        num_correct = np.sum(fused_model_out == validation_labels)
        num_validation_labels = validation_labels.shape[0]
        fused_model_accuracy = num_correct / num_validation_labels
        fused_accuracies.append(fused_model_accuracy)
        
        # Get accuracy values
        team1_accuracies.append(np.sum(np.array(team1_preds_list) == labels) / len(labels))
        team2_accuracies.append(np.sum(np.array(team2_preds_list) == labels) / len(labels))
        team3_accuracies.append(np.sum(np.array(team3_preds_list) == labels) / len(labels))
        team4_accuracies.append(np.sum(np.array(team4_preds_list) == labels) / len(labels))
            
        processed_snrs.append(snr)

# Create the accuracy plots
df = pd.DataFrame(
    {'snr': processed_snrs,
        'Model 1': team1_accuracies,
        'Model 2': team2_accuracies,
        'Model 3': team3_accuracies,
        'Model 4': team4_accuracies,
        'Fused Model': fused_accuracies
    })
df.sort_values(by=['snr'], inplace=True)
df.plot.line(x='snr')

plt.ylabel('Average Probability of Correct Classification')
plt.xlabel('Signal-to-Noise Ratio (dB)')
plt.title('Accuracy vs. SNR')
plt.savefig('../data/fusion_plots/accuracy_vs_nuisance_params/snr_with_fused.png')
plt.show()

df = pd.DataFrame(
    {'snr': processed_snrs,
        'Model 1': team1_accuracies,
        'Model 2': team2_accuracies,
        'Model 3': team3_accuracies,
        'Model 4': team4_accuracies
    })
df.sort_values(by=['snr'], inplace=True)
df.plot.line(x='snr')
plt.ylabel('Average Probability of Correct Classification')
plt.xlabel('Signal-to-Noise Ratio (dB)')
plt.title('Accuracy vs. SNR')
plt.savefig('../data/fusion_plots/accuracy_vs_nuisance_params/snr.png')
plt.show()


# ## Plot accuracy vs bandwidth

# In[11]:


team1_bandwidth_dir = os.path.join(team1_test_dir, 'bandwidths')
team2_bandwidth_dir = os.path.join(team2_test_dir, 'bandwidths')
team3_bandwidth_dir = os.path.join(team3_test_dir, 'bandwidths')
team4_bandwidth_dir = os.path.join(team4_test_dir, 'bandwidths')

bandwidth_folder_names = []
processed_bandwidths = []
team1_accuracies = []
team2_accuracies = []
team3_accuracies = []
team4_accuracies = []
fused_accuracies = []

# Iterate over every folder in the bandwidth directory
for path, subdirs, files in os.walk(team1_bandwidth_dir):
    # If we're in the top-level bandwidth directory, get the values from all the folder names
    if os.path.basename(path) == 'bandwidths':
        bandwidth_folder_names = subdirs
    elif os.path.basename(path) in bandwidth_folder_names:
        bandwidth = float(os.path.basename(path))
        
        # Load the IQ and label data for this bandwidth
        labels = load_labels(path, 1, num_batches, num_examples, MODELS_OBS_INT).numpy()
        team1_dataloader = load_data(os.path.join(team1_bandwidth_dir, str(bandwidth)), 1, num_batches, num_examples, TEAM1_OBS_INT)
        team2_dataloader = load_data(os.path.join(team2_bandwidth_dir, str(bandwidth)), 1, num_batches, num_examples, TEAM2_OBS_INT)
        team3_dataloader = load_data(os.path.join(team3_bandwidth_dir, str(bandwidth)), 1, num_batches, num_examples, TEAM3_OBS_INT)
        team4_dataloader = load_data(os.path.join(team4_bandwidth_dir, str(bandwidth)), 1, num_batches, num_examples, TEAM4_OBS_INT)
        print(f'Loaded data for bandwidth {bandwidth}')
        
        count_all_zeros = 0
        labels = np.squeeze(labels)
        for i in range(len(labels)):
            if np.sum(labels[i]) == 0:
                count_all_zeros += 1
        print("Number of missing labels:", count_all_zeros)
        labels = np.argmax(labels, axis=1)
        
        print('Processing data...')
        # Extract features from Team 1's model
        team1_selected_layer.register_forward_hook(get_features('feats'))
        team1_model.to(device)
        team1_feats_list = []
        team1_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team1_dataloader), desc='Team 1'):
            with torch.inference_mode():
                team1_preds_list.append(np.argmax(team1_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team1_feats_list.append(features['feats'].cpu().numpy())
        team1_feats_list = np.concatenate(team1_feats_list)
        features1 = np.array(team1_feats_list)
        features1 = torch.tensor(features1)
        features1 = features1.reshape(-1, features1.shape[-1])
        if features1.shape[1] != NUM_FEATS_T1:
            print(f'Error: Number of features extracted from team 1 model doesn\'t match expected value ({NUM_FEATS_T1})')
                
        # Extract features from Team 2's model
        team2_selected_layer.register_forward_hook(get_features('feats'))
        team2_model.to(device)
        team2_feats_list = []
        team2_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team2_dataloader), desc='Team 2'):
            with torch.inference_mode():
                team2_preds_list.append(np.argmax(team2_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team2_feats_list.append(features['feats'].cpu().numpy())
        team2_feats_list = np.concatenate(team2_feats_list)
        features2 = np.array(team2_feats_list)
        features2 = torch.tensor(features2)
        features2 = features2.reshape(-1, features2.shape[-1])
        if features2.shape[1] != NUM_FEATS_T2:
            print(f'Error: Number of features extracted from team 2 model doesn\'t match expected value ({NUM_FEATS_T2})')
        
        # Extract features from Team 3's model
        team3_selected_layer.register_forward_hook(get_features('feats'))
        team3_model.to(device)
        team3_feats_list = []
        team3_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team3_dataloader), desc='Team 3'):
            with torch.inference_mode():
                team3_preds_list.append(np.argmax(team3_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team3_feats_list.append(features['feats'].cpu().numpy())
        team3_feats_list = np.concatenate(team3_feats_list)
        features3 = np.array(team3_feats_list)
        features3 = torch.tensor(features3)
        features3 = features3.reshape(-1, features3.shape[-1])
        if features3.shape[1] != NUM_FEATS_T3:
            print(f'Error: Number of features extracted from team 3 model doesn\'t match expected value ({NUM_FEATS_T3})')
        
        # Extract features from team 4's model
        team4_selected_layer.register_forward_hook(get_features('feats'))
        team4_model.to(device)
        team4_feats_list = []
        team4_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team4_dataloader), desc='Team 4'):
            with torch.inference_mode():
                team4_preds_list.append(np.argmax(team4_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team4_feats_list.append(features['feats'].cpu().numpy())
        team4_feats_list = np.concatenate(team4_feats_list)
        features4 = np.array(team4_feats_list)
        features4 = torch.tensor(features4)
        features4 = features4.reshape(-1, features4.shape[-1])
        if features4.shape[1] != NUM_FEATS_T4:
            print(f'Error: Number of features extracted from team 4 model doesn\'t match expected value ({NUM_FEATS_T4})')
        
        combined_tensor = torch.cat((features1, features2, features3, features4), dim=1)
        
        # Create baseline fused model
        train_data, validation_data, train_labels, validation_labels = train_test_split(combined_tensor, labels, test_size=0.2)
        fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
        fused_model.fit(train_data, train_labels, eval_set=[(validation_data, validation_labels)], verbose=0)

        fused_model_out = fused_model.predict(validation_data)
        num_correct = np.sum(fused_model_out == validation_labels)
        num_validation_labels = validation_labels.shape[0]
        fused_model_accuracy = num_correct / num_validation_labels
        fused_accuracies.append(fused_model_accuracy)
        
        # Get accuracy values
        team1_accuracies.append(np.sum(np.array(team1_preds_list) == labels) / len(labels))
        team2_accuracies.append(np.sum(np.array(team2_preds_list) == labels) / len(labels))
        team3_accuracies.append(np.sum(np.array(team3_preds_list) == labels) / len(labels))
        team4_accuracies.append(np.sum(np.array(team4_preds_list) == labels) / len(labels))
            
        processed_bandwidths.append(bandwidth)

# Create the accuracy plots
df = pd.DataFrame(
    {'bandwidth': processed_bandwidths,
        'Model 1': team1_accuracies,
        'Model 2': team2_accuracies,
        'Model 3': team3_accuracies,
        'Model 4': team4_accuracies,
        'Fused Model': fused_accuracies
    })
df.sort_values(by=['bandwidth'], inplace=True)
df.plot.line(x='bandwidth')

plt.ylabel('Average Probability of Correct Classification')
plt.xlabel('Bandwidth')
plt.title('Accuracy vs. Bandwidth')
plt.savefig('../data/fusion_plots/accuracy_vs_nuisance_params/bandwidth_with_fused.png')
plt.show()

df = pd.DataFrame(
    {'bandwidth': processed_bandwidths,
        'Model 1': team1_accuracies,
        'Model 2': team2_accuracies,
        'Model 3': team3_accuracies,
        'Model 4': team4_accuracies
    })
df.sort_values(by=['bandwidth'], inplace=True)
df.plot.line(x='bandwidth')

plt.ylabel('Average Probability of Correct Classification')
plt.xlabel('Bandwidth')
plt.title('Accuracy vs. Bandwidth')
plt.savefig('../data/fusion_plots/accuracy_vs_nuisance_params/bandwidth.png')
plt.show()


# ## Plot accuracy vs center frequency

# In[ ]:


folder_version = 'cent_freqs'

team1_cf_dir = os.path.join(team1_test_dir, folder_version)
team2_cf_dir = os.path.join(team2_test_dir, folder_version)
team3_cf_dir = os.path.join(team3_test_dir, folder_version)
team4_cf_dir = os.path.join(team4_test_dir, folder_version)

cf_folder_names = []
processed_cfs = []
team1_accuracies = []
team2_accuracies = []
team3_accuracies = []
team4_accuracies = []
fused_accuracies = []

# Iterate over every folder in the center frequencies directory
for path, subdirs, files in os.walk(team1_cf_dir):
    # If we're in the top-level center frequency directory, get the values from all the folder names
    if os.path.basename(path) == folder_version:
        cf_folder_names = subdirs
    elif os.path.basename(path) in cf_folder_names:
        cf = float(os.path.basename(path))
        
        # Load the IQ and label data for this center frequency
        labels = load_labels(path, 1, num_batches, num_examples, MODELS_OBS_INT).numpy()
        team1_dataloader = load_data(os.path.join(team1_cf_dir, str(cf)), 1, num_batches, num_examples, TEAM1_OBS_INT)
        team2_dataloader = load_data(os.path.join(team2_cf_dir, str(cf)), 1, num_batches, num_examples, TEAM2_OBS_INT)
        team3_dataloader = load_data(os.path.join(team3_cf_dir, str(cf)), 1, num_batches, num_examples, TEAM3_OBS_INT)
        team4_dataloader = load_data(os.path.join(team4_cf_dir, str(cf)), 1, num_batches, num_examples, TEAM4_OBS_INT)
        print(f'Loaded data for center frequency {cf}')
        
        count_all_zeros = 0
        labels = np.squeeze(labels)
        for i in range(len(labels)):
            if np.sum(labels[i]) == 0:
                count_all_zeros += 1
        print("Number of missing labels:", count_all_zeros)
        labels = np.argmax(labels, axis=1)
        
        print('Processing data...')
        # Extract features from Team 1's model
        team1_selected_layer.register_forward_hook(get_features('feats'))
        team1_model.to(device)
        team1_feats_list = []
        team1_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team1_dataloader), desc='Team 1'):
            with torch.inference_mode():
                team1_preds_list.append(np.argmax(team1_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team1_feats_list.append(features['feats'].cpu().numpy())
        team1_feats_list = np.concatenate(team1_feats_list)
        features1 = np.array(team1_feats_list)
        features1 = torch.tensor(features1)
        features1 = features1.reshape(-1, features1.shape[-1])
        if features1.shape[1] != NUM_FEATS_T1:
            print(f'Error: Number of features extracted from team 1 model doesn\'t match expected value ({NUM_FEATS_T1})')
                
        # Extract features from Team 2's model
        team2_selected_layer.register_forward_hook(get_features('feats'))
        team2_model.to(device)
        team2_feats_list = []
        team2_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team2_dataloader), desc='Team 2'):
            with torch.inference_mode():
                team2_preds_list.append(np.argmax(team2_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team2_feats_list.append(features['feats'].cpu().numpy())
        team2_feats_list = np.concatenate(team2_feats_list)
        features2 = np.array(team2_feats_list)
        features2 = torch.tensor(features2)
        features2 = features2.reshape(-1, features2.shape[-1])
        if features2.shape[1] != NUM_FEATS_T2:
            print(f'Error: Number of features extracted from team 2 model doesn\'t match expected value ({NUM_FEATS_T2})')
        
        # Extract features from Team 3's model
        team3_selected_layer.register_forward_hook(get_features('feats'))
        team3_model.to(device)
        team3_feats_list = []
        team3_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team3_dataloader), desc='Team 3'):
            with torch.inference_mode():
                team3_preds_list.append(np.argmax(team3_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team3_feats_list.append(features['feats'].cpu().numpy())
        team3_feats_list = np.concatenate(team3_feats_list)
        features3 = np.array(team3_feats_list)
        features3 = torch.tensor(features3)
        features3 = features3.reshape(-1, features3.shape[-1])
        if features3.shape[1] != NUM_FEATS_T3:
            print(f'Error: Number of features extracted from team 3 model doesn\'t match expected value ({NUM_FEATS_T3})')
        
        # Extract features from team 4's model
        team4_selected_layer.register_forward_hook(get_features('feats'))
        team4_model.to(device)
        team4_feats_list = []
        team4_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team4_dataloader), desc='Team 4'):
            with torch.inference_mode():
                team4_preds_list.append(np.argmax(team4_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team4_feats_list.append(features['feats'].cpu().numpy())
        team4_feats_list = np.concatenate(team4_feats_list)
        features4 = np.array(team4_feats_list)
        features4 = torch.tensor(features4)
        features4 = features4.reshape(-1, features4.shape[-1])
        if features4.shape[1] != NUM_FEATS_T4:
            print(f'Error: Number of features extracted from team 4 model doesn\'t match expected value ({NUM_FEATS_T4})')
        
        combined_tensor = torch.cat((features1, features2, features3, features4), dim=1)
        
        # Create baseline fused model
        train_data, validation_data, train_labels, validation_labels = train_test_split(combined_tensor, labels, test_size=0.2)
        fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
        fused_model.fit(train_data, train_labels, eval_set=[(validation_data, validation_labels)], verbose=0)

        fused_model_out = fused_model.predict(validation_data)
        num_correct = np.sum(fused_model_out == validation_labels)
        num_validation_labels = validation_labels.shape[0]
        fused_model_accuracy = num_correct / num_validation_labels
        fused_accuracies.append(fused_model_accuracy)
        
        # Get accuracy values
        team1_accuracies.append(np.sum(np.array(team1_preds_list) == labels) / len(labels))
        team2_accuracies.append(np.sum(np.array(team2_preds_list) == labels) / len(labels))
        team3_accuracies.append(np.sum(np.array(team3_preds_list) == labels) / len(labels))
        team4_accuracies.append(np.sum(np.array(team4_preds_list) == labels) / len(labels))
            
        processed_cfs.append(cf)

# Create the accuracy plots
df = pd.DataFrame(
    {'cf': processed_cfs,
        'Model 1': team1_accuracies,
        'Model 2': team2_accuracies,
        'Model 3': team3_accuracies,
        'Model 4': team4_accuracies,
        'Fused Model': fused_accuracies
    })
df.sort_values(by=['cf'], inplace=True)
df.plot.line(x='cf')
plt.ylabel('Average Probability of Correct Classification')
plt.xlabel('Center Frequency')
plt.title('Accuracy vs. Center Frequency')
plt.savefig('../data/fusion_plots/accuracy_vs_nuisance_params/cent_freq_with_fused.png')
plt.show()

df = pd.DataFrame(
    {'cf': processed_cfs,
        'Model 1': team1_accuracies,
        'Model 2': team2_accuracies,
        'Model 3': team3_accuracies,
        'Model 4': team4_accuracies
    })
df.sort_values(by=['cf'], inplace=True)
df.plot.line(x='cf')
plt.ylabel('Average Probability of Correct Classification')
plt.xlabel('Center Frequency')
plt.title('Accuracy vs. Center Frequency')
plt.savefig('../data/fusion_plots/accuracy_vs_nuisance_params/cent_freq.png')
plt.show()


# ## Plot accuracy vs obs int

# In[ ]:


team1_obs_dir = os.path.join(team1_test_dir, 'obs_ints')
team2_obs_dir = os.path.join(team2_test_dir, 'obs_ints')
team3_obs_dir = os.path.join(team3_test_dir, 'obs_ints')
team4_obs_dir = os.path.join(team4_test_dir, 'obs_ints')

obs_folder_names = []
processed_obs = []
team1_accuracies = []
team2_accuracies = []
team3_accuracies = []
team4_accuracies = []
fused_accuracies = []

# Iterate over every folder in the obs ints directory
for path, subdirs, files in os.walk(team1_obs_dir):
    # If we're in the top-level obs int directory, get the values from all the folder names
    if os.path.basename(path) == 'obs_ints':
        obs_folder_names = subdirs
    elif os.path.basename(path) in obs_folder_names:
        obs = int(os.path.basename(path))
        
        # Load the IQ and label data for this obs int
        labels = load_labels(path, 1, num_batches, num_examples, MODELS_OBS_INT).numpy()
        team1_dataloader = load_data(os.path.join(team1_obs_dir, str(obs)), 1, num_batches, num_examples, obs)
        team2_dataloader = load_data(os.path.join(team2_obs_dir, str(obs)), 1, num_batches, num_examples, obs)
        team3_dataloader = load_data(os.path.join(team3_obs_dir, str(obs)), 1, num_batches, num_examples, obs)
        team4_dataloader = load_data(os.path.join(team4_obs_dir, str(obs)), 1, num_batches, num_examples, obs)
        print(f'Loaded data for obs int {obs}')
        
        count_all_zeros = 0
        labels = np.squeeze(labels)
        for i in range(len(labels)):
            if np.sum(labels[i]) == 0:
                count_all_zeros += 1
        print("Number of missing labels:", count_all_zeros)
        labels = np.argmax(labels, axis=1)
        
        print('Processing data...')
        # Extract features from Team 1's model
        team1_selected_layer.register_forward_hook(get_features('feats'))
        team1_model.to(device)
        team1_feats_list = []
        team1_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team1_dataloader), desc='Team 1'):
            with torch.inference_mode():
                team1_preds_list.append(np.argmax(team1_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team1_feats_list.append(features['feats'].cpu().numpy())
        team1_feats_list = np.concatenate(team1_feats_list)
        features1 = np.array(team1_feats_list)
        features1 = torch.tensor(features1)
        features1 = features1.reshape(-1, features1.shape[-1])
        if features1.shape[1] != NUM_FEATS_T1:
            print(f'Error: Number of features extracted from team 1 model doesn\'t match expected value ({NUM_FEATS_T1})')
                
        # Extract features from Team 2's model
        team2_selected_layer.register_forward_hook(get_features('feats'))
        team2_model.to(device)
        team2_feats_list = []
        team2_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team2_dataloader), desc='Team 2'):
            with torch.inference_mode():
                team2_preds_list.append(np.argmax(team2_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team2_feats_list.append(features['feats'].cpu().numpy())
        team2_feats_list = np.concatenate(team2_feats_list)
        features2 = np.array(team2_feats_list)
        features2 = torch.tensor(features2)
        features2 = features2.reshape(-1, features2.shape[-1])
        if features2.shape[1] != NUM_FEATS_T2:
            print(f'Error: Number of features extracted from team 2 model doesn\'t match expected value ({NUM_FEATS_T2})')
        
        # Extract features from Team 3's model
        team3_selected_layer.register_forward_hook(get_features('feats'))
        team3_model.to(device)
        team3_feats_list = []
        team3_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team3_dataloader), desc='Team 3'):
            with torch.inference_mode():
                team3_preds_list.append(np.argmax(team3_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team3_feats_list.append(features['feats'].cpu().numpy())
        team3_feats_list = np.concatenate(team3_feats_list)
        features3 = np.array(team3_feats_list)
        features3 = torch.tensor(features3)
        features3 = features3.reshape(-1, features3.shape[-1])
        if features3.shape[1] != NUM_FEATS_T3:
            print(f'Error: Number of features extracted from team 3 model doesn\'t match expected value ({NUM_FEATS_T3})')
        
        # Extract features from team 4's model
        team4_selected_layer.register_forward_hook(get_features('feats'))
        team4_model.to(device)
        team4_feats_list = []
        team4_preds_list = []
        features = {}
        # Feed the IQ data into the model
        for idx, inputs in tqdm(enumerate(team4_dataloader), desc='Team 4'):
            with torch.inference_mode():
                team4_preds_list.append(np.argmax(team4_model(inputs.to(device)).cpu().numpy(), axis=1)[0])
            team4_feats_list.append(features['feats'].cpu().numpy())
        team4_feats_list = np.concatenate(team4_feats_list)
        features4 = np.array(team4_feats_list)
        features4 = torch.tensor(features4)
        features4 = features4.reshape(-1, features4.shape[-1])
        if features4.shape[1] != NUM_FEATS_T4:
            print(f'Error: Number of features extracted from team 4 model doesn\'t match expected value ({NUM_FEATS_T4})')
        
        combined_tensor = torch.cat((features1, features2, features3, features4), dim=1)
        
        # Create baseline fused model
        train_data, validation_data, train_labels, validation_labels = train_test_split(combined_tensor, labels, test_size=0.2)
        fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
        fused_model.fit(train_data, train_labels, eval_set=[(validation_data, validation_labels)], verbose=0)

        fused_model_out = fused_model.predict(validation_data)
        num_correct = np.sum(fused_model_out == validation_labels)
        num_validation_labels = validation_labels.shape[0]
        fused_model_accuracy = num_correct / num_validation_labels
        fused_accuracies.append(fused_model_accuracy)
        
        # Get accuracy values
        team1_accuracies.append(np.sum(np.array(team1_preds_list) == labels) / len(labels))
        team2_accuracies.append(np.sum(np.array(team2_preds_list) == labels) / len(labels))
        team3_accuracies.append(np.sum(np.array(team3_preds_list) == labels) / len(labels))
        team4_accuracies.append(np.sum(np.array(team4_preds_list) == labels) / len(labels))
            
        processed_obs.append(obs)

# Create the accuracy plots
df = pd.DataFrame(
    {'obs': processed_obs,
        'Model 1': team1_accuracies,
        'Model 2': team2_accuracies,
        'Model 3': team3_accuracies,
        'Model 4': team4_accuracies,
        'Fused Model': fused_accuracies
    })
df.sort_values(by=['obs'], inplace=True)
df.plot.line(x='obs')
plt.ylabel('Average Probability of Correct Classification')
plt.xlabel('Observation Duration')
plt.title('Accuracy vs. Observation Duration')
plt.savefig('../data/fusion_plots/accuracy_vs_nuisance_params/obs_int_with_fused.png')
plt.show()
    
df = pd.DataFrame(
    {'obs': processed_obs,
        'Model 1': team1_accuracies,
        'Model 2': team2_accuracies,
        'Model 3': team3_accuracies,
        'Model 4': team4_accuracies
    })
df.sort_values(by=['obs'], inplace=True)
df.plot.line(x='obs')
plt.ylabel('Average Probability of Correct Classification')
plt.xlabel('Observation Duration')
plt.title('Accuracy vs. Observation Duration')
plt.savefig('../data/fusion_plots/accuracy_vs_nuisance_params/obs_int.png')
plt.show()



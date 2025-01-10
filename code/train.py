
import argparse
import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm

import warnings

from team1_model import Team1Model
from team2_model import Team2Model
from team3_model import Team3Model
from team4_model import Team4Model

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
MODELS_OBS_INT = 2048  # 2048 for both spring and summer datasets

# For spring dataset: T1 - 2048, T2 - 1024, T3 - 1024, T4 - 512
# For summer dataset: T1 - 2048, T2 - 1024, T3 - 512, T4 - 256
TEAM1_DATA_OBS_INT = 2048
TEAM2_DATA_OBS_INT = 1024
TEAM3_DATA_OBS_INT = 1024
TEAM4_DATA_OBS_INT = 512


@dataclass
class TrainingConfig:
    num_epochs: int
    criterion: Any
    optimizer: Any
    model_save_dir: str
    model_save_filename: str
    data_dir: str
    data_obs_int: int


# set the device on GPU is available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

# Name, [modulation type, num symbols], class label
sig_types = [['2-ASK', ['ask', 2], 0],
             ['4-ASK', ['ask', 4], 1],
             ['8-ASK', ['ask', 8], 2],
             ['BPSK', ['psk', 2], 3],
             ['QPSK', ['psk', 4], 4],
             ['16-QAM', ['qam', 16], 5],
             ['Tone', ['constant'], 6],
             ['P-FMCW', ['p_fmcw'], 7]]
num_classes = len(sig_types)


def load_data(channel_path, batch_size, num_batches, num_train_examples, data_obs_int):
    training_data = np.zeros((num_train_examples, 1, 2, MODELS_OBS_INT), dtype=np.float32)
    training_labels = np.zeros((num_train_examples, num_classes), dtype=np.float32)

    last_index = 0
    for k in range(num_batches):
        # This is used if we have a labeldata folder that stores class labels
        label_df = pd.read_csv(f"{channel_path}/labeldata/example_{k + 1}.csv")

        iq_data = np.fromfile(f"{channel_path}/iqdata/example_{k + 1}.dat", np.csingle)
        iq_data = np.reshape(iq_data, (-1, data_obs_int))  # Turn the IQ data into chunks of (chunk size) x (data_obs_int)
        for j in range(iq_data.shape[0]):
            iq_array_norm = iq_data[j][:] / np.max(np.abs(iq_data[j][:]))  # Normalize the observation
            iq_array = np.vstack((iq_array_norm.real, iq_array_norm.imag))  # Separate into 2 subarrays - 1 with only real (in-phase), the other only imaginary (quadrature)

            # Pad the iq array with zeros to meet the observation length requirement
            # This is needed because the CNN models have a fixed input size
            iq_array = np.pad(iq_array, ((0, 0), (0, MODELS_OBS_INT - iq_array[0].size)), mode='constant', constant_values=0)

            training_data[last_index, 0, :, :] = iq_array
            training_labels[last_index, label_df.iloc[j]] = 1.0
            last_index += 1

    return torch.utils.data.DataLoader([[training_data[i], training_labels[i]] for i in range(num_train_examples)], batch_size=batch_size, shuffle=True)


def train(model, num_epochs, criterion, optim, scheduler, dataloader):
    # Put the model in training mode
    model.to(device)
    model.train()

    for _ in tqdm(range(num_epochs)):
        running_loss = 0.0

        # Training step
        for idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()

            # FOR TESTING ONLY: Break after 10 steps
            #if idx == 10:
            #    break

    return model

def setup_training(model, config):
    print(f'Loading dataset at {config.data_dir}')
    train_dir = config.data_dir #os.path.join(config.data_dir, 'train')
    train_iq_files = os.path.join(train_dir, "iqdata", "example_*.dat")
    file_list = glob.glob(train_iq_files)
    num_batches = len(file_list)
    num_train_examples = num_batches * args.chunk_size
    train_data = load_data(train_dir, args.batch_size, num_batches, num_train_examples, config.data_obs_int)

    print('Training model')
    model = train(model, config.num_epochs, config.criterion, config.optimizer, None, train_data)

    save_model_artifacts(model, config.model_save_dir, config.model_save_filename)


def save_model_artifacts(model, model_dir: str, model_name: str):
    """
    Saves a PyTorch model to disk
    :param model: The PyTorch model to be saved
    :param model_dir: The directory to save the model in. Any missing
    parent directories will be automatically created.
    :param model_name: The name of the model file
    :return: None
    """

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(model_dir, model_name)
    print(f'Saving model to {filepath}...')
    torch.save(model.state_dict(), filepath)
    print(" ")
    print('Model has been saved')


def parse_args():
    """
    Parses and loads the command-line arguments sent to the script. These
    will be sent by SageMaker when it launches the training container
    :return:
    """
    print('Parsing command-line arguments...')
    parser = argparse.ArgumentParser()

    # Observation length of the spectrum for each example.
    parser.add_argument("--obs-int", type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--s3_checkpoint_path', type=str, default='')
    parser.add_argument('--chunk-size', type=int, default=50)

    # Data directories
    #parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--team_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TEAM_DATA_DIR'))
    # parser.add_argument('--team2_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TEAM2_DATA_DIR'))
    # parser.add_argument('--team3_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TEAM3_DATA_DIR'))
    # parser.add_argument('--team4_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TEAM4_DATA_DIR'))
    
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    # Checkpoint info
    parser.add_argument('--checkpoint_enabled', type=str, default='False')
    parser.add_argument('--checkpoint_path', type=str, default='/opt/ml/checkpoints')

    print('Completed parsing command-line arguments.')

    return parser.parse_known_args()


if __name__ == '__main__':
    print('Executing the main() function...')
    # Parse command-line arguments
    args, _ = parse_args()
    
    if (args.team_data_dir is None): # or (args.team2_data_dir is None) or (args.team3_data_dir is None) or (args.team4_data_dir is None):
        raise ValueError("A data directory argument wasn't passed in correctly")

    # If running on SageMaker
    #model_dir = os.path.join('.', 'model')  # If running locally

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Every model will have the same hyperparameters
    # We need to check model performance between these and the hyperparameters chosen by 
    # the individual teams (commented out below)
    learning_rate = args.lr
    num_epochs = args.epochs
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam

    # Team 1 model
    print('Starting team 1 model training')
    t1_model = Team1Model(num_classes)
    t1_config = TrainingConfig(num_epochs=num_epochs, criterion=criterion,
                               optimizer=optimizer(t1_model.parameters(), lr=learning_rate), model_save_dir=args.model_dir,
                               model_save_filename='team1_model.pt', data_dir=f'{args.team_data_dir}/1', data_obs_int=TEAM1_DATA_OBS_INT)
    setup_training(t1_model, t1_config)
    # learning_rate_t1 = 0.001
    # num_epochs_t1 = 10
    # criterion_t1 = torch.nn.CrossEntropyLoss()
    # optimizer_t1 = torch.optim.SGD(modelt1.parameters(), lr=args.lr)
    # optimizer_t1 = optimizer(t1_model.parameters(), lr=learning_rate)
    # t1_model = train(t1_model, num_epochs, criterion, optimizer_t1, None, train_data)

    # Team 2 model
    print('Starting team 2 model training')
    t2_model = Team2Model(num_classes)
    t2_config = TrainingConfig(num_epochs=num_epochs, criterion=criterion,
                               optimizer=optimizer(t2_model.parameters(), lr=learning_rate), model_save_dir=args.model_dir,
                               model_save_filename='team2_model.pt', data_dir=f'{args.team_data_dir}/2', data_obs_int=TEAM2_DATA_OBS_INT)
    setup_training(t2_model, t2_config)
    # learning_rate_t2 = 0.01
    # num_epochs_t2 = 200
    # criterion_t2 = torch.nn.CrossEntropyLoss()
    # optimizer_t2 = torch.optim.SGD(modelt2.parameters(), lr=learning_rate)
    # optimizer_t2 = optimizer(modelt2.parameters(), lr=learning_rate)
    # modelt2 = train(modelt2, num_epochs, criterion, optimizer_t2, None, train_data)
    # save_model_artifacts(modelt2, model_dir, 'modelt2.pt')

    # Team 3 model
    print('Starting team 3 model training')
    t3_model = Team3Model(num_classes)
    t3_config = TrainingConfig(num_epochs=num_epochs, criterion=criterion,
                               optimizer=optimizer(t3_model.parameters(), lr=learning_rate), model_save_dir=args.model_dir,
                               model_save_filename='team3_model.pt', data_dir=f'{args.team_data_dir}/3', data_obs_int=TEAM3_DATA_OBS_INT)
    setup_training(t3_model, t3_config)
    # # learning_rate_t3 = 0.001
    # # num_epochs_t3 = 10
    # # criterion_t3 = torch.nn.CrossEntropyLoss()
    # # optimizer_t3 = torch.optim.SGD(modelt3.parameters(), lr=learning_rate)
    # optimizer_t3 = optimizer(modelt3.parameters(), lr=learning_rate)
    # modelt3 = train(modelt3, num_epochs, criterion, optimizer_t3, None, train_data)
    # save_model_artifacts(modelt3, model_dir, 'modelt3.pt')

    # Team 4 model
    print('Starting team 4 model training')
    t4_model = Team4Model(num_classes)
    t4_config = TrainingConfig(num_epochs=num_epochs, criterion=criterion,
                               optimizer=optimizer(t4_model.parameters(), lr=learning_rate), model_save_dir=args.model_dir,
                               model_save_filename='team4_model.pt', data_dir=f'{args.team_data_dir}/4', data_obs_int=TEAM4_DATA_OBS_INT)
    setup_training(t4_model, t4_config)
    # # learning_rate_t4 = 0.001
    # # num_epochs_t4 = 10
    # # criterion_t4 = torch.nn.CrossEntropyLoss()
    # # optimizer_t4 = torch.optim.Adam(modelt4.parameters(), learning_rate)
    # optimizer_t4 = optimizer(modelt4.parameters(), lr=learning_rate)
    # modelt4 = train(modelt4, num_epochs, criterion, optimizer_t4, None, train_data)
    # save_model_artifacts(modelt4, model_dir, 'modelt4.pt')

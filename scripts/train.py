import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfg.models import Team1Model, Team2Model, Team3Model, Team4Model
from utils.iq_dataset import IQDataset

MODELS_OBS_INT = 2048

TEAM1_DATA_OBS_INT = 2048
TEAM2_DATA_OBS_INT = 1024
TEAM3_DATA_OBS_INT = 512
TEAM4_DATA_OBS_INT = 256


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
device = torch.device("cuda")

# Name, [modulation type, num symbols], class label
sig_types = [["2-ASK", ["ask", 2], 0], ["4-ASK", ["ask", 4], 1], ["8-ASK", ["ask", 8], 2], ["BPSK", ["psk", 2], 3], ["QPSK", ["psk", 4], 4], ["16-QAM", ["qam", 16], 5], ["Tone", ["constant"], 6], ["P-FMCW", ["p_fmcw"], 7]]
num_classes = len(sig_types)


def train(model, num_epochs, criterion, optim, scheduler, dataloader):
    # Put the model in training mode
    model.to(device)
    model.train()

    for _ in tqdm(range(num_epochs)):
        running_loss = 0.0

        # Training step
        for data, labels in tqdm(dataloader, leave=False):

            outputs = model(data)
            loss = criterion(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()

    return model


def setup_training(model, config):
    print(f"Loading dataset at {config.data_dir}")
    train_dir = os.path.join(config.data_dir, "train")
    train_dataset = IQDataset(train_dir, config.data_obs_int, padding=MODELS_OBS_INT - config.data_obs_int)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print("Training model")
    model = train(model, config.num_epochs, config.criterion, config.optimizer, None, train_dataloader)

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
    print(f"Saving model to {filepath}...")
    torch.save(model.state_dict(), filepath)
    print(" ")
    print("Model has been saved")


def parse_args():
    """
    Parses and loads the command-line arguments sent to the script.
    :return:
    """
    print("Parsing command-line arguments...")
    parser = argparse.ArgumentParser()

    # Observation length of the spectrum for each example.
    parser.add_argument("--obs-int", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--s3_checkpoint_path", type=str, default="")
    parser.add_argument("--chunk-size", type=int, default=5000)

    # Data directories
    # parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument("--team1_data_dir", type=str, default="data/team1")
    parser.add_argument("--team2_data_dir", type=str, default="data/team2")
    parser.add_argument("--team3_data_dir", type=str, default="data/team3")
    parser.add_argument("--team4_data_dir", type=str, default="data/team4")

    # Checkpoint info
    parser.add_argument("--checkpoint_enabled", type=str, default="False")
    parser.add_argument("--checkpoint_path", type=str, default="/opt/ml/checkpoints")

    print("Completed parsing command-line arguments.")

    return parser.parse_known_args()


if __name__ == "__main__":
    print("Executing the main() function...")
    # Parse command-line arguments
    args, _ = parse_args()

    if (args.team1_data_dir is None) or (args.team2_data_dir is None) or (args.team3_data_dir is None) or (args.team4_data_dir is None):
        raise ValueError("A data directory argument wasn't passed in correctly")

    model_dir = os.path.join(".", "ckpts")

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
    print("Starting team 1 model training")
    t1_model = Team1Model(num_classes)
    t1_config = TrainingConfig(num_epochs=num_epochs, criterion=criterion, optimizer=optimizer(t1_model.parameters(), lr=learning_rate), model_save_dir=model_dir, model_save_filename="team1_model.pt", data_dir=args.team1_data_dir, data_obs_int=TEAM1_DATA_OBS_INT)
    setup_training(t1_model, t1_config)

    # Team 2 model
    print("Starting team 2 model training")
    t2_model = Team2Model(num_classes)
    t2_config = TrainingConfig(num_epochs=num_epochs, criterion=criterion, optimizer=optimizer(t2_model.parameters(), lr=learning_rate), model_save_dir=model_dir, model_save_filename="team2_model.pt", data_dir=args.team2_data_dir, data_obs_int=TEAM2_DATA_OBS_INT)
    setup_training(t2_model, t2_config)

    # Team 3 model
    print("Starting team 3 model training")
    t3_model = Team3Model(num_classes)
    t3_config = TrainingConfig(num_epochs=num_epochs, criterion=criterion, optimizer=optimizer(t3_model.parameters(), lr=learning_rate), model_save_dir=model_dir, model_save_filename="team3_model.pt", data_dir=args.team3_data_dir, data_obs_int=TEAM3_DATA_OBS_INT)
    setup_training(t3_model, t3_config)

    # Team 4 model
    print("Starting team 4 model training")
    t4_model = Team4Model(num_classes)
    t4_config = TrainingConfig(num_epochs=num_epochs, criterion=criterion, optimizer=optimizer(t4_model.parameters(), lr=learning_rate), model_save_dir=model_dir, model_save_filename="team4_model.pt", data_dir=args.team4_data_dir, data_obs_int=TEAM4_DATA_OBS_INT)
    setup_training(t4_model, t4_config)

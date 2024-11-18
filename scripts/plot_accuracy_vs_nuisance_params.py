import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
                num_correct += (model(x).argmax(1) == y).sum().item()

            accuracies[team_idx][float(level)] = num_correct / len(dataset)

    sns.lineplot(accuracies)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["Team 1 Model", "Team 2 Model", "Team 3 Model", "Team 4 Model"])
    # plt.legend(["Team 1 Model", "Team 2 Model", "Team 3 Model", "Team 4 Model"])
    plt.ylabel("Accuracy")
    plt.xlabel(mode)
    plt.savefig(f"plots/accuracy_vs_{mode}.png")
    plt.clf()

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    OBS_INT = 2048
    obs_ints = [2048, 1024, 512, 256]
    feature_extraction_layers = ["fc3", "fc1", "fc3", "fc2"]
    device = "cuda"
    batch_size = 1024

    sig_names = ["2-ASK", "4-ASK", "8-ASK", "BPSK", "QPSK", "16-QAM", "Tone", "P-FMCW"]
    num_classes = len(sig_names)

    models = [Team1Model(num_classes), Team2Model(num_classes), Team3Model(num_classes), Team4Model(num_classes)]

    fused_model = xgb.XGBClassifier(tree_method="hist")
    fused_model.load_model("ckpts/baseline_fused.json")

    accuracy_plots("snr")
    accuracy_plots("cent_freqs")

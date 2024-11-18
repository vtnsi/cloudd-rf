import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfg.models import Team1Model, Team2Model, Team3Model, Team4Model
from utils.iq_dataset import IQDataset

def calculate_importances(mode):

    extracted_features = []

    def get_features():
        def hook(model, input, output):
            extracted_features.append(output.detach())

        return hook

    fusion_data_folder = "data/fusion_data"
    os.makedirs(fusion_data_folder, exist_ok=True)

    all_features = {}
    all_labels = {}

    batch_size = 1024

    for team_idx, model in enumerate(models):
        model.load_state_dict(torch.load(f"ckpts/team{team_idx+1}_model.pt", weights_only=True))
        model.eval()
        model.to(device)

        selected_layer = getattr(model, feature_extraction_layers[team_idx])
        selected_layer.register_forward_hook(get_features())

        padding = OBS_INT - obs_ints[team_idx]

        for folder in tqdm(glob.glob(f"data/team{team_idx+1}/test/{mode}/*")):
            level = os.path.basename(folder)

            dataset = IQDataset(folder, obs_ints[team_idx], padding=padding, device=device)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            if level not in all_features:
                all_features[level] = []

                # Restrict to a single batch for processing speed
                all_labels[level] = dataset.labels.cpu().numpy()[:batch_size]

            extracted_features = []

            # Extract features from a batch
            model(next(iter(dataloader))[0])

            all_features[level].append(torch.vstack(extracted_features).cpu().numpy())

    all_importances = {}
    for level in tqdm(all_features):
        combined_features = np.hstack(all_features[level])
        labels = all_labels[level]

        iteration_importances = []
        for _ in range(3):

            clf = DecisionTreeClassifier()
            clf.fit(combined_features, labels)

            importances = np.split(clf.feature_importances_, np.cumsum(feature_extraction_shapes)[:-1])
            iteration_importances.append(list(map(sum, importances)))

        all_importances[level] = np.mean(iteration_importances, axis=0)

    df = pd.DataFrame(all_importances).T
    df.columns = ["Team 1 Model", "Team 2 Model", "Team 3 Model", "Team 4 Model"]
    sns.lineplot(df)
    plt.ylabel("Average Feature Importance")
    plt.xlabel(f"{mode}")
    plt.title(f"Average Feature Importance vs. {mode}, All Models")
    plt.savefig(f"plots/feature_importances_{mode}.png")


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

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

    OBS_INT = 2048

    models = [Team1Model(num_classes), Team2Model(num_classes), Team3Model(num_classes), Team4Model(num_classes)]
    obs_ints = [2048, 1024, 512, 256]
    feature_extraction_layers = ["fc3", "fc1", "fc3", "fc2"]
    feature_extraction_shapes = [65, 512, 64, 256]
    device = "cuda"

    calculate_importances("snr")
    calculate_importances("cent_freqs")

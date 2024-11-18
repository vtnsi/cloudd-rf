import os

import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfg.models import Team1Model, Team2Model, Team3Model, Team4Model
from utils.iq_dataset import IQDataset


def get_features():
    def hook(model, input, output):
        model_features.append(output.detach())

    return hook


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
all_features = []

device = "cuda"

fusion_data_folder = "data/fusion_data"
os.makedirs(fusion_data_folder, exist_ok=True)

for team_idx, model in enumerate(models):
    model.load_state_dict(torch.load(f"ckpts/team{team_idx+1}_model.pt", weights_only=True))
    model.eval()
    model.to(device)

    padding = OBS_INT - obs_ints[team_idx]
    dataset = IQDataset(f"data/team{team_idx+1}/validation", obs_ints[team_idx], padding=padding, device=device)
    dataloader = DataLoader(dataset, batch_size=1024)

    model_features = []
    selected_layer = getattr(model, feature_extraction_layers[team_idx])
    selected_layer.register_forward_hook(get_features())

    for x, y in tqdm(dataloader):
        with torch.inference_mode():
            preds = model(x)

    model_features = torch.vstack(model_features).cpu()
    all_features.append(model_features)
    torch.save(model_features, os.path.join(fusion_data_folder, f"team{team_idx+1}_features.pt"))

combined_features = torch.hstack(all_features)
combined_labels = dataset.labels.cpu()

torch.save(combined_features, os.path.join(fusion_data_folder, "combined_features.pt"))
torch.save(combined_labels, os.path.join(fusion_data_folder, "combined_labels.pt"))

X_train, X_test, y_train, y_test = train_test_split(combined_features.numpy(), combined_labels.numpy(), test_size=0.2, random_state=42)

model_fused = xgb.XGBClassifier(tree_method="hist")
model_fused.fit(X_train, y_train, eval_set=[(X_test, y_test)])

preds = model_fused.predict(X_test)
acc = (preds == y_test).sum() / preds.shape[0]
print(f"XGBoost accuracy: {acc:.3f}")

model_fused.save_model("ckpts/baseline_fused.json")

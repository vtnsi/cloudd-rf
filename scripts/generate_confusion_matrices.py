import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import confusion_matrix

from cfg.models import Team1Model, Team2Model, Team3Model, Team4Model
from utils.iq_dataset import IQDataset

device = "cuda"

# Load regular fused model
reg_fused_model = xgb.XGBClassifier(tree_method="hist")
reg_fused_model.load_model("ckpts/baseline_fused.json")
print("loaded baseline fused model")

# Load RL fused model
rl_fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
rl_fused_model.load_model("ckpts/rl_fused_model.json")
with open("ckpts/rl_feature_idxes.pkl", "rb") as f:
    rl_feature_idxes = pickle.load(f)

# Load RFE fused model
rfe_fused_model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
rfe_fused_model.load_model("ckpts/rfe_fused_model.json")
with open("ckpts/rfe_feature_idxes.pkl", "rb") as f:
    rfe_feature_idxes = pickle.load(f)

combined_features = torch.load("data/fusion_data/combined_features.pt")
labels = torch.load("data/fusion_data/combined_labels.pt")

sig_names = ["2-ASK", "4-ASK", "8-ASK", "BPSK", "QPSK", "16-QAM", "Tone", "P-FMCW"]

base_preds = reg_fused_model.predict(combined_features)
conf_mat = confusion_matrix(labels, base_preds, normalize="true")
sns.heatmap(conf_mat, annot=True, xticklabels=sig_names, yticklabels=sig_names, fmt=".2f")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Baseline Fused Model Confusion Matrix")
plt.savefig("plots/baseline_confusion_matrix.png")
plt.clf()

rl_preds = rl_fused_model.predict(combined_features[:, rl_feature_idxes])
conf_mat = confusion_matrix(labels, rl_preds, normalize="true")
sns.heatmap(conf_mat, annot=True, xticklabels=sig_names, yticklabels=sig_names, fmt=".2f")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("RL Fused Model Confusion Matrix")
plt.savefig("plots/rl_confusion_matrix.png")
plt.clf()

rfe_preds = rfe_fused_model.predict(combined_features[:, rfe_feature_idxes])
conf_mat = confusion_matrix(labels, rfe_preds, normalize="true")
sns.heatmap(conf_mat, annot=True, xticklabels=sig_names, yticklabels=sig_names, fmt=".2f")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("RFE Fused Model Confusion Matrix")
plt.savefig("plots/rfe_confusion_matrix.png")
plt.clf()

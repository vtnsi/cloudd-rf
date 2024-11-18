import os
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


NUM_FEATURES = 16  # How many of the top features to select for feature reduction
feature_extraction_shapes = [65, 512, 64, 256]

sig_types = [['2-ASK', ['ask', 2], 0],
             ['4-ASK', ['ask', 4], 1],
             ['8-ASK', ['ask', 8], 2],
             ['BPSK', ['psk', 2], 3],
             ['QPSK', ['psk', 4], 4],
             ['16-QAM', ['qam', 16], 5],
             ['Tone', ['constant'], 6],
             ['P-FMCW', ['p_fmcw'], 7]]
num_classes = len(sig_types)

data = torch.load('data/fusion_data/combined_features.pt')
labels = torch.load('data/fusion_data/combined_labels.pt')

def get_reward(train_data, train_labels, validation_data, validation_labels):
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
    clf.fit(train_data, train_labels, eval_set=[(validation_data, validation_labels)], verbose=0)

    return np.sum(clf.predict(validation_data) == validation_labels) / validation_labels.shape[0]

def statistical_feature_selection(data, labels):
    rewards = {}
    curr_features = np.arange(data.shape[1])
    curr_data = data
    for n in [data.shape[1], 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        curr_data = curr_data[:, curr_features]
        train_data, validation_data, train_labels, validation_labels = train_test_split(np.array(curr_data, dtype=np.float32), np.array(labels, dtype=np.int32), test_size=0.2)

        selector = RFE(DecisionTreeClassifier(), n_features_to_select=n, step=100, verbose=1)
        selector = selector.fit(train_data, train_labels)

        clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
        clf.fit(train_data[:,selector.support_], train_labels, eval_set=[(validation_data[:,selector.support_], validation_labels)], verbose=0)

        model_out = np.argmax(clf.predict(validation_data[:, selector.support_]), axis=1)
        num_correct = np.sum(model_out == np.argmax(validation_labels, axis=1))
        num_validation_labels = validation_labels.shape[0]
        acc = num_correct / num_validation_labels

        rewards[n] = acc
        curr_features = selector.support_

    sns.lineplot(rewards)
    plt.title("Accuracy of XGBoost vs number of features")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
                nn.LazyLinear(1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class FusionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SensorFusionBestFeatureDatasetEnv(gym.Env):
    def __init__(self, data, labels, num_classes, n, save_clf=True, load_clf=False):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.n = n
        self.mask_value = -100000

        self.observation_space = spaces.MultiDiscrete([self.data.shape[1]] * self.n)
        self.action_space = spaces.Discrete(self.data.shape[1])

        train_data, validation_data, train_labels, validation_labels = train_test_split(self.data, self.labels, test_size=0.2)
        self.train_data = train_data
        self.train_labels = train_labels.astype(int)
        self.validation_data = validation_data
        self.validation_labels = validation_labels.astype(int)

        self.curr_reward = 0
        self.prev_reward = 0
        self.clf = None

    def extract_important_features_using_decision_tree(self):
        clf = RandomForestClassifier(n_estimators=100, verbose=1)
        clf.fit(self.train_data, self.train_labels)

        return clf.feature_importances_.argsort()[-self.n:]

    def mask_data(self, data, idxes):
        mask = np.ones(data.shape, dtype=bool)
        for i in range(mask.shape[0]):
            mask[i][idxes[i]] = 0
        data[mask] = self.mask_value
        return data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        self.action_idxes = np.ones(self.n, dtype=int)

        return self.action_idxes, {}

    def step(self, action):
        if action in self.action_idxes:
            return self.action_idxes, 0, False, False, {}

        if self.count < self.n:
            self.action_idxes[self.count] = action
        else:
            self.action_idxes[np.random.randint(self.n)] = action

        accuracy = self.calculate_accuracy(self.action_idxes)

        self.count += 1
        if self.count == self.n:
            terminated = True
        else:
            terminated = False

        return self.action_idxes, accuracy, terminated, False, {}

    def calculate_accuracy(self, action_idxes):

        train_data = self.train_data[:, action_idxes]
        validation_data = self.validation_data[:, action_idxes]

        self.clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=5)
        self.clf.fit(train_data, self.train_labels, eval_set=[(validation_data, self.validation_labels)], verbose=0)

        model_out = self.clf.predict(validation_data)
        num_correct = np.sum(model_out == self.validation_labels)
        num_validation_labels = self.validation_labels.shape[0]
        acc = num_correct / num_validation_labels
        return acc

    def render(self):
        return

    def close(self):
        return

class FeatureSelectionModel(nn.Module):
    def __init__(self, n, num_features, env, epsilon=0.999, epsilon_decay=0.001):
        self.n = n
        self.num_features = num_features
        self.env = env
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episode = 0

        self.AOR = np.zeros(num_features)
        self.AOR_counts = np.zeros(num_features)

    def decay_epsilon(self):
        if self.episode > 1500:
            self.epsilon = max(self.epsilon - self.epsilon_decay, 0.0)
        elif self.episode > 500:
            self.epsilon = max(self.epsilon - self.epsilon_decay, 0.1)

    def update_AOR(self, action, reward):
        self.AOR[action] = (self.AOR[action] * self.AOR_counts[action] + reward) / (self.AOR_counts[action] + 1)
        self.AOR_counts[action] += 1

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(np.arange(self.num_features))
        mask = np.zeros(self.AOR.shape)
        mask[state] = True
        return ma.array(np.copy(self.AOR), mask=mask).argmax()

    def learn(self, episodes=16):
        writer = SummaryWriter(f"runs/n_{self.n}_episodes_{episodes}")
        for ep in tqdm(range(episodes)):
            rewards = []
            terminated = False
            state, _ = self.env.reset()
            while not terminated:
                self.decay_epsilon()

                action = self.select_action(state, self.epsilon)

                state, reward, terminated, _, _ = env.step(action)
                rewards.append(reward)
                self.update_AOR(action, reward)
            writer.add_scalar('reward', rewards[-1], ep)
            writer.add_scalar('epsilon', self.epsilon, ep)
            self.episode += 1

def visualize_output_features(d, l, nc):
    feature_avgs = torch.stack([d[l.int()==i].mean(axis=0) for i in range(nc)])
    plt.figure(0, (15, 1))
    sns.heatmap(feature_avgs)
    plt.yticks(ticks=np.arange(nc) + 0.5, labels=['2-ASK', '4-ASK', '8-ASK', 'BPSK', 'QPSK', '16-QAM', 'Tone', 'P-FMCW'], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

def run_eval(num_episodes, policy):
    rewards = []
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        ep_rewards = []
        ep_actions = []
        terminated = False
        while not terminated:
            action = policy(obs)
            _, reward, terminated, _, _ = env.step(action)
            ep_rewards.append(reward)
            ep_actions.append(action)
        rewards.append(ep_rewards[-1])
    return np.mean(rewards)

env = SensorFusionBestFeatureDatasetEnv(data, labels, num_classes, NUM_FEATURES)
env.reset()

device = "cuda"
model = PPO("MlpPolicy", env, verbose=1, batch_size=16, n_steps=64, device=device)
model.learn(total_timesteps=4096, progress_bar=True)
avg_model_rewards = run_eval(10, lambda obs: model.predict(obs)[0])
print(f"Average rewards for RL agent: {avg_model_rewards}")
print('RL Accuracy:', env.calculate_accuracy(env.action_idxes))

# Save the trained model
env.clf.save_model('ckpts/rl_fused_model.json')

# Save a list of the top contributing features we found
with open('ckpts/rl_feature_idxes.pkl', 'wb') as f:
    pickle.dump(env.action_idxes, f)

train_data, validation_data, train_labels, validation_labels = train_test_split(np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32), test_size=0.2)

selector = RFE(DecisionTreeClassifier(), n_features_to_select=NUM_FEATURES, step=100, verbose=1)
selector = selector.fit(train_data, train_labels)
clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
clf.fit(train_data[:,selector.support_], train_labels, eval_set=[(validation_data[:,selector.support_], validation_labels)], verbose=0)
model_out = clf.predict(validation_data[:, selector.support_])
num_correct = np.sum(model_out == validation_labels)
num_validation_labels = validation_labels.shape[0]
acc = num_correct / num_validation_labels
print('RFE Accuracy:', acc)

# Save the trained model
clf.save_model('ckpts/rfe_fused_model.json')

# Get the list of features selected using RFE
rfe_selected_feats = [i for i, x in enumerate(selector.support_) if x]

# Save the list of features
with open('ckpts/rfe_feature_idxes.pkl', 'wb') as f:
    pickle.dump(rfe_selected_feats, f)

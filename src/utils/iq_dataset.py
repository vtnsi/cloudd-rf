import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IQDataset(Dataset):
    """IQ Dataset class for loading IQ data and labels"""

    def __init__(self, data_dir, obs_int, padding=0, device="cuda"):
        """TODO: to be defined.

        :data_dir: TODO
        :obs_int: TODO

        """
        Dataset.__init__(self)

        self.device = device

        self._iq_files = sorted(glob.glob(os.path.join(data_dir, "iqdata", "*.dat")))
        self._label_files = sorted(glob.glob(os.path.join(data_dir, "labeldata", "*.csv")))

        self.labels = np.vstack([pd.read_csv(file).to_numpy() for file in self._label_files]).squeeze()
        self.labels = torch.tensor(self.labels, device=self.device)

        self.data = np.vstack([np.fromfile(file, np.csingle).reshape(-1, obs_int) for file in self._iq_files])
        self.data = self.data / np.max(np.abs(self.data), axis=1)[:, None]
        self.data = np.stack([self.data.real, self.data.imag]).transpose(1, 0, 2)
        self.data = np.pad(self.data, ((0, 0), (0, 0), (0, padding)), mode="constant", constant_values=0)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=self.device).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

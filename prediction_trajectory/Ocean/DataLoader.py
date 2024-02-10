import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, X, Y, Z, B1, B2, mask):
        self.X = X[mask != 0]
        self.Y = Y[mask != 0]
        self.Z = Z[mask != 0]
        self.B1 = B1
        self.B2 = B2

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x1_0 = np.squeeze(self.X[idx])
        x1_1 = x1_0@(self.B2@self.B2.T) + x1_0@(self.B1.T@self.B1)
        x1_2 = x1_1@(self.B2@self.B2.T) + x1_1@(self.B1.T@self.B1)
        y = self.Y[idx]
        z = self.Z[idx]

        class_number = self.binary_list_to_class_number(y)
        
        # Convert y and z to 1-dimensional tensors
        z_tensor = torch.tensor([z]) if np.isscalar(z) else torch.tensor(z)

        class_number = self.binary_list_to_class_number(y)

        # Convert class_number to a 1D tensor with a single value
        y_tensor = torch.tensor(class_number, dtype=torch.long)  # Ensure it's a long tensor

        return torch.Tensor(x1_0), torch.Tensor(x1_1), torch.Tensor(x1_2), z_tensor, y_tensor
    @staticmethod
    def binary_list_to_class_number(binary_list):
        binary_list = binary_list.flatten()
        if np.array_equal(binary_list, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]): return 0
        elif np.array_equal(binary_list, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]): return 1
        elif np.array_equal(binary_list, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]): return 2
        elif np.array_equal(binary_list, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]): return 3
        elif np.array_equal(binary_list, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]): return 4
        elif np.array_equal(binary_list, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]): return 5
        else: return -1  # Or any other default/error value
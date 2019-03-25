import torch
import numpy as np
import sys
import csv
from torch.utils.data import Dataset

b1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
np1 = np.arange(6).reshape((2,3)) + 1
print(b1)
print(np1.view)
import os
import sys

sys.path.append(os.path.join(sys.path[0], ".."))
from utils import compute, compute_masked
import unittest
import torch


class TestModule(unittest.TestCase):
    def testCompute(self):
        ones = torch.tensor([12, 1e-7])
        zeros = torch.zeros((2))
        y_mse, y_rmse, mae, y_mape, ape = compute(ones, zeros)
        print(y_mse, y_rmse, mae, y_mape, ape)
        print(compute_masked(ones, zeros))


if __name__ == "__main__":
    unittest.main()

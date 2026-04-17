import torch
import numpy as np
from model import BranchPredictor
from torch.utils.data import Dataset, DataLoader
from train import evaluate

class BranchDatasetFromCSV(Dataset):
    def __init__(self, csv_path, scaler_path):
        import joblib

        self.df = pd.read_csv(csv_path)
        self.df["edge_prob"] = self.df["prob_num"] / self.df["prob_den"]

        feature_cols = [
            col for col in self.df.columns
            if col.startswith("src_")
            or col.startswith("dst_")
            or col in ["succ_idx", "is_back_edge", "dst_is_loop_header"]
        ]

        X = self.df[feature_cols].fillna(0.0).values
        y = self.df["edge_prob"].values

        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    test_dataset = BranchDatasetFromCSV("splits/test_data.csv", "scaler.pkl")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    input_dim = test_dataset.X.shape[1]
    model = BranchPredictor(input_dim)
    model.load_state_dict(torch.load("branch_predictor.pt"))

    mae, corr = evaluate(model, test_loader)

    print(f"TEST: MAE={mae:.4f}, Corr={corr:.4f}")

    # should figure out some baselines to evaluate model against here
    # like: picking 0.5 every time, picking the mean of the edge probs from the train set every time, picking 1/0
    # this ^ needs to be coded
    # and then against previous work that we can find (this doesn't need to be coded lol)

    # we also said we would evaluate based on:
    # - mean absolute error betweeen predicted and actual (which we record already in evaluate)
    # hot edge prediction accuracy (need to figure this out)
    # and overlap with profiling-based hot paths (need to figure this out)
    # the last two metrics want us to reconstruct the hot path. or so we said in our proposal. so do we wanna do that.
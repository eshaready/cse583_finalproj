from torch.utils.data import DataLoader, Dataset, Subset
from model import BranchPredictor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
from tqdm import tqdm
import random

# the dataset loader should return this:
# X: dimension [num_samples x num_features] (feature list)
# y: dimension [num_samples x 1] (probabilities from profiling, within 0.0–1.0)
class BranchDataset(Dataset):
    def __init__(self, data_dir, scaler=None):
        # load all csvs, and also track file so it's split by file
        dfs = []

        for fname in os.listdir(data_dir):
            if fname.endswith(".csv"):
                path = os.path.join(data_dir, fname)
                df = pd.read_csv(path)

                df["__file__"] = fname
                dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        # calculate edge prob, clean & store
        self.df["edge_prob"] = self.df["prob_num"] / self.df["prob_den"]
        self.df = self.df.replace([np.inf, -np.inf], 0).dropna()
        self.df["edge_prob"] = self.df["edge_prob"].clip(0.0, 1.0)
        self.df = self.df[self.df["prob_den"] > 0]

        # select features. dont need identifiers or raw counts basically 
        feature_cols = []
        for col in self.df.columns:
            if (
                col.startswith("src_")
                or col.startswith("dst_")
                or col in ["succ_idx", "is_back_edge", "dst_is_loop_header"]
            ):
                feature_cols.append(col)
        self.feature_cols = sorted(feature_cols)

        # for X: doing some more cleaning to fix some errors with <unnamed>s lol
        X_df = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        X = X_df.fillna(0.0).values
        y = self.df["edge_prob"].values

        # normalize
        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)

        # to tensor in the format we want
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def save_scaler(self, path="scaler.pkl"):
        joblib.dump(self.scaler, path)

def split_by_file(dataset, test_ratio=0.2, val_ratio=0.2):
    random.seed(12)

    files = dataset.df["__file__"].unique().tolist()
    random.shuffle(files)

    n_total = len(files)
    n_test = int(test_ratio * n_total)
    n_val = int(val_ratio * n_total)

    test_files = files[:n_test]
    val_files = files[n_test:n_test + n_val]
    train_files = files[n_test + n_val:]

    print("Train files:", train_files)
    print("Val files:", val_files)
    print("Test files:", test_files)

    train_idx = dataset.df[dataset.df["__file__"].isin(train_files)].index.tolist()
    val_idx = dataset.df[dataset.df["__file__"].isin(val_files)].index.tolist()
    test_idx = dataset.df[dataset.df["__file__"].isin(test_files)].index.tolist()

    return train_idx, val_idx, test_idx, train_files, val_files, test_files

def save_splits(dataset, test_idx, val_idx, train_idx):
    os.makedirs("splits", exist_ok=True)

    test_df = dataset.df.iloc[test_idx]
    test_df.to_csv("splits/test_data.csv", index=False)

    val_df = dataset.df.iloc[val_idx]
    val_df.to_csv("splits/val_data.csv", index=False)

    train_df = dataset.df.iloc[train_idx]
    train_df.to_csv("splits/train_data.csv", index=False)

def evaluate(model, loader, device="cpu"):
    model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            prob = torch.sigmoid(logits)

            preds.append(prob.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mae = np.mean(np.abs(preds - targets))
    corr = np.corrcoef(preds.squeeze(), targets.squeeze())[0, 1]

    return mae, corr

def train_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3, device="cpu"):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_mae = float("inf")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        total_loss = 0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch_X, batch_y in batch_bar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        if val_loader is not None: 
            mae, corr = evaluate(model, val_loader, device=device)
            print(f"\nVal MAE: {mae:.4f}, Corr: {corr:.4f}")

            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), "best_model.pt")
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}: loss = {avg_loss:.4f}")

def main():
    print("Creating dataset...")
    dataset = BranchDataset("branch_edge_dataset_package/")
    dataset.save_scaler("scaler.pkl")

    # train/val/test split
    print("Splitting dataset into train/val/test...")
    train_idx, val_idx, test_idx, train_files, val_files, test_files = split_by_file(dataset)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False)

    save_splits(dataset, test_idx, val_idx, train_idx)
    print(f"Train size: {len(train_idx)}")
    print(f"Val size: {len(val_idx)}")
    print(f"Test size: {len(test_idx)}")

    # now the actual model 
    input_dim = dataset.X.shape[1]
    model = BranchPredictor(input_dim)

    train_model(model, train_loader, val_loader, epochs=75, lr=1e-3, device="cpu")

    torch.save(model.state_dict(), "last_model.pt")

    # printing test stats here too, altho stuff is saved so they can be evaluated later too
    mae, corr = evaluate(model, test_loader)
    print(f"TEST MAE: {mae:.4f}, Corr={corr:.4f}")

main()
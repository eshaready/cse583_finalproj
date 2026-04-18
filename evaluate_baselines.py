import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from model import BranchPredictor


CSV_FILES = [
    "branch_edge_dataset_package/deepsjeng_all_edges_clean.csv",
    "branch_edge_dataset_package/gcc_all_edges_clean.csv",
    "branch_edge_dataset_package/mcf_all_edges_clean.csv",
    "branch_edge_dataset_package/omnetpp_all_edges_clean.csv",
    "branch_edge_dataset_package/perlbench_all_edges_clean.csv",
    "branch_edge_dataset_package/x264_all_edges_clean.csv",
    "branch_edge_dataset_package/xz_all_edges_clean.csv",
]


class BranchDatasetFromCSVs(Dataset):
    def __init__(self, csv_paths, scaler_path):
        dfs = []
        for path in csv_paths:
            df = pd.read_csv(path)
            df["__file__"] = path
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        df["edge_prob"] = df["prob_num"] / df["prob_den"]
        df = df.replace([np.inf, -np.inf], 0).dropna()
        df["edge_prob"] = df["edge_prob"].clip(0.0, 1.0)
        df = df[df["prob_den"] > 0]

        feature_cols = []
        for col in df.columns:
            if (
                col.startswith("src_")
                or col.startswith("dst_")
                or col in ["succ_idx", "is_back_edge", "dst_is_loop_header"]
            ):
                feature_cols.append(col)
        feature_cols = sorted(feature_cols)

        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        y = df["edge_prob"].values

        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def metrics(preds, targets):
    mae  = np.mean(np.abs(preds - targets))
    corr = np.corrcoef(preds, targets)[0, 1]
    return mae, corr


def main():
    scaler_pkl = "scaler.pkl"
    model_pt   = "last_model.pt"

    print(f"Loading {len(CSV_FILES)} CSVs (test set)...")
    ds = BranchDatasetFromCSVs(CSV_FILES, scaler_pkl)
    X  = ds.X.numpy()
    y  = ds.y.numpy().squeeze()
    print(f"Test samples: {len(y)}")

    print("Loading train split for fitting baselines...")
    ds_train = BranchDatasetFromCSVs(["splits/train_data.csv"], scaler_pkl)
    X_train  = ds_train.X.numpy()
    y_train  = ds_train.y.numpy().squeeze()
    print(f"Train samples: {len(y_train)}")

    results = {}

    # 1. Constant 0.5
    results["Constant 0.5"] = metrics(np.full_like(y, 0.5), y)

    # 2. Train Mean
    print(y_train.mean())
    results["Train Mean"] = metrics(np.full_like(y, y_train.mean()), y)

    # 3. Linear Regression
    print("Fitting Linear Regression...")
    lr = LinearRegression().fit(X_train, y_train)
    results["Linear Regression"] = metrics(lr.predict(X).clip(0, 1), y)

    # 4. Random Forest
    print("Fitting Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    results["Random Forest"] = metrics(rf.predict(X).clip(0, 1), y)

    # 5. Neural Network
    print("Evaluating Neural Network...")
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    model  = BranchPredictor(ds.X.shape[1])
    model.load_state_dict(torch.load(model_pt, map_location="cpu"))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds.append(torch.sigmoid(model(X_batch)).numpy())
            targets.append(y_batch.numpy())
    results["Neural Network"] = metrics(
        np.concatenate(preds).squeeze(),
        np.concatenate(targets).squeeze()
    )

    # Print the eval table
    print("\n" + "=" * 52)
    print(f"{'Model':<25} {'MAE':>8} {'Correlation':>12}")
    print("-" * 52)
    for name, (mae, corr) in results.items():
        print(f"{name:<25} {mae:>8.4f} {corr:>12.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
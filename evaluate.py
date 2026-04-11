import torch
import numpy as np
from model import BranchPredictor

def predict(model, features):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = model(x)
        prob = torch.sigmoid(logits)
    return prob.item()

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()
        y_true = y.squeeze().numpy()

    mae = np.mean(np.abs(preds - y_true))
    corr = np.corrcoef(preds, y_true)[0, 1]

    return mae, corr

def main():
    model = BranchPredictor(num_features)
    model.load_state_dict(torch.load("branch_predictor.pt"))

    # ok do your predicting and your evaluating here
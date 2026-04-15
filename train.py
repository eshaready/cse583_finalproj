from torch.utils.data import DataLoader, TensorDataset
from model import BranchPredictor
from sklearn.preprocessing import StandardScaler
import joblib

# TODO: write a dataset loader once we have format of the features 
# and the dataset loader should return this:
# X: dimension [num_samples x num_features] (feature list)
# y: dimension [num_samples x 1] (probabilities from profiling, within 0.0–1.0)

def train_model(model, X, y, epochs=50, batch_size=64, lr=1e-3):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

def main():
    model = BranchPredictor(num_features) # we dont know num features yet lol
    features, groundtruth = [] # we need to write the dataset loader function to load this

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    joblib.dump(scaler, "scaler.pkl")
    
    train_model(model, features, groundtruth)
    torch.save(model.state_dict(), "branch_predictor.pt")
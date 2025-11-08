import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)

device = "cuda"
split = 0.15
batch_size = 64
lr = 0.0005
epochs = 3000


feature_list = [
    "LotArea",
    "OverallQual",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
]

label = "SalePrice"


class HouseDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]


class MLP(nn.Module):
    def __init__(self, input_features=8):
        super().__init__()
        self.ln1 = nn.Linear(input_features, 64)
        self.ln2 = nn.Linear(64, 32)
        self.ln3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ln1(x))
        x = self.dropout(x)
        x = self.relu(self.ln2(x))
        x = self.ln3(x)
        return x


def train_model(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


def predict_keggle(model, loader, shuffle=False):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in loader:
            features = features.to(device)
            outputs = model(features)

            predictions.append(outputs.cpu())

    return torch.cat(predictions).numpy()


if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")

    X = train_df[feature_list].to_numpy()
    y = train_df[label].to_numpy().reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=split, random_state=42
    )

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_val_scaled = scaler_x.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    train_dataset = HouseDataset(X_train_scaled, y_train_scaled)
    val_dataset = HouseDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_features=len(feature_list)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)

        if (epoch + 1) % 100 == 0:
            val_loss = evaluate_model(model, val_loader, criterion)
            print(
                f"epoch {epoch + 1} train loss: {train_loss:.6f} val loss {val_loss:.6f}"
            )

    keggle_df = pd.read_csv("./data/test.csv")
    keggle_ids = keggle_df["Id"].to_numpy()
    X_keggle = keggle_df[feature_list].fillna(0).to_numpy()

    X_keggle_scaled = scaler_x.transform(X_keggle)
    keggle_dataset = HouseDataset(X_keggle_scaled)
    keggle_loader = DataLoader(keggle_dataset, batch_size=batch_size, shuffle=False)

    keggle_pred_scaled = predict_keggle(model, keggle_loader)
    keggle_pred_y = scaler_y.inverse_transform(keggle_pred_scaled)

    keggle_submission_df = pd.DataFrame(
        {"Id": keggle_ids, "SalePrice": keggle_pred_y.flatten()}
    )
    keggle_submission_df.to_csv("submission.csv", index=False)

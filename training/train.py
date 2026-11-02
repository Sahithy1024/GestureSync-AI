import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Configuration
# -----------------------------

DATA_FILE = "data/gestures.csv"
MODEL_DIR = "models"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
RANDOM_STATE = 42


# -----------------------------
# Neural Network
# -----------------------------

class GestureMLP(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

print(f"Dataset size: {len(df)}")
print(f"Number of features: {X.shape[1]}")

print("\nClass distribution:")
print(df["label"].value_counts())


# -----------------------------
# Encode Labels
# -----------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

class_names = list(label_encoder.classes_)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# -----------------------------
# Train / Validation / Test
# -----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\nDataset split:")
print(f"Training:   {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Testing:    {len(X_test)}")


# -----------------------------
# Convert to PyTorch tensors
# -----------------------------

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test, dtype=torch.long)


# -----------------------------
# DataLoaders
# -----------------------------

train_dataset = torch.utils.data.TensorDataset(
    X_train,
    y_train
)

val_dataset = torch.utils.data.TensorDataset(
    X_val,
    y_val
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# -----------------------------
# Create Model
# -----------------------------

model = GestureMLP(
    input_size=X.shape[1],
    num_classes=len(class_names)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------

best_val_accuracy = 0.0

os.makedirs(MODEL_DIR, exist_ok=True)

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # ---- Training ----
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total


    # ---- Validation ----
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            f"{MODEL_DIR}/gesture_model.pth"
        )


# -----------------------------
# Save Label Mapping
# -----------------------------

with open(
    f"{MODEL_DIR}/label_mapping.json",
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


# -----------------------------
# Test Model
# -----------------------------

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        f"{MODEL_DIR}/gesture_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    inputs = X_test.to(device)

    outputs = model(inputs)

    predictions = torch.argmax(
        outputs,
        dim=1
    )

    all_predictions = predictions.cpu().numpy()
    all_labels = y_test.numpy()


# -----------------------------
# Evaluation
# -----------------------------

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

test_accuracy = (
    np.array(all_predictions) ==
    np.array(all_labels)
).mean()

print(f"\nTest Accuracy: {test_accuracy:.4f}")

print("\nBest Validation Accuracy:",
      f"{best_val_accuracy:.4f}")

print("\nModel saved to:")
print("models/gesture_model.pth")

print("\nLabel mapping saved to:")
print("models/label_mapping.json")

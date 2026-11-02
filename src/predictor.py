import json

import numpy as np
import torch
import torch.nn as nn


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


class GesturePredictor:

    def __init__(
        self,
        model_path = "../models/gesture_model.pth",
        label_path = "../models/label_mapping.json"
    ):

        self.device = torch.device("cpu")

        # Load labels
        with open(label_path, "r") as file:
            self.labels = json.load(file)

        # Create model
        self.model = GestureMLP(
            input_size=63,
            num_classes=len(self.labels)
        )

        # Load trained weights
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, features):

        features = np.asarray(
            features,
            dtype=np.float32
        )

        # Convert to tensor
        tensor = torch.tensor(
            features,
            dtype=torch.float32
        ).unsqueeze(0)

        tensor = tensor.to(self.device)

        # Prediction
        with torch.no_grad():

            output = self.model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        gesture = self.labels[
            prediction.item()
        ]

        confidence = confidence.item()

        return gesture, confidence
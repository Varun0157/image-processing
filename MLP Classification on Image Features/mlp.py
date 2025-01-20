# initial reference code.
# I want to create a simple multi layer perceptron in pytorch for image classification. the data is in the form of csv files, with columns label, pixel1, pixel2, ..., pixel784.
# Write code for the Dataset, dataloader, and the multi layer perceptron itself.

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.X = self.data.drop("label", axis=1).values
        self.y = self.data["label"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.X[idx].astype(np.float32) / 255.0
        image = torch.FloatTensor(image)

        label = self.y[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10):
        super(MLP, self).__init__()

        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.2),
                ]
            )
            current_size = hidden_size
        layers.append(nn.Linear(current_size, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def main():
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    train_dataset = ImageDataset("train.csv")
    test_dataset = ImageDataset("test.csv")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Initialize model, loss function, and optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print epoch statistics
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class RandomNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(RandomNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        # Initialize weights with a seed
        torch.manual_seed(42)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


def generate_data_with_noise(input_size, num_samples, noise_mean, noise_std):
    torch.manual_seed(14)  # Set the seed for reproducibility
    input_data = torch.rand(num_samples, input_size) * 5 - 2.5  # Scale to range [-3.5, 3.5]
    noisy_input_data = input_data + torch.randn_like(input_data) * noise_std + noise_mean
    return input_data, noisy_input_data


def plot_data_with_labels(input_data, predicted_labels, title=""):
    plt.figure(figsize=(8, 6))
    plt.scatter(input_data[:, 0], input_data[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Class')
    plt.title(f'Data Points with Predicted Labels and Gaussian Noise\n{title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def train_model(model, input_data, labels, criterion, optimizer, num_epochs=500):
    model.train()
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return losses


def plot_decision_boundaries(model, input_data, labels, title=""):
    x_min, x_max = input_data[:, 0].min() - 1, input_data[:, 0].max() + 1
    y_min, y_max = input_data[:, 1].min() - 1, input_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    grid_input = torch.tensor(grid_input, dtype=torch.float32)
    grid_output = model(grid_input)
    grid_labels = grid_output.argmax(axis=1)
    plt.contourf(xx, yy, grid_labels.reshape(xx.shape), alpha=0.3, cmap='viridis')
    plt.title(f'Decision Boundary\n{title}')
    plt.show()


def plot_data_with_decision_boundary(model, input_data, noisy_input_data, labels, title=""):
    plt.figure(figsize=(8, 6))

    # Plot decision boundary
    x_min, x_max = noisy_input_data[:, 0].min() - 1, noisy_input_data[:, 0].max() + 1
    y_min, y_max = noisy_input_data[:, 1].min() - 1, noisy_input_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    grid_input = torch.tensor(grid_input, dtype=torch.float32)
    grid_output = model(grid_input)
    grid_labels = grid_output.argmax(axis=1)
    plt.contourf(xx, yy, grid_labels.reshape(xx.shape), alpha=0.3, cmap='viridis')

    # Plot data points with noise and labels
    plt.scatter(noisy_input_data[:, 0], noisy_input_data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Class')
    plt.title(f'Data Points with Decision Boundary\n{title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.show()




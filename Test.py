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
        torch.manual_seed(2) # 2, 24 and 26 are good
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

# Define the sizes
input_size = 2
hidden_size1 = 16
hidden_size2 = 16
output_size = 6

# Set the seed for PyTorch
torch.manual_seed(14)

# Create an instance of the neural network
model = RandomNeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

# Generate uniformly sampled input data
num_samples = 500
input_data = torch.rand(num_samples, input_size) * 5 - 2.5  # Scale to range [-3.5, 3.5]

# Get model predictions for the input data
output = model(input_data)
predicted_labels = output.argmax(dim=1)

# Add Gaussian noise to the data points
noise_mean = 0
noise_std = 0.4
noisy_input_data = input_data + torch.randn_like(input_data) * noise_std + noise_mean

# Plot data points with noise
plt.figure(figsize=(8, 6))
plt.scatter(noisy_input_data[:, 0], noisy_input_data[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
plt.colorbar(label='Class')
plt.title('Data Points with Predicted Labels and Gaussian Noise')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot decision boundaries
x_min, x_max = -4, 4
y_min, y_max = -4, 4
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_input = np.c_[xx.ravel(), yy.ravel()]
grid_input = torch.tensor(grid_input, dtype=torch.float32)
grid_output = model(grid_input)
grid_labels = grid_output.argmax(axis=1)
plt.contourf(xx, yy, grid_labels.reshape(xx.shape), alpha=0.3, cmap='viridis')

plt.show()

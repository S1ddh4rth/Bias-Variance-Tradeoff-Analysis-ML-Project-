# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
from Neural_Network import *
# import Test as NN

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Define the sizes
    input_size = 2
    hidden_size = 20
    output_size = 6

    # Create an instance of the original neural network
    original_model = RandomNeuralNetwork(input_size, hidden_size, hidden_size, output_size)

    # Generate uniformly sampled input data with noise
    num_samples = 500
    noise_mean = 0
    noise_std = 0.4
    input_data, noisy_input_data = generate_data_with_noise(input_size, num_samples, noise_mean, noise_std)

    # Get original model predictions for the input data
    original_output = original_model(input_data)
    original_predicted_labels = original_output.argmax(dim=1)

    # Plot data points with noise and labels
    plot_data_with_labels(noisy_input_data, original_predicted_labels, "Ground truth decision boundary")

    # Plot original decision boundary
    plot_data_with_decision_boundary(original_model, input_data, noisy_input_data, original_predicted_labels,
                                      "Ground truth decision boundary")

    # Train and plot 5 models of increasing complexity
    hidden_size = 0
    for i in range(9):
        # Increase the hidden size for each subsequent model
        hidden_size += 4
        # Create an instance of the neural network with increased complexity
        model = RandomNeuralNetwork(input_size, hidden_size, hidden_size, output_size)

        # Training parameters
        labels = torch.LongTensor(original_predicted_labels)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        losses = train_model(model, input_data, noisy_input_data, labels, criterion, optimizer)

        # Plot data points with decision boundary
        plot_data_with_decision_boundary(model, input_data, noisy_input_data, original_predicted_labels,
                                         f"Hidden layer size: {hidden_size}")

        # Plot the training loss
        # plot_training_loss(losses)

if __name__ == "__main__":
    main()
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
import Neural_Network as NN
# import Test as NN

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Define the sizes
    input_size = 2
    hidden_size1 = 16
    hidden_size2 = 16
    output_size = 6

    # Set random seeds for reproducibility
    torch.manual_seed(14)
    np.random.seed(14)

    # Create an instance of the neural network
    model = NN.RandomNeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

    # Generate uniformly sampled input data with noise
    num_samples = 500
    noise_mean = 0
    noise_std = 0.4
    input_data, noisy_input_data = NN.generate_data_with_noise(input_size, num_samples, noise_mean, noise_std)

    # Get model predictions for the input data
    output = model(input_data)
    predicted_labels = output.argmax(dim=1)

    # Plot data points with noise and labels
    NN.plot_data_with_labels(noisy_input_data, predicted_labels)

    # Plot original decision boundary
    NN.plot_decision_boundaries(model, input_data) ######################################################################

    # Training parameters
    labels = torch.LongTensor(predicted_labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#-----------------------------------------------------------------------------------------------------------------------------------------
    # Train the model
    losses = NN.train_model(model, input_data, noisy_input_data, labels, criterion, optimizer) ################################

    # Plot the training loss
    NN.plot_training_loss(losses) ###################################

    # Plot decision boundaries after training
    NN.plot_decision_boundaries(model, input_data) ########################################################

    # Plot data points with noise and labels again to overlay decision boundaries
    NN.plot_data_with_labels(noisy_input_data, predicted_labels)




if __name__ == "__main__":
    main()


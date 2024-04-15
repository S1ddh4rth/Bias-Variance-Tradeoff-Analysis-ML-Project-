import numpy as np
import torch
from Neural_Network import *
from sklearn.model_selection import train_test_split

def main():
    # Define the sizes
    input_size = 2
    hidden_size = 8
    output_size = 4

    # Create an instance of the original neural network
    original_model = RandomNeuralNetwork(input_size, hidden_size, hidden_size, output_size)

    # Generate uniformly sampled input data with noise
    num_samples = 1000
    noise_mean = 0
    noise_std = 0.5
    input_data, noisy_input_data = generate_data_with_noise(input_size, num_samples, noise_mean, noise_std)

    # Get original model predictions for the input data
    original_output = original_model(input_data)
    original_predicted_labels = original_output.argmax(dim=1)

    # Split data into training and test sets
    input_train, input_test, labels_train, labels_test = train_test_split(noisy_input_data, original_predicted_labels, test_size=0.2, random_state=42)

    # Plot original train data with labels
    plot_data_with_labels(input_train, labels_train, title="Original Train Data with Labels")

    # Plot original test data with labels
    plot_data_with_labels(input_test, labels_test, title="Original Test Data with Labels")

    # Plot decision boundary of the original model
    plot_data_with_decision_boundary(original_model, input_data, noisy_input_data, original_predicted_labels, title="Decision Boundary of Original Model")

    # Train and plot 5 models of increasing complexity
    hidden_sizes = []
    train_losses = []
    test_losses = []

    hidden_size = 0
    for i in range(9):
        # Increase the hidden size for each subsequent model
        hidden_size += 4
        hidden_sizes.append(hidden_size)

        # Create an instance of the neural network with increased complexity
        model = RandomNeuralNetwork(input_size, hidden_size, hidden_size, output_size)

        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model on the training set
        train_loss = train_model(model, input_train, labels_train, criterion, optimizer)
        train_losses.append(train_loss[-1])  # Append the final training loss

        # Test the model on the test set
        # Using sourceTensor.clone().detach() instead of torch.tensor(sourceTensor)
        test_output = model(input_test.clone().detach())
        test_loss = criterion(test_output, labels_test.clone().detach())

        test_losses.append(test_loss.item())

        # Overlay data points onto decision boundaries
        plot_data_with_decision_boundary(model, input_data, noisy_input_data, original_predicted_labels,
                                         title=f"Decision Boundary with Hidden layer size: {hidden_size}")

    # Plot error vs hidden_layer size graph
    plt.figure(figsize=(12, 6))

    # Plot training error
    plt.subplot(1, 2, 1)
    plt.plot(hidden_sizes, train_losses, marker='o', color='blue', label='Train Error')
    plt.title("Train Error vs Complexity")
    plt.xlabel("Complexity (Hidden Layer Size)")
    plt.ylabel("Train Error")
    plt.grid(True)
    plt.legend()

    # Plot test error
    plt.subplot(1, 2, 2)
    plt.plot(hidden_sizes, test_losses, marker='o', color='orange', label='Test Error')
    plt.title("Test Error vs Complexity")
    plt.xlabel("Complexity (Hidden Layer Size)")
    plt.ylabel("Test Error")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

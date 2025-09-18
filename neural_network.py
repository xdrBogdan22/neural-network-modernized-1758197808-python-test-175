#!/usr/bin/env python3
"""
Neural Network Implementation from Scratch

This module contains the core neural network implementation using pure Python
without external machine learning libraries.
"""

import random
import math

class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.

    Uses sigmoid activation function and backpropagation for training.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network.

        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output classes
            learning_rate (float): Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights randomly
        self.weights_input_hidden = self._initialize_weights(input_size, hidden_size)
        self.weights_hidden_output = self._initialize_weights(hidden_size, output_size)

        # Initialize biases
        self.bias_hidden = [0.0] * hidden_size
        self.bias_output = [0.0] * output_size

    def _initialize_weights(self, rows, cols):
        """Initialize weights with small random values."""
        return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            # Handle overflow by returning appropriate boundary values
            return 0.0 if x < 0 else 1.0

    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1.0 - x)

    def _forward_pass(self, inputs):
        """
        Perform forward pass through the network.

        Args:
            inputs (list): Input features

        Returns:
            tuple: (hidden_outputs, final_outputs)
        """
        # Calculate hidden layer outputs
        hidden_inputs = []
        for j in range(self.hidden_size):
            weighted_sum = sum(inputs[i] * self.weights_input_hidden[i][j]
                             for i in range(self.input_size))
            hidden_inputs.append(weighted_sum + self.bias_hidden[j])

        hidden_outputs = [self._sigmoid(x) for x in hidden_inputs]

        # Calculate output layer outputs
        output_inputs = []
        for k in range(self.output_size):
            weighted_sum = sum(hidden_outputs[j] * self.weights_hidden_output[j][k]
                             for j in range(self.hidden_size))
            output_inputs.append(weighted_sum + self.bias_output[k])

        final_outputs = [self._sigmoid(x) for x in output_inputs]

        return hidden_outputs, final_outputs

    def _backward_pass(self, inputs, hidden_outputs, final_outputs, expected_outputs):
        """
        Perform backward pass (backpropagation) to update weights.

        Args:
            inputs (list): Input features
            hidden_outputs (list): Hidden layer outputs from forward pass
            final_outputs (list): Final outputs from forward pass
            expected_outputs (list): Expected target outputs
        """
        # Calculate output layer errors
        output_errors = []
        for k in range(self.output_size):
            error = expected_outputs[k] - final_outputs[k]
            output_errors.append(error * self._sigmoid_derivative(final_outputs[k]))

        # Calculate hidden layer errors
        hidden_errors = []
        for j in range(self.hidden_size):
            error_sum = sum(output_errors[k] * self.weights_hidden_output[j][k]
                           for k in range(self.output_size))
            hidden_errors.append(error_sum * self._sigmoid_derivative(hidden_outputs[j]))

        # Update weights and biases for hidden to output layer
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += (self.learning_rate *
                                                   output_errors[k] * hidden_outputs[j])

        for k in range(self.output_size):
            self.bias_output[k] += self.learning_rate * output_errors[k]

        # Update weights and biases for input to hidden layer
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += (self.learning_rate *
                                                  hidden_errors[j] * inputs[i])

        for j in range(self.hidden_size):
            self.bias_hidden[j] += self.learning_rate * hidden_errors[j]

    def _one_hot_encode(self, class_label, num_classes):
        """Convert class label to one-hot encoded vector."""
        one_hot = [0.0] * num_classes
        one_hot[int(class_label)] = 1.0
        return one_hot

    def train(self, X_train, y_train, epochs=1000, verbose=True):
        """
        Train the neural network using backpropagation.

        Args:
            X_train (list): Training features
            y_train (list): Training labels
            epochs (int): Number of training epochs
            verbose (bool): Whether to print training progress
        """
        num_classes = len(set(y_train))

        for epoch in range(epochs):
            total_error = 0.0

            for i in range(len(X_train)):
                # Forward pass
                hidden_outputs, final_outputs = self._forward_pass(X_train[i])

                # Convert label to one-hot encoding
                expected_outputs = self._one_hot_encode(y_train[i], num_classes)

                # Calculate error for this sample
                sample_error = sum((expected_outputs[j] - final_outputs[j]) ** 2
                                 for j in range(len(expected_outputs)))
                total_error += sample_error

                # Backward pass
                self._backward_pass(X_train[i], hidden_outputs, final_outputs, expected_outputs)

            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                avg_error = total_error / len(X_train)
                print(f"         Epoch {epoch + 1}/{epochs}, Average Error: {avg_error:.6f}")

    def predict(self, inputs):
        """
        Make a prediction for given inputs.

        Args:
            inputs (list): Input features

        Returns:
            int: Predicted class label
        """
        _, outputs = self._forward_pass(inputs)
        return outputs.index(max(outputs))

    def evaluate(self, X_test, y_test):
        """
        Evaluate the network on test data.

        Args:
            X_test (list): Test features
            y_test (list): Test labels

        Returns:
            float: Accuracy score
        """
        correct = 0
        total = len(X_test)

        for i in range(total):
            prediction = self.predict(X_test[i])
            if prediction == int(y_test[i]):
                correct += 1

        return correct / total if total > 0 else 0.0
# Neural Network Implementation from Scratch

This project is an educational tool demonstrating neural network implementation for classification tasks without external machine learning libraries. It's designed for students, developers, and machine learning enthusiasts seeking to understand neural network fundamentals.

## Features

- Pure Python implementation of a feedforward neural network with backpropagation
- Configurable network parameters (learning rate, epochs, hidden neurons)
- Cross-validation for model evaluation
- Data preprocessing and normalization
- Console output for training progress and results

## Project Structure

- `main.py`: Entry point for the application
- `neural_network.py`: Core neural network implementation
- `data_processor.py`: Data loading and preprocessing utilities
- `cross_validation.py`: Cross-validation implementation
- `utils.py`: Helper functions
- `data/`: Directory containing the dataset

## Usage

```bash
# Run the neural network with default parameters
python main.py

# Run with custom parameters
python main.py --learning_rate 0.1 --epochs 500 --hidden_neurons 5 --folds 5
```

## Requirements

- Python 3.6+
- No external machine learning libraries required

## Dataset

The project uses the Wheat Seeds dataset, which contains measurements of wheat seeds from different varieties. The dataset includes 7 measurement features and a class label indicating the wheat variety.

## Educational Purpose

This implementation prioritizes clarity and educational value over performance. The code is designed to be transparent and easy to understand, making it an ideal learning tool for understanding how neural networks work at a fundamental level.
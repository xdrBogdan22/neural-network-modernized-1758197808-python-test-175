#!/usr/bin/env python3
"""
Neural Network Implementation from Scratch - Main Entry Point

This is the main entry point for the neural network implementation.
It provides command-line interface for training and evaluating the neural network.
"""

import argparse
import sys
from neural_network import NeuralNetwork
from data_processor import DataProcessor
from cross_validation import CrossValidation

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Neural Network Implementation from Scratch')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for training (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs (default: 1000)')
    parser.add_argument('--hidden_neurons', type=int, default=10,
                       help='Number of neurons in hidden layer (default: 10)')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--data_file', type=str, default='data/seeds_dataset.txt',
                       help='Path to dataset file (default: data/seeds_dataset.txt)')

    return parser.parse_args()

def main():
    """Main function to run neural network training and evaluation."""
    print("🧠 Neural Network Implementation from Scratch")
    print("=" * 50)

    # Parse command-line arguments
    args = parse_arguments()

    # Initialize data processor
    print("📊 Loading and preprocessing data...")
    try:
        data_processor = DataProcessor()
        X, y = data_processor.load_data(args.data_file)
        X_normalized = data_processor.normalize_features(X)

        print(f"   ✅ Loaded {len(X)} samples with {len(X[0])} features")
        print(f"   ✅ Found {len(set(y))} classes")

    except FileNotFoundError:
        print(f"   ❌ Dataset file not found: {args.data_file}")
        print(f"   💡 Please ensure the dataset file exists or specify correct path with --data_file")
        sys.exit(1)
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        sys.exit(1)

    # Initialize neural network
    print(f"\n🏗️  Initializing neural network...")
    print(f"   • Input features: {len(X[0])}")
    print(f"   • Hidden neurons: {args.hidden_neurons}")
    print(f"   • Output classes: {len(set(y))}")
    print(f"   • Learning rate: {args.learning_rate}")

    # Perform cross-validation
    print(f"\n🔄 Starting {args.folds}-fold cross-validation...")
    cv = CrossValidation(args.folds)

    accuracies = []
    for fold in range(args.folds):
        print(f"\n   📂 Fold {fold + 1}/{args.folds}")

        # Split data for this fold
        X_train, X_val, y_train, y_val = cv.get_fold_data(X_normalized, y, fold)

        # Create and train neural network
        nn = NeuralNetwork(
            input_size=len(X[0]),
            hidden_size=args.hidden_neurons,
            output_size=len(set(y)),
            learning_rate=args.learning_rate
        )

        # Train the network
        print(f"      🏋️  Training for {args.epochs} epochs...")
        nn.train(X_train, y_train, epochs=args.epochs, verbose=False)

        # Evaluate on validation set
        accuracy = nn.evaluate(X_val, y_val)
        accuracies.append(accuracy)

        print(f"      ✅ Validation accuracy: {accuracy:.4f}")

    # Display final results
    print(f"\n📈 Cross-Validation Results:")
    print(f"   • Individual accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"   • Mean accuracy: {sum(accuracies)/len(accuracies):.4f}")
    print(f"   • Standard deviation: {(sum([(acc - sum(accuracies)/len(accuracies))**2 for acc in accuracies]) / len(accuracies))**0.5:.4f}")

    print(f"\n🎉 Neural network training completed successfully!")

if __name__ == "__main__":
    main()
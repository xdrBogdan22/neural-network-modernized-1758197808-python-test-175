#!/usr/bin/env python3
"""
Cross-Validation Module for Neural Network Implementation

This module provides k-fold cross-validation functionality for evaluating
the neural network model performance.
"""

import random

class CrossValidation:
    """
    Implements k-fold cross-validation for model evaluation.
    """

    def __init__(self, k_folds=5, random_seed=42):
        """
        Initialize cross-validation.

        Args:
            k_folds (int): Number of folds for cross-validation
            random_seed (int): Random seed for reproducible results
        """
        self.k_folds = k_folds
        self.random_seed = random_seed
        self.fold_indices = None

    def create_folds(self, data_size):
        """
        Create fold indices for cross-validation.

        Args:
            data_size (int): Total number of data samples

        Returns:
            list: List of fold indices
        """
        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Create list of indices and shuffle
        indices = list(range(data_size))
        random.shuffle(indices)

        # Calculate fold size
        fold_size = data_size // self.k_folds
        remainder = data_size % self.k_folds

        # Create folds
        folds = []
        start_idx = 0

        for i in range(self.k_folds):
            # Add one extra sample to some folds if there's a remainder
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_fold_size

            fold_indices = indices[start_idx:end_idx]
            folds.append(fold_indices)

            start_idx = end_idx

        self.fold_indices = folds
        return folds

    def get_fold_data(self, features, labels, fold_index):
        """
        Get training and validation data for a specific fold.

        Args:
            features (list): All feature vectors
            labels (list): All labels
            fold_index (int): Index of the fold to use as validation set

        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        if self.fold_indices is None:
            self.create_folds(len(features))

        if fold_index >= self.k_folds:
            raise ValueError(f"Fold index {fold_index} exceeds number of folds {self.k_folds}")

        # Get validation indices for current fold
        val_indices = set(self.fold_indices[fold_index])

        # Create training and validation sets
        X_train = []
        X_val = []
        y_train = []
        y_val = []

        for i in range(len(features)):
            if i in val_indices:
                X_val.append(features[i])
                y_val.append(labels[i])
            else:
                X_train.append(features[i])
                y_train.append(labels[i])

        return X_train, X_val, y_train, y_val

    def get_fold_info(self):
        """
        Get information about the current fold configuration.

        Returns:
            dict: Dictionary containing fold information
        """
        if self.fold_indices is None:
            return {}

        fold_sizes = [len(fold) for fold in self.fold_indices]

        return {
            'num_folds': self.k_folds,
            'fold_sizes': fold_sizes,
            'total_samples': sum(fold_sizes),
            'random_seed': self.random_seed
        }

    def stratified_split(self, features, labels, fold_index):
        """
        Create stratified fold that maintains class distribution.

        Args:
            features (list): All feature vectors
            labels (list): All labels
            fold_index (int): Index of the fold to use as validation set

        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        # Group samples by class
        class_samples = {}
        for i, label in enumerate(labels):
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(i)

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Shuffle samples within each class
        for class_label in class_samples:
            random.shuffle(class_samples[class_label])

        # Create stratified folds
        val_indices = set()
        for class_label, sample_indices in class_samples.items():
            class_fold_size = len(sample_indices) // self.k_folds
            class_remainder = len(sample_indices) % self.k_folds

            # Calculate validation indices for this class
            start_idx = fold_index * class_fold_size + min(fold_index, class_remainder)
            current_fold_size = class_fold_size + (1 if fold_index < class_remainder else 0)
            end_idx = start_idx + current_fold_size

            # Add validation indices for this fold
            for i in range(start_idx, min(end_idx, len(sample_indices))):
                val_indices.add(sample_indices[i])

        # Create training and validation sets
        X_train = []
        X_val = []
        y_train = []
        y_val = []

        for i in range(len(features)):
            if i in val_indices:
                X_val.append(features[i])
                y_val.append(labels[i])
            else:
                X_train.append(features[i])
                y_train.append(labels[i])

        return X_train, X_val, y_train, y_val

    def evaluate_model(self, model_class, features, labels, **model_params):
        """
        Evaluate a model using k-fold cross-validation.

        Args:
            model_class: Class of the model to evaluate
            features (list): All feature vectors
            labels (list): All labels
            **model_params: Parameters to pass to model constructor

        Returns:
            dict: Dictionary containing evaluation results
        """
        accuracies = []
        fold_results = []

        for fold in range(self.k_folds):
            # Get fold data
            X_train, X_val, y_train, y_val = self.get_fold_data(features, labels, fold)

            # Create and train model
            model = model_class(**model_params)
            model.train(X_train, y_train)

            # Evaluate model
            accuracy = model.evaluate(X_val, y_val)
            accuracies.append(accuracy)

            fold_results.append({
                'fold': fold,
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            })

        # Calculate statistics
        mean_accuracy = sum(accuracies) / len(accuracies)
        std_accuracy = (sum([(acc - mean_accuracy) ** 2 for acc in accuracies]) / len(accuracies)) ** 0.5

        return {
            'fold_accuracies': accuracies,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'fold_details': fold_results,
            'num_folds': self.k_folds
        }
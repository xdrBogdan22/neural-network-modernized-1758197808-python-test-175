#!/usr/bin/env python3
"""
Data Processing Module for Neural Network Implementation

This module handles data loading, preprocessing, and normalization
for the neural network implementation.
"""

import os
import math

class DataProcessor:
    """
    Handles data loading and preprocessing operations.
    """

    def __init__(self):
        """Initialize the data processor."""
        pass

    def load_data(self, file_path):
        """
        Load dataset from file.

        Args:
            file_path (str): Path to the dataset file

        Returns:
            tuple: (features, labels) where features is a list of feature vectors
                   and labels is a list of class labels

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If the data format is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        features = []
        labels = []

        try:
            with open(file_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue

                    # Split line into components
                    parts = line.split()
                    if len(parts) < 2:
                        continue  # Skip invalid lines

                    try:
                        # Last column is the label, others are features
                        feature_vector = [float(x) for x in parts[:-1]]
                        label = int(float(parts[-1])) - 1  # Convert to 0-indexed

                        features.append(feature_vector)
                        labels.append(label)

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping invalid line {line_num}: {line}")
                        continue

        except IOError as e:
            raise ValueError(f"Error reading dataset file: {e}")

        if not features:
            raise ValueError("No valid data found in the dataset file")

        print(f"   📊 Loaded {len(features)} samples")
        print(f"   🔢 Feature dimensions: {len(features[0]) if features else 0}")
        print(f"   🏷️  Classes: {sorted(set(labels))}")

        return features, labels

    def normalize_features(self, features):
        """
        Normalize features using min-max normalization.

        Args:
            features (list): List of feature vectors

        Returns:
            list: Normalized feature vectors (values between 0 and 1)
        """
        if not features:
            return features

        num_features = len(features[0])
        normalized_features = []

        # Calculate min and max for each feature
        min_values = [float('inf')] * num_features
        max_values = [float('-inf')] * num_features

        for feature_vector in features:
            for i, value in enumerate(feature_vector):
                min_values[i] = min(min_values[i], value)
                max_values[i] = max(max_values[i], value)

        # Normalize each feature vector
        for feature_vector in features:
            normalized_vector = []
            for i, value in enumerate(feature_vector):
                # Min-max normalization: (x - min) / (max - min)
                if max_values[i] != min_values[i]:
                    normalized_value = (value - min_values[i]) / (max_values[i] - min_values[i])
                else:
                    normalized_value = 0.0  # Handle case where all values are the same

                normalized_vector.append(normalized_value)

            normalized_features.append(normalized_vector)

        return normalized_features

    def standardize_features(self, features):
        """
        Standardize features using z-score normalization.

        Args:
            features (list): List of feature vectors

        Returns:
            list: Standardized feature vectors (mean=0, std=1)
        """
        if not features:
            return features

        num_features = len(features[0])
        standardized_features = []

        # Calculate mean and standard deviation for each feature
        means = [0.0] * num_features
        stds = [0.0] * num_features

        # Calculate means
        for feature_vector in features:
            for i, value in enumerate(feature_vector):
                means[i] += value

        for i in range(num_features):
            means[i] /= len(features)

        # Calculate standard deviations
        for feature_vector in features:
            for i, value in enumerate(feature_vector):
                stds[i] += (value - means[i]) ** 2

        for i in range(num_features):
            stds[i] = math.sqrt(stds[i] / len(features))

        # Standardize each feature vector
        for feature_vector in features:
            standardized_vector = []
            for i, value in enumerate(feature_vector):
                # Z-score normalization: (x - mean) / std
                if stds[i] != 0:
                    standardized_value = (value - means[i]) / stds[i]
                else:
                    standardized_value = 0.0  # Handle case where std is 0

                standardized_vector.append(standardized_value)

            standardized_features.append(standardized_vector)

        return standardized_features

    def split_data(self, features, labels, train_ratio=0.8, random_seed=42):
        """
        Split data into training and testing sets.

        Args:
            features (list): Feature vectors
            labels (list): Corresponding labels
            train_ratio (float): Ratio of data to use for training
            random_seed (int): Random seed for reproducible splits

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        import random

        # Set random seed for reproducibility
        random.seed(random_seed)

        # Create list of indices and shuffle
        indices = list(range(len(features)))
        random.shuffle(indices)

        # Calculate split point
        split_point = int(len(features) * train_ratio)

        # Split indices
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        # Create training and testing sets
        X_train = [features[i] for i in train_indices]
        X_test = [features[i] for i in test_indices]
        y_train = [labels[i] for i in train_indices]
        y_test = [labels[i] for i in test_indices]

        return X_train, X_test, y_train, y_test

    def get_data_statistics(self, features, labels):
        """
        Get basic statistics about the dataset.

        Args:
            features (list): Feature vectors
            labels (list): Corresponding labels

        Returns:
            dict: Dictionary containing dataset statistics
        """
        if not features:
            return {}

        num_samples = len(features)
        num_features = len(features[0])
        unique_labels = sorted(set(labels))

        # Calculate class distribution
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Feature statistics
        feature_stats = []
        for i in range(num_features):
            feature_values = [feature_vector[i] for feature_vector in features]
            feature_stats.append({
                'min': min(feature_values),
                'max': max(feature_values),
                'mean': sum(feature_values) / len(feature_values),
            })

        return {
            'num_samples': num_samples,
            'num_features': num_features,
            'num_classes': len(unique_labels),
            'class_labels': unique_labels,
            'class_distribution': class_counts,
            'feature_statistics': feature_stats
        }
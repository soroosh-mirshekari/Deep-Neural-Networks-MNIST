# Deep Neural Networks on MNIST

## Overview
This project presents an end-to-end implementation and evaluation of a deep neural network for handwritten digit classification using the MNIST dataset. The focus of the project is on understanding the internal mechanics of neural networks and training dynamics rather than relying on high-level abstractions.

## Objective
The primary goals of this project are to:
- design and implement a fully connected deep neural network,
- study the impact of architectural depth on learning behavior,
- apply regularization techniques to improve training stability and generalization.

## Approach
The workflow consists of:
- preprocessing grayscale image data and transforming it into normalized feature
  vectors,
- designing a multi-layer neural network with several hidden layers,
- training the model using gradient-based optimization,
- incorporating techniques such as batch normalization, dropout, and weight
  regularization to stabilize learning.

The implementation emphasizes clarity, modularity, and reproducibility.

## Key Concepts
deep neural networks • supervised learning • backpropagation •
regularization techniques • optimization methods • training stability

## Dataset
The project uses the MNIST dataset, a widely adopted benchmark for evaluating
classification models on handwritten digit images.

## Challenges
Key challenges addressed in this project include:
- selecting appropriate hyperparameters to balance convergence and stability,
- mitigating overfitting in deep architectures,
- handling training instabilities caused by the interaction of normalization and
  regularization techniques.

## How to Run
1. Ensure required Python dependencies are installed  
2. Run the main training script or notebook:
   ```bash
   python NEURAL_NETWORKS_AI.py

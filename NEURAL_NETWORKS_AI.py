import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).T / 255.0
x_test = x_test.reshape(-1, 28*28).T / 255.0

def one_hot(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot.T

y_train = one_hot(y_train)
y_test = one_hot(y_test)

# Create validation set
val_size = 10000
x_val = x_train[:, -val_size:]
y_val = y_train[:, -val_size:]
x_train = x_train[:, :-val_size]
y_train = y_train[:, :-val_size]

class NeuralNetwork:
    def __init__(self, layer_sizes, keep_prob=0.8):
        self.layer_sizes = layer_sizes
        self.keep_prob = keep_prob
        self.parameters = {}
        self.initialize_parameters()
        
    def initialize_parameters(self):
        for l in range(1, len(self.layer_sizes)):
            n_in = self.layer_sizes[l-1]
            n_out = self.layer_sizes[l]
            self.parameters[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(2./n_in)
            self.parameters[f'b{l}'] = np.zeros((n_out, 1))
            if l < len(self.layer_sizes) - 1:
                self.parameters[f'gamma{l}'] = np.ones((n_out, 1))
                self.parameters[f'beta{l}'] = np.zeros((n_out, 1))
    
    def batch_norm(self, Z, l):
        eps = 1e-8
        gamma = self.parameters[f'gamma{l}']
        beta = self.parameters[f'beta{l}']
        mu = np.mean(Z, axis=1, keepdims=True)
        var = np.var(Z, axis=1, keepdims=True)
        Z_norm = (Z - mu) / np.sqrt(var + eps)
        return gamma * Z_norm + beta
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def drelu(self, Z):
        return (Z > 0).astype(float)
    
    def softmax(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return e_Z / e_Z.sum(axis=0, keepdims=True)
    
    def forward(self, X, training=True):
        caches = {'A0': X}
        L = len(self.layer_sizes) - 1
        
        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, caches[f'A{l-1}']) + b
            Z_norm = self.batch_norm(Z, l)
            A = self.relu(Z_norm)
            
            if training and self.keep_prob < 1:
                D = (np.random.rand(*A.shape) < self.keep_prob).astype(float)
                A *= D
                A /= self.keep_prob
                caches[f'D{l}'] = D
                
            caches[f'Z{l}'] = Z
            caches[f'Z_norm{l}'] = Z_norm
            caches[f'A{l}'] = A
        
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        Z = np.dot(W, caches[f'A{L-1}']) + b
        A = self.softmax(Z)
        caches[f'Z{L}'] = Z
        caches[f'A{L}'] = A
        
        return A, caches
    
    def cross_entropy_loss(self, A, Y):
        m = Y.shape[1]
        return -np.sum(Y * np.log(A + 1e-15)) / m
    
    def backward(self, X, Y, caches, lmbda=0.05):
        grads = {}
        m = Y.shape[1]
        L = len(self.layer_sizes) - 1
        
        dZ = (caches[f'A{L}'] - Y) / m
        grads[f'dW{L}'] = np.dot(dZ, caches[f'A{L-1}'].T) + (lmbda * self.parameters[f'W{L}'])/m
        grads[f'db{L}'] = np.sum(dZ, axis=1, keepdims=True)
        
        for l in reversed(range(1, L)):
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZ)
            
            if f'D{l}' in caches:
                dA *= caches[f'D{l}']
                dA /= self.keep_prob
            
            Z = caches[f'Z{l}']
            Z_norm = caches[f'Z_norm{l}']
            gamma = self.parameters[f'gamma{l}']
            
            dZ_norm = dA * self.drelu(Z_norm)
            sigma = np.sqrt(np.var(Z, axis=1, keepdims=True) + 1e-8)
            mu = np.mean(Z, axis=1, keepdims=True)
            
            dsigma = np.sum(dZ_norm * (Z - mu) * (-0.5) * sigma**-3, axis=1, keepdims=True)
            dmu = np.sum(dZ_norm * (-1/sigma), axis=1, keepdims=True) + dsigma * np.mean(-2 * (Z - mu), axis=1, keepdims=True)
            dZ = (dZ_norm / sigma) + dsigma * 2 * (Z - mu) / m + dmu / m
            
            grads[f'dgamma{l}'] = np.sum(dA * Z_norm, axis=1, keepdims=True)
            grads[f'dbeta{l}'] = np.sum(dA, axis=1, keepdims=True)
            grads[f'dW{l}'] = np.dot(dZ, caches[f'A{l-1}'].T) + (lmbda * self.parameters[f'W{l}'])/m
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True)
        
        return grads
    
    def update_parameters(self, grads, learning_rate, t, beta1=0.9, beta2=0.999):
        L = len(self.layer_sizes) - 1
        eps = 1e-8
        
        if not hasattr(self, 'v'):
            self.v = {}
            self.s = {}
            for l in range(1, L+1):
                self.v[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.v[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
                self.s[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.s[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
                if l < L:
                    self.v[f'dgamma{l}'] = np.zeros_like(self.parameters[f'gamma{l}'])
                    self.v[f'dbeta{l}'] = np.zeros_like(self.parameters[f'beta{l}'])
                    self.s[f'dgamma{l}'] = np.zeros_like(self.parameters[f'gamma{l}'])
                    self.s[f'dbeta{l}'] = np.zeros_like(self.parameters[f'beta{l}'])
        
        for l in range(1, L+1):
            for param in ['W', 'b']:
                key = f'd{param}{l}'
                self.v[key] = beta1 * self.v[key] + (1 - beta1) * grads[key]
                self.s[key] = beta2 * self.s[key] + (1 - beta2) * (grads[key]**2)
                
                v_corrected = self.v[key] / (1 - beta1**t)
                s_corrected = self.s[key] / (1 - beta2**t)
                
                self.parameters[f'{param}{l}'] -= learning_rate * v_corrected / (np.sqrt(s_corrected) + eps)
            
            if l < L:
                for param in ['gamma', 'beta']:
                    key = f'd{param}{l}'
                    self.v[key] = beta1 * self.v[key] + (1 - beta1) * grads[key]
                    self.s[key] = beta2 * self.s[key] + (1 - beta2) * (grads[key]**2)
                    
                    v_corrected = self.v[key] / (1 - beta1**t)
                    s_corrected = self.s[key] / (1 - beta2**t)
                    
                    self.parameters[f'{param}{l}'] -= learning_rate * v_corrected / (np.sqrt(s_corrected) + eps)

    def train(self, X, Y, learning_rate=0.001, epochs=2000, lmbda=0.05, batch_size=256):
        costs = []
        train_accuracies = []
        m = X.shape[1]
        
        for i in range(1, epochs+1):
            batch_indices = np.random.choice(m, batch_size, replace=False)
            X_batch = X[:, batch_indices]
            Y_batch = Y[:, batch_indices]
            
            AL, caches = self.forward(X_batch, training=True)
            grads = self.backward(X_batch, Y_batch, caches, lmbda)
            
            data_loss = self.cross_entropy_loss(AL, Y_batch)
            reg_loss = 0.5 * lmbda * sum(np.sum(self.parameters[f'W{l}']**2) 
                                       for l in range(1, len(self.layer_sizes))) / m
            total_cost = data_loss + reg_loss
            costs.append(total_cost)
            
            self.update_parameters(grads, learning_rate, i)
            
            if i % 100 == 0:
                AL, _ = self.forward(X_batch, training=False)
                accuracy = np.mean(np.argmax(AL, axis=0) == np.argmax(Y_batch, axis=0))
                train_accuracies.append(accuracy)
                print(f"Epoch {i}: Cost = {total_cost:.4f}, Accuracy = {accuracy:.4f}")
        
        return costs, train_accuracies

# Network architecture and training
layer_sizes = [784, 512, 256, 128, 64, 10]
nn = NeuralNetwork(layer_sizes, keep_prob=0.8)

costs, train_acc = nn.train(
    x_train, y_train,
    learning_rate=0.001,
    epochs=2000,
    lmbda=0.05,
    batch_size=256
)

# Evaluation
def evaluate(X, Y, nn):
    AL, _ = nn.forward(X, training=False)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y, axis=0)
    return np.mean(predictions == labels)

train_accuracy = evaluate(x_train, y_train, nn)
val_accuracy = evaluate(x_val, y_val, nn)
test_accuracy = evaluate(x_test, y_test, nn)

print("\nFinal Results:")
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot learning curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title("Training Cost")
plt.xlabel("Epoch")
plt.ylabel("Cost")

plt.subplot(1, 2, 2)
plt.plot(np.arange(100, 2001, 100), train_acc)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
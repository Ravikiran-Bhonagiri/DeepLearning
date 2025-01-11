# Deep Learning Mathematical Equations

In this document, we will cover the fundamental mathematical equations that govern the forward pass, backward pass, and key components in deep learning, along with a diagram illustrating how weights, biases, and activation functions interact.

## 1. **Forward Pass of a Neuron**

A neural network is composed of multiple neurons, where each neuron processes an input using a weighted sum followed by an activation function.

### Equation for a Single Neuron:

```math
    z = \sum_{i=1}^{n} w_i x_i + b
```

Where:
- `X` (\(\mathbf{X}\)) is a matrix of shape *n* × *m* where *n* is the number of features and *m* is the number of samples.
- `W` (\(\mathbf{W}\)) is a matrix of shape *k* × *n* where *k* is the number of neurons and *n* is the number of input features.
- `b` (\(\mathbf{b}\)) is a vector of shape *k* × 1 where *k* corresponds to the number of neurons.
- `Z` (\(\mathbf{Z}\)) is the output matrix of shape *k* × *m*, representing the result before the activation function is applied.


After applying the activation function \(f(\mathbf{Z})\), we get the activation output \(\mathbf{A}\).

## 4. **Loss Function**. **Loss Function**

The loss function quantifies the difference between the predicted output and the true output. Common loss functions include:

- **Mean Squared Error (MSE)** (for regression):

```math
    L = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
```

- **Cross-Entropy Loss** (for classification):

```math
    L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
```

Where:
- \( y_i \) = true label.
- \( \hat{y}_i \) = predicted label.

## 5. **Pseudocode for Forward and Backward Pass**

Below is a pseudocode representation of how weights are updated during the forward and backward pass using gradient descent:

### **Pseudocode:**

```
Initialize weights W, biases b
Learning rate eta

For each epoch:
    For each batch of training data (X, Y):
        # Forward Pass
        Z = W * X + b  # Linear combination of inputs and weights
        A = activation_function(Z)  # Apply activation function

        # Compute Loss
        Loss = loss_function(A, Y)  # Compute loss between predictions and true labels

        # Backward Pass
        dLoss/dA = derivative_of_loss(A, Y)  # Derivative of loss w.r.t. activation output
        dA/dZ = derivative_of_activation(Z)  # Derivative of activation function
        dLoss/dZ = dLoss/dA * dA/dZ  # Chain rule to compute derivative of loss w.r.t. Z

        dLoss/dW = dLoss/dZ * X.T  # Derivative of loss w.r.t. weights
        dLoss/db = sum(dLoss/dZ)  # Derivative of loss w.r.t. biases

        # Update weights and biases
        W = W - eta * dLoss/dW
        b = b - eta * dLoss/db

    Print epoch loss
```

## 6. **Explanation of Pseudocode Steps:**
1. **Initialization:** Start with random weights \( W \) and biases \( b \).
2. **Forward Pass:** Compute the weighted sum of inputs and pass it through the activation function.
3. **Loss Computation:** Calculate the loss to measure the performance of the model.
4. **Backward Pass:** Use the chain rule to compute gradients for weights and biases.
5. **Weight Update:** Update the weights and biases using the computed gradients and learning rate.

## 7. **Gradient Computation**

For each weight and bias, the gradients are computed using the chain rule:

```math
    \frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_i}
```

Where:
- \( \frac{\partial L}{\partial a} \) = derivative of the loss with respect to the activation.
- \( \frac{\partial a}{\partial z} \) = derivative of the activation with respect to the weighted sum.
- \( \frac{\partial z}{\partial w_i} \) = derivative of the weighted sum with respect to the weights.

## 8. **Example Backpropagation Steps**

1. Compute the loss \( L \).
2. Compute \( \frac{\partial L}{\partial a} \) for the output layer.
3. Propagate the gradients backward through the network.
4. Update the weights and biases using the computed gradients.

---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

        # To store activations for visualization
        self.hidden_features = None
    
    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        
    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.tanh(self.z2)

        # Store hidden features for visualization
        self.hidden_features = self.a1
        return self.a2


    def backward(self, X, y):
        # TODO: compute gradients using chain rule

        # TODO: update weights with gradient descent

        # TODO: store gradients for visualization
        m = X.shape[0]
        dz2 = (self.a2 - y) / m
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    hidden_features = mlp.hidden_features  # Hidden activations (n_samples, n_hidden_neurons)

    # --- Hidden Space Visualization ---
    ax_hidden.set_title("Hidden Layer Features")
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap="bwr", alpha=0.7
    )
    ax_hidden.set_xlabel("Hidden Neuron 1")
    ax_hidden.set_ylabel("Hidden Neuron 2")
    ax_hidden.set_zlabel("Hidden Neuron 3")

    # Hyperplane Visualization
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 50),
        np.linspace(-1, 1, 50)
    )
    hidden_hyperplane = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2[0]) / (mlp.W2[2] + 1e-5)
    ax_hidden.plot_surface(xx, yy, hidden_hyperplane, alpha=0.3, color="orange")

    # --- Input Space Decision Boundary ---
    ax_input.set_title("Input Space Decision Boundary")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    input_space = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(input_space)
    zz = predictions.reshape(xx.shape)
    ax_input.contourf(xx, yy, zz, alpha=0.7, cmap="bwr")
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors="k")

    # --- Gradient Visualization ---
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap="bwr",
        alpha=0.7,
    )
    ax_hidden.set_xlabel("Hidden Neuron 1")
    ax_hidden.set_ylabel("Hidden Neuron 2")
    ax_hidden.set_zlabel("Hidden Neuron 3")

    # Add a plane in the hidden space
    xx, yy = np.meshgrid(
        np.linspace(hidden_features[:, 0].min() - 1, hidden_features[:, 0].max() + 1, 50),
        np.linspace(hidden_features[:, 1].min() - 1, hidden_features[:, 1].max() + 1, 50),
    )
    if mlp.W2.shape[0] > 2:
        zz = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2[0]) / (mlp.W2[2] + 1e-5)
        ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color="orange", edgecolor="none")

    # --- Input Space Decision Boundary ---
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    input_grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(input_grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, predictions, levels=50, cmap="bwr", alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors="k")

    # --- Gradient Visualization ---
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")

    # Node Positions
    input_positions = [0.2, 0.8]  # Input layer (x1, x2)
    hidden_positions = [0.3, 0.5, 0.7]  # Hidden layer (h1, h2, h3)
    output_position = 0.5  # Output layer (y)

    # Draw connections from input to hidden layer
    for i, input_pos in enumerate(input_positions):
        for j, hidden_pos in enumerate(hidden_positions):
            weight = mlp.W1[i, j]
            grad_magnitude = abs(weight)
            ax_gradient.plot(
                [0.1, 0.5],  # X-coordinates for the line
                [input_pos, hidden_pos],  # Y-coordinates for the line
                color="purple",
                alpha=min(1.0, 0.2 + grad_magnitude),  # Line alpha capped at 1.0
                lw=0.5 + grad_magnitude,  # Line thickness capped at reasonable levels
            )

    # Draw connections from hidden to output layer
    for j, hidden_pos in enumerate(hidden_positions):
        weight = mlp.W2[j, 0]
        grad_magnitude = abs(weight)
        ax_gradient.plot(
            [0.5, 0.9],  # X-coordinates for the line
            [hidden_pos, output_position],  # Y-coordinates for the line
            color="purple",
            alpha=min(1.0, 0.2 + grad_magnitude),  # Line alpha capped at 1.0
            lw=0.5 + grad_magnitude,  # Line thickness capped at reasonable levels
        )

    # Draw input nodes
    for i, input_pos in enumerate(input_positions):
        ax_gradient.add_patch(Circle((0.1, input_pos), radius=0.05, color="blue", alpha=0.7))
        ax_gradient.text(0.1, input_pos, f"x{i+1}", fontsize=10, ha="center", color="white")

    # Draw hidden nodes
    for j, hidden_pos in enumerate(hidden_positions):
        ax_gradient.add_patch(Circle((0.5, hidden_pos), radius=0.05, color="blue", alpha=0.7))
        ax_gradient.text(0.5, hidden_pos, f"h{j+1}", fontsize=10, ha="center", color="white")

    # Draw output node
    ax_gradient.add_patch(Circle((0.9, output_position), radius=0.05, color="blue", alpha=0.7))
    ax_gradient.text(0.9, output_position, "y", fontsize=10, ha="center", color="white")

    # Adjust gradient visualization limits
    ax_gradient.set_xlim(0, 1.0)
    ax_gradient.set_ylim(0, 1.0)


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
import numpy as np

# Activation function (Step Function)
def activation_fn(x):
    return 1 if x >= 0 else 0

# Take input from the user
n = int(input("Enter number of training samples: "))
features = int(input("Enter number of features: "))
num_classes = int(input("Enter number of output classes: "))

# Input samples
X = []
T = []

print("Enter inputs followed by class label (space separated):")
for _ in range(n):
    *x, t = map(int, input().split())
    X.append(x)
    T.append(t)

X = np.array(X)
T = np.array(T)

# Learning rate
alpha = float(input("Enter learning rate (e.g., 0.1): "))

# Initialize weights and bias for each class
weights = np.zeros((num_classes, features))
bias = np.zeros(num_classes)

# One-vs-All Training
epoch = 0
while True:
    print(f"\nEpoch {epoch + 1}")
    error_count = 0
    for i in range(n):
        x_i = X[i]
        target_class = T[i]

        for cls in range(num_classes):
            # Convert target to binary for this class
            target = 1 if cls == target_class else 0

            # Compute net input
            y_in = np.dot(weights[cls], x_i) + bias[cls]
            y = activation_fn(y_in)

            # Update weights if misclassified
            if y != target:
                delta = alpha * (target - y)
                weights[cls] += delta * x_i
                bias[cls] += delta
                error_count += 1

            print(f"Class {cls}: Input: {x_i}, Target: {target}, Output: {y}, Weights: {weights[cls]}, Bias: {bias[cls]}")

    epoch += 1
    if error_count == 0:
        print("\nTraining complete. Final weights and biases per class:")
        for cls in range(num_classes):
            print(f"Class {cls} - Weights: {weights[cls]}, Bias: {bias[cls]}")
        break

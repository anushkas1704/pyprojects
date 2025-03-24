import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(2) 
        self.bias = 0

    def activation(self, x):
        return 1 if x >= 0 else 0 
    def train(self, X, y):
        for _ in range(self.epochs):
            weight_changed = False
            for xi, target in zip(X, y):
                net_input = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation(net_input)

          
                if y_pred != target:
                    update = self.learning_rate * (target - y_pred)
                    self.weights += update * xi
                    self.bias += update
                    weight_changed = True

            if not weight_changed:  
                break

    def predict(self, X):
        net_input = np.dot(X, self.weights) + self.bias
        return self.activation(net_input)



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
y = np.array([0, 0, 0, 1])  


perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.train(X, y)


print("AND Gate Predictions:")
for xi in X:
    print(f"Input: {xi} -> Predicted Output: {perceptron.predict(xi)}")
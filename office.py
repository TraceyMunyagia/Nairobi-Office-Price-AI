import numpy as np
import matplotlib.pyplot as plt


# Mean Squared Error Function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Gradient Descent Step Function
def gradient_descent_step(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * x + c
    dm = (-2 / n) * np.sum(x * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c


# Simulating training over 10 epochs
def train_model(x, y, epochs=10, learning_rate=0.01):
    m, c = np.random.rand(), np.random.rand()  # Initialize m and c randomly
    errors = []

    for epoch in range(epochs):
        # Predict and calculate error
        y_pred = m * x + c
        error = mean_squared_error(y, y_pred)
        errors.append(error)

        # Print error for each epoch
        print(f"Epoch {epoch + 1}, MSE: {error:.4f}")

        # Update m and c using gradient descent
        m, c = gradient_descent_step(x, y, m, c, learning_rate)

    return m, c, errors


# Example dataset (replace with your actual dataset values)
x = np.array([100, 200, 300, 400, 500])  # Example office sizes
y = np.array([250, 450, 600, 800, 1000])  # Example office prices

# Train the model
m, c, errors = train_model(x, y, epochs=10, learning_rate=0.0001)

# Plot the line of best fit
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, m * x + c, color="red", label="Best fit line")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.legend()
plt.show()

# Predict office price when size is 100 sq. ft.
size = 100
predicted_price = m * size + c
print(f"Predicted office price for 100 sq. ft.: {predicted_price:.2f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X_np, y_np, alpha, color):
    theta = np.array([[1.0],[2.0]], dtype=np.float64)
    iterations = 100
    cost_function = np.zeros((iterations, 1))
    # Gradient Descent
    for i in range(iterations):
        # Calculate predictions
        predictions = np.dot(X_np, theta)
        # Calculate errors
        errors = predictions - y_np
        # Update parameters
        theta -= alpha * (1 / len(y_np)) * np.dot(X_np.T, errors)
        cost_function[i, 0] = (1/ 2*(len(y_np)))* (np.sum(errors))**2
    # Generate predictions using the learned parameters
    predictions = np.dot(X_np, theta)
    plt.figure(1)
    # Plot the regression line
    plt.plot(x, predictions, label=f'Regression Line (alpha={alpha})', color= color)
    return iterations, cost_function

# Read your CSV file
csv_path = 'D:/Work and Study/HK2_nam3/AI/ex1.csv'
data_cs = pd.read_csv(csv_path)

# Extract features ('x') and target variable ('y')
X = data_cs[['x']]
y = data_cs['y']

# Add a column of ones to X to account for the intercept term
X['intercept'] = 1

# Convert features and target variable to NumPy arrays
X_np = np.array(X, dtype=np.float64)
x = np.array(data_cs['x'], dtype=np.float64)
y_n = np.array(y, dtype=np.float64)
y_np = np.reshape(y_n, (100, 1))

# Hyperparameters
alpha = [0.0001,0.0002, 0.0003, 0.0004, 0.00051]

#print('cost_function')
#print(cost_function.shape)
#print('x_cost')
#print(x_cost.shape)

#Draw each of the cost function with different learning rates
for learning_rate in alpha:
    iter,J = gradient_descent(X_np, y_np, learning_rate, color=np.random.rand(3,))
    #Figure 2
    plt.figure(2)
    #cost_fuction
    plt.plot(range(iter), J,label=f'alpha={learning_rate}' )

#Figure 1
plt.figure(1)
plt.scatter(x, y_np, label='Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Linear Regression')
plt.legend()

#Figure 2
plt.figure(2)
plt.xlabel('LOOP')
plt.ylabel('COST')
plt.title('Cost_Fuction')
plt.legend()
plt.show()

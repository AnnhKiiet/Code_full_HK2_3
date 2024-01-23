import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def cost_function(x,y,theta0,theta1):
    m = len(y)
    #Predictions h_theta function
    predictions = theta0 + theta1*x
    #Caculate cost function J
    J = (1/(2*m))*np.sum((predictions-y)**2)
    return J

def gradient_descent(x,y,theta0,theta1,alpha,num):
    #Number data in file csv
    m = len(y)
    #Save value of cost function after per loop
    J_history = []
    for ite in range(num):
        predictions = theta0 + theta1*x
        errors = predictions - y
        #Partial derivatives with theta0 and theta1
        gradient_theta0 = (1/m)*np.sum(errors)
        gradient_theta1 = (1/m)*np.sum(errors*x)
        #Update theta0 and theta1
        theta0 = theta0 - alpha*gradient_theta0
        theta1 = theta1 - alpha*gradient_theta1
        
        J = cost_function(x,y,theta0,theta1)
        J_history.append(J)
    return theta0,theta1,J_history

#Read data from data.csv
data = pd.read_csv('D:/Work and Study/HK2_nam3/AI/ex1.csv')

#Take x and y from file
x = data['x'].values
y = data['y'].values

#Learning rate
learning_rate = [0.0005,0.001, 0.003, 0.005, 0.007]

#Number of loops
num = 1000

#Graph the change of cost function J with learning rate values
for alpha in learning_rate:
    theta0 = 0
    theta1 = 0
    _,_,J_histoty = gradient_descent(x,y,theta0,theta1,alpha,num)
    plt.plot(range(num),J_histoty,label=f'alpha={alpha}')
plt.xlabel("LOOP")
plt.ylabel("COST FUNCTION J")
plt.title("the change of cost function J with learning rate values")
plt.legend()
plt.show()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def CIR(r0, kappa, theta, sigma, T, N, seed=42):
    np.random.seed(seed)
    dt = T/N
    t = np.linspace(0, T, N+1)
    r = np.zeros(N+1)
    r[0] = r0

    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))
        r[i] = r[i-1] + kappa*(theta - r[i-1])*dt + sigma*np.sqrt(r[i-1])*dW

    return t, r

# Read data from CSV
data = pd.read_csv('selicdados2.csv')

# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Example usage
r0 = train_data['SelicDia'].iloc[0]  # Initial interest rate
kappa = 0.1  # Mean reversion speed
theta = train_data['SelicDia'].mean()  # Long-term mean interest rate
sigma = train_data['SelicDia'].std()  # Volatility
T = 1376  # Time horizon
N = len(train_data) - 1  # Number of time steps

t_train, r_train = CIR(r0, kappa, theta, sigma, T, N)

# Predict interest rate for test data
r0_test = test_data['SelicDia'].iloc[0]  # Initial interest rate for test data
N_test = len(test_data) - 1  # Number of time steps for test data
t_test, r_test = CIR(r0_test, kappa, theta, sigma, T, N_test)



# Plotting the interest rate paths
plt.plot(t_train, r_train, label='Real')
plt.plot(t_test, r_test, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.title('CIR Model - Real vs Predicted')
plt.legend()
plt.show()
plt.savefig("CIR#_output.pdf")

rmse = np.sqrt(mean_squared_error(test_data['SelicDia'], r_test)) 
mae = mean_absolute_error(test_data['SelicDia'], r_test) 
mse = mean_squared_error(test_data['SelicDia'], r_test)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

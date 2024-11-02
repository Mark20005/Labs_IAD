import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Generate data
np.random.seed(12)

n = 100  # number of observations
alpha_real = 0.4
beta_real = 0.6
A_real = 1.2

K = np.random.uniform(1, 100, n)
L = np.random.uniform(1, 100, n)

# Generate Y without random error
Y_true = A_real * (K ** alpha_real) * (L ** beta_real)

# Generate normally distributed random error
error = st.norm.rvs(scale=0.05 * Y_true)  # standard deviation is 5%
Y = Y_true + error  # add error to Y

# Step 3: Log transformation and check rank
log_Y = np.log(Y)
log_K = np.log(K)
log_L = np.log(L)

# Build matrix H
H = np.column_stack((np.ones(n), log_K, log_L))

# Check the rank of matrix H
rank_H = np.linalg.matrix_rank(H)
if rank_H < H.shape[1]:
    print("Matrix H is rank-deficient, regenerate K and L")
else:
    print("Matrix H is full rank")

# Step 4: Estimate parameters using OLS
model = sm.OLS(log_Y, H)
results = model.fit()
log_Y_pred = results.predict(H)
# Get estimated parameters
A_estimated = np.exp(results.params[0])
alpha_estimated = results.params[1]
beta_estimated = results.params[2]

print(f"Estimated parameter A: {A_estimated}")
print(f"Estimated parameter α: {alpha_estimated}")
print(f"Estimated parameter β: {beta_estimated}")

# Step 5: Check coefficient significance
print(results.summary())

# Step 6: R-squared coefficient of determination
R_squared = results.rsquared
print(f"R-squared: {R_squared}")

# Step 7: Inverse transformation by exponentiation
Y_estimated = A_estimated * (K ** alpha_estimated) * (L ** beta_estimated)

# Display results
df = pd.DataFrame({
    'K': K,
    'L': L,
    'Y_true': Y_true,
    'Y_estimated': Y_estimated,
    'Y_with_error': Y
})

print(df.head())

# Step 4: Fit linear regression using sklearn for drawing the regression line
x = log_K.reshape(-1, 1)  # Feature variable
y = log_Y  # Target variable

# Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(x, y)

# Generate predictions for the regression line
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred = linear_model.predict(x_range)

# Step 5: Plot scatter of log(Y) vs log(K) and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(log_K, log_Y, color='blue', label='Data points', alpha=0.7)
plt.plot(x_range, y_pred, color='red', label='Regression Line')
plt.xlabel('log(K)')
plt.ylabel('log(Y)')
plt.title('Log-Log Scatter plot with Fitted Regression Line')
plt.legend()
plt.grid(True)
plt.show()

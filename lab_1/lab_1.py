import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(10)
N = 100
X_prob = np.random.rand(N)
Y_prob = np.random.rand(N)

X_det_up = np.linspace(0, 10, N)
Y_det_up = np.linspace(0, 10, N)


X_det_down = np.linspace(10, 0, N)
Y_det_down = np.linspace(10, 0, N)

X_stoch_up = X_prob + X_det_up
X_stoch_down = X_prob + X_det_down
Y_stoch_up = Y_prob + Y_det_up
Y_stoch_down = Y_prob + Y_det_down


def compute_statistics(X, Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    M_XY = np.mean((X - X_mean) * (Y - Y_mean))
    D_X = np.mean((X - X_mean) ** 2)
    D_Y = np.mean((Y - Y_mean) ** 2)
    return X_mean, Y_mean, M_XY, D_X, D_Y


X_mean_prob, Y_mean_prob, M_XY_prob, D_X_prob, D_Y_prob = compute_statistics(X_prob, Y_prob)
X_mean_stoch_up, Y_mean_stoch_up, M_XY_stoch_up, D_X_stoch_up, D_Y_stoch_up = compute_statistics(X_stoch_up, Y_stoch_up)
X_mean_stoch_down, Y_mean_stoch_down, M_XY_stoch_down, D_X_stoch_down, D_Y_stoch_down = compute_statistics(X_stoch_down, Y_stoch_down)


def compute_correlation(M_XY, D_X, D_Y):
    return M_XY / (np.sqrt(D_X * D_Y))


R_XY_prob = compute_correlation(M_XY_prob, D_X_prob, D_Y_prob)
R_XY_stoch_up = compute_correlation(M_XY_stoch_up, D_X_stoch_up, D_Y_stoch_up)
R_XY_stoch_down = compute_correlation(M_XY_stoch_down, D_X_stoch_down, D_Y_stoch_down)

print(f"Correlation between X_prob, Y_prob: : {R_XY_prob}")
print(f"Correlation between X_stoch_up, Y_stoch_up: {R_XY_stoch_up}")
print(f"Correlation between X_stoch_down, Y_stoch_down: {R_XY_stoch_down}")

to_dict = {'X_prob': X_prob, 'Y_prob': Y_prob,
           'X_stoch_up': X_stoch_up, 'Y_stoch_up': Y_stoch_up,
           'X_stoch_down': X_stoch_down, 'Y_stoch_down': Y_stoch_down}
print('---------------------------------------------------------------------------')
df_check = pd.DataFrame(to_dict)
print('Correlation between X_prob, Y_prob using numpy method: ', df_check['X_prob'].corr(df_check['Y_prob']))
print('Correlation between X_stoch_up, Y_stoch_up using numpy method: ',
      df_check['X_stoch_up'].corr(df_check['Y_stoch_up']))
print('Correlation between X_stoch_down, Y_stoch_down using numpy method: ',
      df_check['X_stoch_down'].corr(df_check['Y_stoch_down']))

# First scatter plot
plt.figure(figsize=(8, 6))
plt_1 = sns.scatterplot(x='X_prob', y='Y_prob', data=df_check, color='purple')
plt_1.set_title('Correlation between X_prob, Y_prob')
plt.show()

# Second scatter plot
plt.figure(figsize=(8, 6))
plt_2 = sns.scatterplot(x='X_stoch_up', y='Y_stoch_up', data=df_check, color='green', marker='+')
plt_2.set_title('Correlation between X_stoch_up, Y_stoch_up')
plt.show()

# Third scatter plot
plt.figure(figsize=(8, 6))
plt_3 = sns.scatterplot(x='X_stoch_down', y='Y_stoch_down', data=df_check, color='red', marker='+')
plt_3.set_title('Correlation between X_stoch_down, Y_stoch_down')
plt.show()

# Combined Heatmap and Pairplot in a single figure
plt.figure(figsize=(16, 12))
# Set the context for plots
sns.set_context('talk')

# Heatmap
heat_1 = sns.heatmap(df_check.corr(), annot=True)
plt.title('Heatmap of Correlations')

plt.show()  # Show the heatmap first

# Separate figure for pairplot
pair_1 = sns.pairplot(data=df_check)
plt.show()

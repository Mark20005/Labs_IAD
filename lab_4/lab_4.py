import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Генерація випадкових даних
np.random.seed(42)

# Задаємо кількість точок і генеруємо координати для трьох наборів
num_points = 5
set1 = np.random.normal(loc=[10, 10], scale=2, size=(num_points, 2)).astype(int)
set2 = np.random.normal(loc=[30, 10], scale=2, size=(num_points, 2)).astype(int)
set3 = np.random.normal(loc=[20, 30], scale=2, size=(num_points, 2)).astype(int)

# Об'єднуємо всі точки в один набір даних
data = np.vstack([set1, set2, set3])
data_frame = pd.DataFrame(data, columns=["1-а властивість", "2-а властивість"])
data_frame.index += 1  # Щоб індексація починалася з 1

# Формування Таблиці 4.2
print("Таблиця 4.2. Набір даних:")
data_frame.to_excel('excel_files/table_4_2.xlsx', engine='openpyxl')
print(data_frame)

# Нормалізація даних за допомогою мінімаксного відхилення
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
normalized_frame = pd.DataFrame(normalized_data, columns=["1-а властивість", "2-а властивість"])
normalized_frame.index += 1  # Щоб індексація починалася з 1

# Формування Таблиці 4.3
print("\nТаблиця 4.3. Нормалізований за мінімаксним відхиленням набір даних:")
normalized_frame.to_excel('excel_files/table_4_3.xlsx', engine='openpyxl')
print(normalized_frame)

# Ініціалізація кластерів та k-means алгоритм
from sklearn.cluster import KMeans

k = 3  # Задаємо кількість кластерів
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_
data_frame["№ кластеру"] = labels + 1  # Додаємо колонку з номером кластера

# Формування Таблиці 4.5
print("\nТаблиця 4.5. Розподіл об’єктів по кластерах:")
data_frame.to_excel('excel_files/table_4_5.xlsx', engine='openpyxl')
print(data_frame)

# Ітеративний процес кластеризації для формування Таблиці 4.4
iterations = []
distances = []

for i in range(1, 7):  # Ітерації (приклад з 6 кроками)
    kmeans = KMeans(n_clusters=k, n_init=1, max_iter=i, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    max_distance = np.max(np.min(kmeans.transform(data), axis=1))  # Максимальна відстань у кластері
    iterations.append([*labels, max_distance])
    distances.append(max_distance)

# Формування Таблиці 4.4
iterations_frame = pd.DataFrame(iterations, columns=["об’єкт 1", "об’єкт 2", "об’єкт 3", "об’єкт 4", "об’єкт 5",
                                                     "об’єкт 6", "об’єкт 7", "об’єкт 8", "об’єкт 9", "об’єкт 10",
                                                     "об’єкт 11", "об’єкт 12", "об’єкт 13", "об’єкт 14", "об’єкт 15",
                                                     "Максимальна відстань"])
iterations_frame.index += 1  # Щоб ітерації починалися з 1

print("\nТаблиця 4.4. Розподіл об’єктів по кластерах на кожному кроці:")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
iterations_frame.to_excel('excel_files/table_4_4.xlsx', engine='openpyxl')

print(iterations_frame)

# Візуалізація результатів
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')
plt.title("Розподіл об’єктів по кластерах")
plt.xlabel("1-а властивість")
plt.ylabel("2-а властивість")
plt.show()

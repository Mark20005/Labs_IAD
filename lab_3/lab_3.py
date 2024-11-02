import numpy as np
import pandas as pd


data = {
    'Restaurant': ['R1', 'R2', 'R3', 'R4'],
    'Number_of_Seats': [50, 80, 30, 70],
    'Average_Bill_1000UAH': [15, 25, 10, 30],
    'Cuisine_Type': ['Italian', 'Japanese', 'Fast Food', 'Mexican']
}

df = pd.DataFrame(data)

new_restaurant = {'Number_of_Seats': 65, 'Average_Bill_1000UAH': 18}

features = df[['Number_of_Seats', 'Average_Bill_1000UAH']].values
new_data_features = np.array([new_restaurant['Number_of_Seats'], new_restaurant['Average_Bill_1000UAH']])

distances = np.sqrt(np.sum((features - new_data_features) ** 2, axis=1))

df['Distance'] = distances

nearest_neighbor = df.loc[df['Distance'].idxmin()]

predicted_cuisine = nearest_neighbor['Cuisine_Type']

new_row = {
    'Restaurant': 'R5',
    'Number_of_Seats': new_restaurant['Number_of_Seats'],
    'Average_Bill_1000UAH': new_restaurant['Average_Bill_1000UAH'],
    'Cuisine_Type': predicted_cuisine,
    'Distance': np.nan  # Відстань до самого себе
}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Виводимо оновлену таблицю
print("Оновлена таблиця з відстанями для кожного ресторану:")
print(df)
df.to_excel("output.xlsx")
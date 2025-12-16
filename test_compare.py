import pandas as pd
from LinearRegressionModel import LinearRegressionModel
from ModelComparator import ModelComparator

def polynomial_feature(col, degree):
    return (f"{col}^{degree}", lambda df: df[col] ** degree)

def multiplication_feature(col1, col2):
    return (f"{col1}_{col2}", lambda df: df[col1] * df[col2])

def addition_feature(col1, col2):
    return (f"{col1}_{col2}", lambda df: df[col1] + df[col2])

df = pd.read_excel('dataset.xlsx')
train_range = range(192, 216)
test_range = range(216, 240)  # следующие 24 наблюдения

# Создаём несколько моделей
model1 = LinearRegressionModel(df, target_col='Ypowerconsumption', name='M999')
model1.fit(train_range, ['X0 Фиктивная переменная', 'hour', 'temp'])

model2 = LinearRegressionModel(df, target_col='Ypowerconsumption', name='M9999')
model2.fit(train_range, [
    'X0 Фиктивная переменная',
    'hour',
    polynomial_feature('hour', 2)
])

# Прогноз для обеих
model1.forecast(test_range)
model2.forecast(test_range)
model1.plot_forecast()

# Сравнение
comparator = ModelComparator([model1, model2])
print("Сравнение на обучении:")
print(comparator.compare_metrics(on='train'))

print("\nСравнение на тесте:")
print(comparator.compare_metrics(on='test'))
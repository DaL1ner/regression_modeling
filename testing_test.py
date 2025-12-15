import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import metrics
from numpy import corrcoef
from scipy.stats import f
from scipy.stats import t
from scipy.stats import pearsonr
from statsmodels.regression import linear_model
from LinearRegressionModel import LinearRegressionModel
from ModelComparator import ModelComparator

def polynomial_feature(col, degree):
    return (f"{col}^{degree}", lambda df: df[col] ** degree)

def multiplication_feature(col1, col2):
    return (f"{col1}_{col2}", lambda df: df[col1] * df[col2])

def addition_feature(col1, col2):
    return (f"{col1}_{col2}", lambda df: df[col1] + df[col2])


# 1. Загружаем данные ОДИН РАЗ
df = pd.read_excel('dataset.xlsx')

# 2. Создаём модель
model = LinearRegressionModel(df, target_col='Ypowerconsumption', name='M1')

# 3. Обучаем на нужных данных и признаках
train_range = range(192, 216)
features = ['X0 Фиктивная переменная', 'hour', 'temp', 'cloud', 'wet', 'winddir', 'windspeed', 'daylength', 'day/night']
model.fit(train_range, features)

# 4. Проверяем адекватность и значимость
model.validate()
model.test_coefficients()

# 5. Строим графики для обучающей выборки
model.plot_training(show_ci=True)

# 6. Прогнозируем на следующие 24 часа
test_range = range(216, 240)  # следующие 24 наблюдения
model.forecast(test_range)

# 7. Строим график прогноза
model.plot_forecast()

# 8. Смотрим метрики
print("Метрики на обучении:", model.metrics_train)
print("Метрики на тесте:", model.metrics_test)
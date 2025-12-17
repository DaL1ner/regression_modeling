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
# train_range = range(192, 216) # первые 24 наблюдения
# test_range = range(216, 240)  # следующие 24 наблюдения

train_range = range(192, 240) # первые 48 наблюдений
test_range = range(240, 264)  # следующие 24 наблюдения


# Модели
model1 = LinearRegressionModel(df, target_col='Ypowerconsumption', name='M1')
model1.fit(train_range, ['X0 Фиктивная переменная', 'hour', 'temp', 'cloud', 'wet', 'winddir', 'windspeed', 'daylength', 'day/night'])

model2 = LinearRegressionModel(df, target_col='Ypowerconsumption', name='M2')
model2.fit(train_range, ['X0 Фиктивная переменная', 'hour', 'temp'])

model3 = LinearRegressionModel(df, target_col='Ypowerconsumption', name='M3')
model3.fit(train_range, ['X0 Фиктивная переменная', 'hour', 'temp', polynomial_feature('temp', 2)])

model4 = LinearRegressionModel(df, target_col='Ypowerconsumption', name='M4')
model4.fit(train_range, ['X0 Фиктивная переменная', 'hour', 'temp', polynomial_feature('hour', 2)])

model5 = LinearRegressionModel(df, target_col='Ypowerconsumption', name='M5')
model5.fit(train_range, ['X0 Фиктивная переменная', 'hour', 'temp', multiplication_feature('hour', 'temp')])

# model5.validate()
# model5.test_coefficients()

# Предсказание
model1.forecast(test_range)
model2.forecast(test_range)
model3.forecast(test_range)
model4.forecast(test_range)
model5.forecast(test_range)


# Сравнение
comparator = ModelComparator([model1, model2, model3, model4, model5])
print("Сравнение на обучении:")
print(comparator.compare_metrics('all', on='train'))
# print("\nСравнение на тесте:")
# print(comparator.compare_metrics(on='test'))

# Выводим общий график
comparator.plot_all_models([model1, model2, model3, model4, model5], plot_type='forecast')
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

###################
## МОДЕЛИРОВАНИЕ ##
###################

### (ВЫБОРКА МОДЕЛИ) ###

# границы выборки
N1 = 192
N2 = 215
N = N2 - N1 + 1

# чтение данных
df = pd.read_excel('dataset.xlsx')
df_exp = df.iloc[N1:N2+1, :] # выборка

#print(df_exp)

# переменные
dates = df_exp.iloc[:, 1] # дата наблюдения
X0 = df_exp.iloc[:, 4] # фиктивная переменная
X1 = df_exp.iloc[:, 5] # час
X2 = df_exp.iloc[:, 6] # температура
X3 = df_exp.iloc[:, 7] # облачность
X4 = df_exp.iloc[:, 8] # влажность
X5 = df_exp.iloc[:, 9] # направление ветра
X6 = df_exp.iloc[:, 10] # сила ветра
X7 = df_exp.iloc[:, 11] # длительность дня
X8 = df_exp.iloc[:, 12] # день/ночь
X9 = df_exp.iloc[:, 13] # рабочий/нерабочий день
Y = df_exp.iloc[:, 14] # зависимая переменная потребления электроэнергии

# матрицы X и Y
X_t = np.vstack((X0, X1, X2, X3, X4, X5, X6, X7, X8))
K = 9
Y = np.vstack(Y)
X = np.transpose(X_t)
# print(X)
# print(Y)

# поиск коэффициентов
B = np.linalg.inv(X.T @ X) @ X.T @ Y
# print(B)

# Предсказанные значения
YR = X @ B

### (ОЦЕНКА АДЕКВАТНОСТИ) ###

# Дисперсия адекватности (ошибки модели)
Dad = np.sum((Y - YR) ** 2) / (N - K)

# Среднее Y
YSR = np.mean(Y)

# Дисперсия Y (без учёта модели)
DY = np.sum((Y - YSR) ** 2) / (N - 1)

# Расчётный F-критерий
FR = DY / Dad

# Табличное значение F (для α=0.05)
F = f.ppf(1 - 0.05, N - 1, N - K)

# Вывод
print(f"Модель M1: N = {N}, K = {K}")
print("\n=== Оценка адекватности модели ===")
print(f"Расчётное значение F-критерия Фишшера = {round(FR, 4)}")
print(f"Табличное значение F-критерия Фишшера F(α=0.05) = {round(F, 4)}")
if FR > F:
    print("=> Модель адекватна по критерию Фишшера")
else:
    print("=> Модель неадекватна по критерию Фишшера")


### (ОЦЕНКА КАЧЕСТВА) ###

r, p_val = pearsonr(Y.flatten(), YR.flatten()) # корреляция между Y и YR
r2 = corrcoef(Y.flatten(), YR.flatten())[0, 1] # считаем тоже самое но другой функцией
R_squared = (1 - Dad/DY)
R_squared_adjusted = 1 - ((1 - R_squared) * (N - 1) / (N - K - 1))
MSE = np.sum((Y - YR) ** 2) / N
RMSE = math.sqrt(MSE)
MAE = abs(Y - YR).mean()
MAE2 = metrics.mean_absolute_error(Y, YR) # считаем тоже самое но другой функцией
RelativeError = MAE / Y.mean() * 100
MAPE = (abs(Y - YR) / Y).mean() * 100
WAPE = (abs(Y - YR).sum() / Y.sum()) * 100
RSS = np.sum((Y - YR) ** 2)
max_log_likelihood = -N/2 * np.log(2 * np.pi * MSE) - N/2
AIC = N * np.log(RSS) + 2 * K
AICc = AIC + (2 * K * (K + 1)) / (N - K - 1)
BIC = -2 * max_log_likelihood + K * np.log(N)
Cp = RSS / (MSE - (N - 2*K))
# akaikeWeught
# добавить дополнительные оценки

# точность модели
print(f"\n=== Качество модели внутри выборки ===")
print(f"Коэффициент корреляции между Y и YR = {round(r, 4)}")
# print(f"Коэффициент корреляции 2 между Y и YR = {round(r2, 4)}")
print(f"Коэффициент детерминации R² = {round(R_squared, 4)}")
print(f"Скорректированный коэффициент детерминации R² adjusted = {round(R_squared_adjusted, 4)}")
print(f"Среднеквадратичная ошибка MSE = {round(MSE, 4)}, кВТ*ч")
print(f"Корень среднеквадратичной ошибки RMSE = {round(RMSE, 4)}, кВТ*ч")
print(f"Абсолютная ошибка MAE = {round(MAE, 4)}, кВТ*ч")
# print(f"Абсолютная ошибка 2 MAE = {round(MAE2, 4)}, кВТ*ч")
print(f"Относительная ошибка RE = {round(RelativeError, 4)}%")
print(f"Cредняя абсолютная процентная ошибка MAPE = {round(MAPE, 4)}%")
print(f"Взвешенная абсолютная процентная ошибка WAPE = {round(WAPE, 4)}%")
print(f"Mallow's Cp = {round(Cp, 4)}")
# объём потерянной информации - для сравнения моделей
print(f"Информационный критерий Акаике AIC через RSS = {round(AIC, 4)}")
print(f"Cкорректированный информационный критерий Акаике AICc через RSS = {round(AICc, 4)}") # для малой выборки
print(f"Критерий Байеса BIC = {round(BIC, 4)}")



### (ПРОВЕРКА ЗНАЧИМОСТИ КОЭФИЦИЕНТОВ) ###

# матрица нормальных уравнений
G = np.linalg.inv(X.T @ X)

# табличное значение t-критерия Стьюдента
t = t.ppf(1 - 0.05/2, N-K)

# Доверительные интервалы (полуширина) коэффициентов регрессии
delta = t * np.sqrt(Dad * np.diag(G))

print("\n=== Проверка значимости коэфициентов ===")
for i in range(K):
    significant = abs(B[i, 0]) > delta[i]
    print(f"Коэффициент регрессии β{i} = {B[i,0]}, Δ{i} = {delta[i]} => {'ЗНАЧИМ' if significant else 'НЕЗНАЧИМ'}")



### (Доверительные интервалы для предсказаний) ###
G = np.linalg.inv(X.T @ X) # уже посчитана

# Вычисляем x_i^T G x_i для каждой строки
SE_mean = np.array([np.sqrt(Dad * x @ G @ x.T) for x in X])

# Границы доверительного коридора
YR_lower = YR.flatten() - t * SE_mean
YR_upper = YR.flatten() + t * SE_mean



### (ВИЗУАЛИЗАЦИЯ) ###

# добавить обозначение разницы верхней и нижней границы доверительного коридора 3сигма (фото в репозитории)

# Визуализация сравнения реальных значений с предстаказанными
plt.figure(figsize=(14, 6))

# График 1: Реальные и предсказанные значения
plt.subplot(1, 2, 1)
plt.plot(dates, Y, 'o-', label='Реальное значение, Y', color='blue')
plt.plot(dates, YR, 's--', label='Предсказанное значение, YR', color='red')
plt.fill_between(dates, YR_lower, YR_upper, color='red', alpha=0.2, label='Доверительный интервал')

i_annot = 5
x_annot = dates.iloc[i_annot]
y_upper = YR_upper[i_annot]
y_lower = YR_lower[i_annot]
offset_x = pd.Timedelta(hours=1)

# Стрелка вверх (от верхней границы)
plt.annotate('',
             xy=(x_annot, y_upper),
             xytext=(x_annot, y_upper + 15),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5, mutation_scale=20))

# Стрелка вниз (от нижней границы)
plt.annotate('',
             xy=(x_annot, y_lower),
             xytext=(x_annot, y_lower - 15),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5, mutation_scale=20))

# Подпись "3σ"
plt.text(x_annot - offset_x, y_upper + 17, '3σ',
         fontsize=12, color='red', va='center', ha='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))

plt.title('Реальные и предсказанные значения на исходной выборке\n(Модель M1 | K=9, N=24)')
plt.xlabel('Дата и время наблюдения')
plt.ylabel('Потребление электроэнергии, кВт*ч')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)


# График 2: Y и YR
plt.subplot(1, 2, 2)
plt.scatter(Y, YR, color='purple', alpha=0.7)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=1)  # диагональ y=x
plt.title(f'Сравнение Y и YR на исходной выборке\nr = {round(r, 4)}')
plt.xlabel('Реальное Y')
plt.ylabel('Предсказанное YR')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()



#############
## ПРОГНОЗ ##
#############



### Прогноз на новые данные (следующие 24 часа) ###
N_test = 24
df_test = df.iloc[N2+1 : N2+1+N_test, :]

# Даты для прогноза
dates_test = pd.to_datetime(df_test.iloc[:, 1])
# Собираем признаки (X0–X8) для новых данных
X0_t = df_test.iloc[:, 4]
X1_t = df_test.iloc[:, 5]
X2_t = df_test.iloc[:, 6]
X3_t = df_test.iloc[:, 7]
X4_t = df_test.iloc[:, 8]
X5_t = df_test.iloc[:, 9]
X6_t = df_test.iloc[:, 10]
X7_t = df_test.iloc[:, 11]
X8_t = df_test.iloc[:, 12]
X9_t = df_test.iloc[:, 13]
Y_test = df_test.iloc[:, 14]

X_test_t = np.vstack((X0_t, X1_t, X2_t, X3_t, X4_t, X5_t, X6_t, X7_t, X8_t))
X_test = X_test_t.T

# Предсказание на новых данных
YR_test = X_test @ B


# Оценка качества прогноза
r_test, _ = pearsonr(Y_test, YR_test.flatten())
R_squared_test = (1 - Dad/DY)
R_squared_adjusted_test = 1 - ((1 - R_squared_test) * (N - 1) / (N - K - 1))
MSE_test = np.sum((Y_test - YR_test) ** 2) / N
RMSE_test = math.sqrt(MSE_test)
MAE_test = abs(Y_test - YR_test).mean()
MAE2_test = metrics.mean_absolute_error(Y_test, YR_test) # считаем тоже самое но другой функцией
RelativeError_test = MAE_test / Y_test.mean() * 100
MAPE_test = (abs(Y_test - YR_test) / Y_test).mean() * 100
WAPE_test = (abs(Y_test - YR_test).sum() / Y_test.sum()) * 100
RSS_test = np.sum((Y_test - YR_test) ** 2)
max_log_likelihood_test = -N/2 * np.log(2 * np.pi * MSE_test) - N/2
AIC_test = N * np.log(RSS_test) + 2 * K
AICc_test = AIC_test + (2 * K * (K + 1)) / (N - K - 1)
BIC_test = -2 * max_log_likelihood_test + K * np.log(N)
Cp_test = RSS_test / (MSE_test - (N - 2*K))

# точность модели
print(f"\n=== Качество модели на новых данных ===")
print(f"Коэффициент корреляции между Y и YR = {round(r_test, 4)}")
print(f"Коэффициент детерминации R² = {round(R_squared_test, 4)}")
print(f"Скорректированный коэффициент детерминации R² adjusted = {round(R_squared_adjusted_test, 4)}")
print(f"Среднеквадратичная ошибка MSE = {round(MSE_test, 4)}, кВТ*ч")
print(f"Корень среднеквадратичной ошибки RMSE = {round(RMSE_test, 4)}, кВТ*ч")
print(f"Абсолютная ошибка MAE = {round(MAE_test, 4)}, кВТ*ч")
# print(f"Абсолютная ошибка 2 MAE = {round(MAE2_test, 4)}, кВТ*ч")
print(f"Относительная ошибка RE = {round(RelativeError_test, 4)}%")
print(f"Cредняя абсолютная процентная ошибка MAPE = {round(MAPE_test, 4)}%")
print(f"Взвешенная абсолютная процентная ошибка WAPE = {round(WAPE_test, 4)}%")
print(f"Mallow's Cp = {round(Cp_test, 4)}")
# объём потерянной информации - для сравнения моделей
print(f"Информационный критерий Акаике AIC через RSS = {round(AIC_test, 4)}")
print(f"Cкорректированный информационный критерий Акаике AICc через RSS = {round(AICc_test, 4)}") # для малой выборки
print(f"Критерий Байеса BIC = {round(BIC_test, 4)}")


### (ВИЗУАЛИЗАЦИЯ) ###


# Объединённый график: обучение + прогноз
plt.figure(figsize=(14, 6))

# Объединение данных обучения и теста
all_dates = pd.concat([dates, dates_test])
all_Y_real = np.concatenate([Y.flatten(), Y_test])
all_Y_pred = np.concatenate([YR.flatten(), YR_test.flatten()])

# Рисуем график
train_size = len(dates)
plt.plot(all_dates[:train_size], all_Y_real[:train_size], 'o-',
    color='steelblue', label='Реальное (обучение)', markersize=4)
plt.plot(all_dates[:train_size], all_Y_pred[:train_size], 's--',
    color='crimson', label='Прогноз (обучение)', markersize=4)
plt.plot(all_dates[train_size:], all_Y_real[train_size:], 'o-',
    color='gray', label='Реальное (24 ч вперёд)', markersize=4, alpha=0.7)
plt.plot(all_dates[train_size:], all_Y_pred[train_size:], 'd--',
         color='purple', label='Прогноз (24 ч вперёд)', markersize=5)

plt.title('Прогноз потребления электроэнергии на 24 часа вперёд')
plt.xlabel('Дата и время наблюдения')
plt.ylabel('Потребление, кВт*ч')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()



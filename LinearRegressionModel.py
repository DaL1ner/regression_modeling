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
from BaseModel import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self, df, target_col, name = None, alpha=0.05):
        super().__init__(df, target_col, name, alpha)
        self.features = []  # выбранные признаки (например, ['hour', 'day/night'])
        self.B = None # коэффициенты модели
        self.Y = None # зависимая переменная потребления электроэнергии обучающей выборки
        self.YR = None # предсказанные значения на обучающей выборке
        self.Y_test = None # зависимая переменная потребления электроэнергии тестовой выборки
        self.YR_test = None # предсказанные значения на тестовой выборке
        self.G = None  # матрица нормальных уравнений
        self.Dad = None  # дисперсия адекватности на обучающей выборке
        # self.Dad_test = None  # дисперсия адекватности на тестовой выборке
        self.t_crit = None  # критическое значение t
        self.N = None  # число наблюдений обучающей выборки
        self.N_test = None  # число наблюдений тестовой выборки (с учётом обучающей)
        self.K = None  # число параметров

    def _build_X_matrix(self, range, features):
        """Собирает матрицу X для указанных индексов и признаков"""
        subset = self.df.iloc[range]
        X_columns = []
        feature_names = []

        for feat in features:
            if isinstance(feat, str):
                name = feat
                values = subset[name].values
            elif isinstance(feat, tuple) and len(feat) == 2:
                name, func = feat
                values = func(subset).values
            else:
                raise ValueError(f"Неподдерживаемый формат признака: {feat}")

            X_columns.append(values)
            feature_names.append(name)

        X = np.column_stack(X_columns)
        dates = subset.iloc[:, 1]
        self.feature_names = feature_names

        return X, dates

    def calculate_metrics(self, Y, YR, prefix):
        """Считает все метрики качества"""

        if prefix == 'test':
            N = self.N_test
            YSR = np.mean(self.Y)
            DY = np.sum((Y - YSR) ** 2) / (N - 1)

            r, p_val = pearsonr(Y.flatten(), YR.flatten()) # корреляция между Y и YR
            R_squared = (1 - self.Dad/DY)
            R_squared_adjusted = 1 - ((1 - R_squared) * (N - 1) / (N - self.K - 1))
            MSE = np.sum((Y - YR) ** 2) / N
            RMSE = math.sqrt(MSE)
            MAE = abs(Y - YR).mean()
            RelativeError = MAE / Y.mean() * 100
            MAPE = (abs(Y - YR) / Y).mean() * 100
            WAPE = (abs(Y - YR).sum() / Y.sum()) * 100
            RSS = np.sum((Y - YR) ** 2)

            metrics_dict = {
                'R': round(r, 4), # Коэффициент корреляции
                'R²': round(R_squared, 4), # Коэффициент детерминации
                'R² adjusted': round(R_squared_adjusted, 4), # Скорректированный коэффициент детерминации
                'MSE': round(MSE, 4), # Среднеквадратичная ошибка
                'RMSE': round(RMSE, 4), # Корень среднеквадратичной ошибки
                'MAE': round(MAE, 4), # Абсолютная ошибка
                'RE': round(RelativeError, 4), # Относительная ошибка
                'MAPE': round(MAPE, 4), # Cредняя абсолютная процентная ошибка
                'WAPE': round(WAPE, 4), # Взвешенная абсолютная процентная ошибка
                'RSS': round(RSS, 4),
            }
        elif prefix == 'train' and self.Y is not None:
            N = self.N
            YSR = np.mean(Y)
            DY = np.sum((Y - YSR) ** 2) / (N - 1)

            r, p_val = pearsonr(Y.flatten(), YR.flatten())
            R_squared = (1 - self.Dad/DY)
            R_squared_adjusted = 1 - ((1 - R_squared) * (N - 1) / (N - self.K - 1))
            MSE = np.sum((Y - YR) ** 2) / N
            RMSE = math.sqrt(MSE)
            MAE = abs(Y - YR).mean()
            RelativeError = MAE / Y.mean() * 100
            MAPE = (abs(Y - YR) / Y).mean() * 100
            WAPE = (abs(Y - YR).sum() / Y.sum()) * 100
            RSS = np.sum((Y - YR) ** 2)
            max_log_likelihood = -N/2 * np.log(2 * np.pi * MSE) - N/2
            AIC = N * np.log(RSS) + 2 * self.K
            AICc = AIC + (2 * self.K * (self.K + 1)) / (N - self.K - 1)
            BIC = -2 * max_log_likelihood + self.K * np.log(N)
            Cp = RSS / (MSE - (N - 2*self.K))

            metrics_dict = {
                'R': round(r, 4), # Коэффициент корреляции
                'R²': round(R_squared, 4), # Коэффициент детерминации
                'R² adjusted': round(R_squared_adjusted, 4), # Скорректированный коэффициент детерминации
                'MSE': round(MSE, 4), # Среднеквадратичная ошибка
                'RMSE': round(RMSE, 4), # Корень среднеквадратичной ошибки
                'MAE': round(MAE, 4), # Абсолютная ошибка
                'RE': round(RelativeError, 4), # Относительная ошибка
                'MAPE': round(MAPE, 4), # Cредняя абсолютная процентная ошибка
                'WAPE': round(WAPE, 4), # Взвешенная абсолютная процентная ошибка
                'RSS': round(RSS, 4), #
                'AIC': round(AIC, 4), # Информационный критерий Акаике
                'AICc': round(AICc, 4), # Cкорректированный информационный критерий Акаике
                'BIC': round(BIC, 4), # Критерий Байеса
                'Mallows Cp': round(Cp, 4) #
            }
        else:
            raise Exception("Неверный префикс")

        if prefix == 'train':
            self.metrics_train = metrics_dict
        elif prefix == 'test':
            self.metrics_test = metrics_dict

        return metrics_dict

    def predict(self, X):
        """Выдаёт матрицу-столбец Y с предсказанием"""
        if not self.is_fitted:
            raise Exception("Модель не обучена")
        return X @ self.B

    def fit(self, train_range, features):
        """Обучает модель на указанных индексах и признаках"""
        self.features = features
        X, self.dates_train = self._build_X_matrix(train_range, features)
        self.Y = self.df.iloc[train_range][self.target_col].values.reshape(-1, 1)
        self.Y = self.Y.reshape(-1)

        # МНК
        self.G = np.linalg.inv(X.T @ X)
        self.B = (self.G @ X.T @ self.Y).reshape(-1, 1)
        self.is_fitted = True

        # Предсказания на обучающей выборке
        self.YR = self.predict(X).flatten()

        # Дисперсия адекватности
        self.N = len(train_range)
        self.K = X.shape[1]
        self.Dad = np.sum((self.Y - self.YR) ** 2) / (self.N - self.K)

        # табличное значение t-критерия Стьюдента
        self.t = t.ppf(1 - self.alpha/2, self.N - self.K)

        # Доверительные интервалы для предсказаний
        self.SE_mean = np.array([np.sqrt(self.Dad * x @ self.G @ x.T) for x in X])
        self.YR_lower = self.YR.flatten() - self.t * self.SE_mean
        self.YR_upper = self.YR.flatten() + self.t * self.SE_mean

        # Расчёт метрик
        self.calculate_metrics(self.Y, self.YR, prefix='train')

        return self

    def forecast(self, test_range):
        """Даёт прогноз на тестовые данные"""
        if not self.is_fitted:
            raise Exception("Модель не обучена")

        # Собираем X_test
        X_test, self.dates_test = self._build_X_matrix(test_range, self.features)
        self.Y_test = self.df.iloc[test_range][self.target_col].values

        # Предсказания на тестовой выборке
        self.YR_test = self.predict(X_test).flatten()

        # G_test = np.linalg.inv(X_test.T @ X_test)

        # Дисперсия адекватности
        self.N_test = len(test_range)
        # self.Dad_test = np.sum((self.Y_test - self.YR_test) ** 2) / (self.N_test - self.K)

        # табличное значение t-критерия Стьюдента
        t_test = t.ppf(1 - self.alpha/2, self.N - self.K)

        # Доверительные интервалы для предсказаний
        self.SE_mean_test = np.array([np.sqrt(self.Dad * x @ self.G @ x.T) for x in X_test])
        self.YR_lower_test = self.YR_test.flatten() - t_test * self.SE_mean_test
        self.YR_upper_test = self.YR_test.flatten() + t_test * self.SE_mean_test

        # Расчёт метрик на тесте
        self.calculate_metrics(self.Y_test, self.YR_test, prefix='test')

        return self.YR_test

    def validate(self, show=True):
        """Проверка адекватности (F-тест)"""
        if not self.is_fitted:
            raise Exception("Модель не обучена")

        # Среднее Y
        YSR = np.mean(self.Y)

        # Дисперсия Y (без учёта модели)
        DY = np.sum((self.Y - YSR) ** 2) / (self.N - 1)

        # Расчётный F-критерий
        FR = DY / self.Dad

        # Табличное значение F (для α=0.05)
        F = f.ppf(1 - 0.05, self.N - 1, self.N - self.K)

        # Вывод
        if show:
            print(f"Модель {self.name}: N = {self.N}, K = {self.K}")
            print("\n=== Оценка адекватности модели ===")
            print(f"Расчётное значение F-критерия Фишшера = {round(FR, 4)}")
            print(f"Табличное значение F-критерия Фишшера F(α=0.05) = {round(F, 4)}")
            if FR > F:
                print("=> Модель адекватна по критерию Фишшера")
            else:
                print("=> Модель неадекватна по критерию Фишшера")

        return FR

    def test_coefficients(self):
        """Проверка значимости коэффициентов (t-тест)"""
        if not self.is_fitted:
            raise Exception("Модель не обучена")

        # Доверительные интервалы (полуширина) коэффициентов регрессии
        delta = self.t * np.sqrt(self.Dad * np.diag(self.G))

        print("\n=== Проверка значимости коэфициентов ===")
        for i in range(self.K):
            significant = abs(self.B[i, 0]) > delta[i]
            print(f"Коэффициент регрессии β{i} = {self.B[i,0]}, Δ{i} = {delta[i]} => {'ЗНАЧИМ' if significant else 'НЕЗНАЧИМ'}")



    def plot_training(self, show_ci=True):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.dates_train, self.Y, 'o-', label='Реальное значение, Y', color='blue')
        plt.plot(self.dates_train, self.YR, 's--', label='Предсказанное значение, YR', color='red')

        if show_ci:
            plt.fill_between(
                self.dates_train,
                self.YR_lower,
                self.YR_upper,
                color='red', alpha=0.2, label='Доверительный интервал'
            )

            i_annot = 5
            x_annot = self.dates_train.iloc[i_annot]
            y_upper = self.YR_upper[i_annot]
            y_lower = self.YR_lower[i_annot]
            offset_x = pd.Timedelta(hours=1)

            # Стрелки и подпись "3σ"
            plt.annotate('', xy=(x_annot, y_upper), xytext=(x_annot, y_upper + 15),
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5, mutation_scale=20))
            plt.annotate('', xy=(x_annot, y_lower), xytext=(x_annot, y_lower - 15),
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5, mutation_scale=20))
            plt.text(x_annot - offset_x, y_upper + 20, '3σ', fontsize=12, color='red',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))

        plt.title(f'Реальные и предсказанные значения\n(Модель {self.name} | K={self.K}, N={self.N})')
        plt.xlabel('Дата и время наблюдения')
        plt.ylabel('Потребление электроэнергии, кВт*ч')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)

        # Второй график: scatter Y vs YR
        plt.subplot(1, 2, 2)
        plt.scatter(self.Y, self.YR, color='purple', alpha=0.7)
        plt.plot([self.Y.min(), self.Y.max()],
                 [self.Y.min(), self.Y.max()], 'k--', lw=1)
        #plt.title(f'Сравнение Y и YR\nr = {self.metrics_train["correlation"]:.4f}')
        plt.xlabel('Реальное Y')
        plt.ylabel('Предсказанное YR')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_forecast(self, test_range=None):
        """Строит график с объединением обучающей и тестовой выборок"""
        if test_range is not None:
            self.forecast(test_range)  # если ещё не прогнозировали

        # Объединяем даты, реальные и предсказанные значения
        all_dates = pd.concat([self.dates_train, self.dates_test])
        all_Y = np.concatenate([self.Y.flatten(), self.Y_test])
        all_YR = np.concatenate([self.YR.flatten(), self.YR_test])
        all_YR_lower = np.concatenate([self.YR_lower, self.YR_lower_test])
        all_YR_upper = np.concatenate([self.YR_upper, self.YR_upper_test])

        plt.figure(figsize=(14, 6))
        train_size = len(self.dates_train)

        # Реальное значение (обучение)
        plt.plot(all_dates[:train_size], all_Y[:train_size], 'o-',
                 color='steelblue', label='Реальное (обучение)', markersize=4)

        # Предсказанное (обучение)
        plt.plot(all_dates[:train_size], all_YR[:train_size], 's--',
                 color='crimson', label='Прогноз (обучение)', markersize=4)

        # Реальное значение (тест)
        plt.plot(all_dates[train_size-1:], all_Y[train_size-1:], 'o-',
                 color='gray', label='Реальное (тест)', markersize=4, alpha=0.7)

        # Предсказанное (тест)
        plt.plot(all_dates[train_size:], all_YR[train_size:], 'd--',
                 color='purple', label='Прогноз (тест)', markersize=5)

        # Доверительные интервалы
        plt.fill_between(
            all_dates[:train_size],
            all_YR_lower[:train_size],
            all_YR_upper[:train_size],
            color='crimson', alpha=0.2, label='Доверительный интервал (обучение)'
        )
        plt.fill_between(
            all_dates[train_size:],
            all_YR_lower[train_size:],
            all_YR_upper[train_size:],
            color='purple', alpha=0.2, label='Доверительный интервал (тест)'
        )

        # Вертикальная пунктирная линия на стыке
        boundary_date = self.dates_train.iloc[-1]  # последняя дата обучения
        plt.axvline(x=boundary_date, color='black', linestyle='--', linewidth=1, label='Граница обучения')


        plt.title('Прогноз потребления электроэнергии')
        plt.xlabel('Дата и время наблюдения')
        plt.ylabel('Потребление, кВт*ч')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
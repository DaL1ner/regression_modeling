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
        self.coefficients = None
        self.G = None  # матрица (X^T X)^{-1}
        self.Dad = None  # дисперсия адекватности
        self.t_crit = None  # критическое значение t
        self.N = None  # число наблюдений
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
                # Новый формат: (имя, функция)
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

    def predict(self, X):
        """Предсказывает для новых данных"""
        if not self.is_fitted:
            raise Exception("Модель не обучена")
        return X @ self.coefficients

    def fit(self, train_range, features):
        self.features = features
        self.X_train, self.dates_train = self._build_X_matrix(train_range, features)
        self.Y = self.df.iloc[train_range][self.target_col].values.reshape(-1, 1)
        self.Y = self.Y.reshape(-1)

        # МНК: B = (X^T X)^{-1} X^T Y
        XtX = self.X_train.T @ self.X_train
        self.G = np.linalg.inv(XtX)
        self.coefficients = self.G @ self.X_train.T @ self.Y
        self.is_fitted = True

        # Предсказания на обучающей выборке
        self.YR = self.predict(self.X_train).flatten()

        # Дисперсия адекватности
        self.N = len(train_range)
        self.K = self.X_train.shape[1]
        self.Dad = np.sum((self.Y - self.YR) ** 2) / (self.N - self.K)

        # Критическое значение t
        self.t_crit = t.ppf(1 - self.alpha/2, self.N - self.K)

        # Доверительные интервалы для предсказаний
        self.SE_mean = np.array([np.sqrt(self.Dad * x @ self.G @ x.T) for x in self.X_train])
        self.YR_lower = self.YR.flatten() - self.t_crit * self.SE_mean
        self.YR_upper = self.YR.flatten() + self.t_crit * self.SE_mean

        # Расчёт метрик
        self.calculate_metrics(self.Y, self.YR, prefix='train')

        return self

    def forecast(self, test_range):
        """Даёт прогноз на тестовые данные и сохраняет результаты для визуализации"""
        if not self.is_fitted:
            raise Exception("Модель не обучена")

        # Собираем X_test
        X_test, self.dates_test = self._build_X_matrix(test_range, self.features)
        self.Y_test = self.df.iloc[test_range][self.target_col].values
        self.YR_test = self.predict(X_test).flatten()

        # Расчёт метрик на тесте
        self.calculate_metrics(self.Y_test, self.YR_test, prefix='test')

        return self.YR_test

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

        plt.figure(figsize=(14, 6))
        train_size = len(self.dates_train)

        plt.plot(all_dates[:train_size], all_Y[:train_size], 'o-',
                 color='steelblue', label='Реальное (обучение)', markersize=4)
        plt.plot(all_dates[:train_size], all_YR[:train_size], 's--',
                 color='crimson', label='Прогноз (обучение)', markersize=4)
        plt.plot(all_dates[train_size:], all_Y[train_size:], 'o-',
                 color='gray', label='Реальное (прогноз)', markersize=4, alpha=0.7)
        plt.plot(all_dates[train_size:], all_YR[train_size:], 'd--',
                 color='purple', label='Прогноз (24 ч вперёд)', markersize=5)

        plt.title('Прогноз потребления электроэнергии')
        plt.xlabel('Дата и время наблюдения')
        plt.ylabel('Потребление, кВт*ч')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
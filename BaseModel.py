from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Абстрактный класс для всех моделей"""

    def __init__(self, df, target_col, name=None, alpha=0.05):
        self.df = df
        self.target_col = target_col
        self.alpha = alpha
        self.dates_train = None
        self.Y = None
        self.YR = None
        self.metrics_train = {} # список рассчитанных метрик для обучающей выборки
        self.metrics_test = {} # список рассчитанных метрик для тестовой выборки
        self.is_fitted = False
        self.name = name

    @abstractmethod
    def fit(self, train_indices, features):
        """Обучает модель на указанных индексах и признаках"""
        pass

    @abstractmethod
    def predict(self, X):
        """Выдаёт матрицу-столбец Y с предсказанием"""
        pass

    def forecast(self, test_range):
        """Даёт прогноз на тестовые данные"""
        pass

    def validate(self, show=True):
        """Проверка адекватности (F-тест)"""
        pass

    def test_coefficients(self):
        """Проверка значимости коэффициентов (t-тест)"""
        pass

    def calculate_metrics(self, Y, YR, prefix):
        """Считает все метрики качества"""
        pass

    def plot_training(self, show_ci=True):
        """Строит график для обучающей выборки"""
        pass

    def plot_forecast(self, test_range):
        """Строит график прогноза на новые данные"""
        pass
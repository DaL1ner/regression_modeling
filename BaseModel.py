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
        self.is_fitted = False
        self.name = name

    @abstractmethod
    def fit(self, train_indices, features):
        """Обучает модель на указанных индексах и признаках"""
        pass

    @abstractmethod
    def predict(self, X_new):
        """Даёт предсказания для новых данных"""
        pass

    def validate(self):
        """Проверка адекватности (F-тест)"""
        pass

    def test_coefficients(self):
        """Проверка значимости коэффициентов (t-тест)"""
        pass

    def calculate_metrics(self, Y_true, YR, prefix):
        """Считает все метрики качества"""
        pass

    def plot_training(self, show_ci=True):
        """Строит график для обучающей выборки (как у тебя первый график)"""
        pass

    def plot_forecast(self, test_range):
        """Строит график прогноза на новые данные (как у тебя третий график)"""
        pass
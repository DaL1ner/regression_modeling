import pandas as pd

class ModelComparator:
    def __init__(self, models):
        self.models = models  # список обученных моделей

    def compare_metrics(self, metrics_to_show=None, on='train'):
        """Сравнивает модели по указанным метрикам"""
        if metrics_to_show is None:
            metrics_to_show = ['R²', 'RMSE', 'MAE']

        results = {}
        for model in self.models:
            if model.name is None:
                name = f"Модель_{id(model)}"
            else:
                name = model.name

            if on == 'train':
                metrics = model.metrics_train
            else:
                metrics = model.metrics_test

            results[name] = {metric: metrics.get(metric, 'N/A') for metric in metrics_to_show}

        return pd.DataFrame(results).T
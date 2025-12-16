import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    def plot_all_models(self, models, plot_type='training', show_ci=False):
        """
        Строит графики для всех моделей друг под другом.

        plot_type: 'training' или 'forecast'
        show_ci: показывать доверительные интервалы
        """
        n_models = len(models)
        if n_models == 0:
            print("Нет моделей для визуализации")
            return

        # Определяем размер фигуры
        fig_height = 6 * n_models
        fig, axes = plt.subplots(n_models, 1, figsize=(14, fig_height), sharex=True)

        # Если только одна модель — axes не список
        if n_models == 1:
            axes = [axes]

        for i, model in enumerate(models):
            ax = axes[i]

            if plot_type == 'training':
                # График обучающей выборки
                ax.plot(model.dates_train, model.Y, 'o-', label='Реальное значение, Y', color='blue')
                ax.plot(model.dates_train, model.YR, 's--', label='Предсказанное значение, YR', color='red')

                if show_ci:
                    ax.fill_between(
                        model.dates_train,
                        model.YR_lower,
                        model.YR_upper,
                        color='red', alpha=0.2, label='Доверительный интервал'
                    )

                    # Аннотация "3σ" (опционально, можно убрать для чистоты)
                    if len(model.dates_train) > 5:
                        i_annot = 5
                        x_annot = model.dates_train.iloc[i_annot]
                        y_upper = model.YR_upper[i_annot]
                        y_lower = model.YR_lower[i_annot]
                        offset_x = pd.Timedelta(hours=1)

                        ax.annotate('', xy=(x_annot, y_upper), xytext=(x_annot, y_upper + 15),
                                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, mutation_scale=20))
                        ax.annotate('', xy=(x_annot, y_lower), xytext=(x_annot, y_lower - 15),
                                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, mutation_scale=20))
                        ax.text(x_annot - offset_x, y_upper + 20, '3σ', fontsize=12, color='red',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))


                ax.set_title(f'Модель {model.name} | K={model.K}, N={model.N}')
                ax.set_ylabel('Потребление, кВт*ч')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
                ax.tick_params(axis='x', rotation=45)

                # Вертикальная пунктирная линия на стыке
                boundary_date = model.dates_train.iloc[-1]  # последняя дата обучения
                plt.axvline(x=boundary_date, color='black', linestyle='--', linewidth=1, label='Граница обучения')

            elif plot_type == 'forecast':
                # График прогноза
                all_dates = pd.concat([model.dates_train, model.dates_test])
                all_Y = np.concatenate([model.Y.flatten(), model.Y_test])
                all_YR = np.concatenate([model.YR.flatten(), model.YR_test])
                all_YR_lower = np.concatenate([model.YR_lower, model.YR_lower_test])
                all_YR_upper = np.concatenate([model.YR_upper, model.YR_upper_test])

                train_size = len(model.dates_train)

                ax.plot(all_dates[:train_size], all_Y[:train_size], 'o-',
                        color='steelblue', label='Реальное (обучение)', markersize=4)
                ax.plot(all_dates[:train_size], all_YR[:train_size], 's--',
                        color='crimson', label='Прогноз (обучение)', markersize=4)
                ax.plot(all_dates[train_size-1:], all_Y[train_size-1:], 'o-',
                        color='gray', label='Реальное (тест)', markersize=4, alpha=0.7)
                ax.plot(all_dates[train_size:], all_YR[train_size:], 'd--',
                        color='purple', label='Прогноз (тест)', markersize=5)

                # Заливка для обучения
                ax.fill_between(
                    all_dates[:train_size],
                    all_YR_lower[:train_size],
                    all_YR_upper[:train_size],
                    color='crimson', alpha=0.2, label='Доверительный интервал (обучение)'
                )

                # Заливка для прогноза
                ax.fill_between(
                    all_dates[train_size:],
                    all_YR_lower[train_size:],
                    all_YR_upper[train_size:],
                    color='purple', alpha=0.2, label='Доверительный интервал (прогноз)'
                )

                ax.set_title(f'Прогноз потребления электроэнергии — Модель {model.name}')
                ax.set_ylabel('Потребление, кВт*ч')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
                ax.tick_params(axis='x', rotation=45)

                # Вертикальная пунктирная линия на стыке
                boundary_date = model.dates_train.iloc[-1]  # последняя дата обучения
                ax.axvline(x=boundary_date, color='black', linestyle='--', linewidth=1, label='Граница обучения')

            # Подписываем ось X только на последнем графике
            if i == n_models - 1:
                ax.set_xlabel('Дата и время наблюдения')
            else:
                ax.set_xlabel('')

        plt.tight_layout()
        plt.show()
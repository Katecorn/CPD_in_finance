import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SyntheticDataGenerator:
    def __init__(self):
        pass
    
    def calculate_daily_returns(self, df):
        """
        Рассчитывает подневные доходности для каждого столбца, кроме 'date'.
        """
        returns_df = df.copy()
        returns_df.iloc[:, 1:] = df.iloc[:, 1:].pct_change()
        returns_df.dropna(inplace=True)
        return returns_df

    def apply_breakpoint(self, synthetic_values, breakpoint_type, shift_value, trend_slope, noise_level, min_value, max_value):
        """
        Применяет разладку к сегменту данных.
        """
        length = len(synthetic_values)
        
        if breakpoint_type == 'shift':
            synthetic_values += shift_value
        elif breakpoint_type == 'trend':
            trend_effect = trend_slope * np.arange(length)
            synthetic_values += trend_effect
        elif breakpoint_type == 'variance':
            variance_effect = np.random.normal(0, noise_level, length)
            synthetic_values += variance_effect
        
        # Ограничение значений ряда
        # synthetic_values = np.clip(synthetic_values, min_value, max_value)
        
        return synthetic_values

    def detect_outliers(self, series):
        """
        Определяет количество выбросов в ряду на основе метода IQR.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]

        # искуственно ограничиваем число change points, так как ряды могут быть очень волатильны
        if len(outliers) < 10 and len(outliers) > 0 :
            return len(outliers)
        elif len(outliers) == 0:
            return 1
        else:
            return 11
            
        
        return len(outliers)

    def generate_change_points(self, df):
        """
        Генерирует синтетический временной ряд с разными точками разладки для каждого ряда.

        Параметры:
        df (pd.DataFrame): Исходный датафрейм с ценами монет, включая столбец 'date'.

        Возвращает:
        tuple: (датафрейм с доходностями, датафрейм с разметкой change points, датафрейм с синтетическими ценами).
        """
        coins = df.columns[1:]  # Исключаем столбец 'date'
        returns_df = pd.DataFrame({'Date': df['Date'][1:]})  # Создаем датафрейм для доходностей
        change_points_df = pd.DataFrame({'Date': df['Date']})  # Создаем датафрейм для разметки change points
        synthetic_values_df = pd.DataFrame({'Date': df['Date']})  # Создаем датафрейм для синтетических цен

        for coin in coins:
            values = df[coin].values
            length = len(values)
            
            # Определяем количество точек разладки на основе количества выбросов в ряду
            num_breakpoints = self.detect_outliers(pd.Series(values))
            num_breakpoints = max(1, num_breakpoints)  # Убедимся, что есть хотя бы одна точка разладки

            breakpoints = np.sort(np.random.choice(np.arange(1, length-1), size=num_breakpoints, replace=False))
            breakpoints = np.append(breakpoints, length)  # Добавляем конец ряда для обработки последнего сегмента

            max_value = np.max(values) * 1.2
            min_value = np.min(values) * 0.8  # 0.8 чтобы учесть возможное уменьшение
            std_dev = np.std(values)
            range_value = np.ptp(values)

            synthetic_values = values.copy()
            for i, breakpoint_index in enumerate(breakpoints[:-1]):
                next_breakpoint = breakpoints[i + 1]
                breakpoint_type = np.random.choice(['shift', 'trend', 'variance'])
                
                # Подбор параметров разладки на основе характеристик ряда
                shift_value = np.random.uniform(0.01 * std_dev, 0.1 * std_dev)
                trend_slope = np.random.uniform(0.001, 0.005) * range_value / length
                noise_level = np.random.uniform(0.001 * std_dev, 0.1 * std_dev)

                segment_values = synthetic_values[breakpoint_index:next_breakpoint].copy()
                segment_values = self.apply_breakpoint(
                    segment_values,
                    breakpoint_type,
                    shift_value,
                    trend_slope,
                    noise_level,
                    min_value,
                    max_value
                )
                
                synthetic_values[breakpoint_index:next_breakpoint] = segment_values
            
            # Добавляем синтетические значения в датафрейм
            synthetic_values_df[coin] = synthetic_values
            synthetic_series = pd.Series(synthetic_values, name=coin)
            returns_df[coin] = synthetic_series.pct_change().iloc[1:].values
            
            # Создаем разметку change points для текущей монеты
            change_points = np.zeros(length)
            for bp in breakpoints[:-1]:
                change_points[bp] = 1
            change_points_series = pd.Series(change_points, name=coin)
            change_points_df[coin] = change_points_series

        returns_df.dropna(inplace=True)  # Удаляем NaN значения из доходностей
        return returns_df, change_points_df, synthetic_values_df

    def plot_series(self, original_df, synthetic_df, change_points_df):
        """
        Визуализирует исходный и синтетический временные ряды с отметкой точек разладки.

        Параметры:
        original_df (pd.DataFrame): Исходный датафрейм с колонками 'date' и ценами монет.
        synthetic_df (pd.DataFrame): Синтетический датафрейм с колонками 'date' и ценами монет.
        change_points_df (pd.DataFrame): Датафрейм с разметкой change points.
        """
        coins = original_df.columns[1:]  # Исключаем столбец 'date'
        for coin in coins:
            plt.figure(figsize=(15, 3))
            plt.subplot(2, 1, 1)
            plt.plot(original_df['Date'], original_df[coin], label='Original Series', color='blue')
            plt.plot(synthetic_df['Date'], synthetic_df[coin], alpha=0.8, label='Synthetic Series', color='orange')
            change_points = change_points_df[coin] == 1
            plt.scatter(original_df['Date'][change_points], synthetic_df[coin][change_points], color='red', label='Change Point', zorder=5)
            plt.xticks(rotation=45)
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title(f'Original and Synthetic Series with Change Points for {coin}')
            
            plt.subplot(2, 1, 2)
            original_returns = original_df[coin].pct_change().dropna()
            synthetic_returns = synthetic_df[coin].pct_change().dropna()
            plt.plot(original_df['Date'][1:], original_returns, label='Original Returns', color='blue')
            plt.plot(synthetic_df['Date'][1:], synthetic_returns, alpha=0.8, label='Synthetic Returns', color='orange')
            change_points_returns = change_points_df[coin].iloc[1:] == 1
            plt.scatter(original_df['Date'][1:][change_points_returns], synthetic_returns[change_points_returns], color='red', label='Change Point', zorder=5)
            plt.xticks(rotation=45)
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Returns')
            plt.title(f'Original and Synthetic Returns with Change Points for {coin}')
            plt.tight_layout()
            
            plt.show()

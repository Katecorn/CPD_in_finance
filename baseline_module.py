import time
import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np
from ruptures.metrics import hausdorff
from sklearn.model_selection import ParameterGrid
from scipy.stats import norm, lognorm
import random

class Stat:
    def __init__(self, threshold, direction="unknown", init_stat=0.0):
        self._direction = str(direction)
        self._threshold = float(threshold)
        self._stat = float(init_stat)
        self._alarm = self._stat / self._threshold

    @property
    def direction(self):
        return self._direction

    @property
    def stat(self):
        return self._stat

    @property
    def alarm(self):
        return self._alarm

    @property
    def threshold(self):
        return self._threshold

    def update(self, **kwargs):
        self._alarm = self._stat / self._threshold

    def reset_stat(self):
        self._stat = 0.0

def likelihood_ratio(x, mean_0, mean_8, variance):
    likelihood_x = norm.pdf(x, loc=mean_8, scale=np.sqrt(variance))
    likelihood_zero = norm.pdf(x, loc=mean_0, scale=np.sqrt(variance))
    return np.log(likelihood_x / likelihood_zero)

def log_likelihood_ratio(x, mean_0, mean_8, variance):
    likelihood_x = lognorm.logpdf(x, s=variance, loc=mean_8)
    likelihood_zero = lognorm.logpdf(x, s=variance, loc=mean_0)
    return likelihood_x - likelihood_zero

class Cumsum_mod(Stat):
    def __init__(self, d_m, threshold, direction="unknown", init_stat=0):
        self.d_m = d_m
        super(Cumsum_mod, self).__init__(threshold, direction, init_stat)

    def update(self, value):
        zeta_k = log_likelihood_ratio(value, self.d_m, 0, 1)
        self._stat = max(0, self._stat + zeta_k)
        super(Cumsum_mod, self).update()

class MeanExpNoDataException(Exception):
    pass

class MeanExp:
    def __init__(self, new_value_weight, load_function=np.median):
        self._load_function = load_function
        self._new_value_weight = new_value_weight
        self.load([])

    @property
    def value(self):
        if self._weights_sum <= 1:
            raise MeanExpNoDataException('self._weights_sum <= 1')
        return self._values_sum / self._weights_sum

    def update(self, new_value, **kwargs):
        self._values_sum = (1 - self._new_value_weight) * self._values_sum + new_value
        self._weights_sum = (1 - self._new_value_weight) * self._weights_sum + 1.0

    def load(self, old_values):
        if old_values:
            old_values = [value for ts, value in old_values]
            mean = float(self._load_function(old_values))
            self._weights_sum = min(float(len(old_values)), 1.0 / self._new_value_weight)
            self._values_sum = mean * self._weights_sum
        else:
            self._values_sum = 0.0
            self._weights_sum = 0.0

def cumsum_CPD(X, coin, alpha, beta, m_d, thresh, stat_udt=True, graph=False):
    stat_trajectory = []
    mean_values = []
    var_values = []

    mean_exp = MeanExp(new_value_weight=alpha)
    var_exp = MeanExp(new_value_weight=beta)
    cumsum = Cumsum_mod(m_d, thresh)

    change_points = []

    for i, x_i in enumerate(X):
        try:
            mean_estimate = mean_exp.value
        except MeanExpNoDataException:
            mean_estimate = 0.

        try:
            var_estimate = var_exp.value
        except MeanExpNoDataException:
            var_estimate = 1.

        adjusted_value = (x_i - mean_estimate) / np.sqrt(var_estimate)
        cumsum.update(adjusted_value)

        mean_exp.update(x_i)
        diff_value = (x_i - mean_estimate) ** 2
        var_exp.update(diff_value)

        stat_trajectory.append(cumsum._alarm)
        mean_values.append(mean_estimate)
        var_values.append(np.sqrt(var_estimate))

        if cumsum._alarm >= thresh:
            if stat_udt:
                change_points.append(i)
                cumsum.reset_stat()
            else:
                change_points.append(i)

    if graph:
        plt.figure(figsize=(10, 6))
        plt.plot(X, label=f'Исходный ряд {coin}', color='blue')
        plt.plot(mean_values, color='black', label='mean')

        for cp in change_points:
            plt.axvline(cp, color='red', linestyle='--')
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.title(f'Ряд {coin} с моментами разладки')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(stat_trajectory, label='Статистика', color='red')
        plt.axhline(thresh, color='green', linestyle='--', label='Порог срабатывания')
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.title(f'Статистика кумулятивных сумм {coin}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return stat_trajectory, mean_values, var_values, change_points

class ChangePointDetection:
    def __init__(self, data, method='pelt', model='l2', min_size=10, jump=5, metric='bic'):
        self.data = data
        self.method = method
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.metric = metric
        self.T = data.shape[0]
        self.sigma = np.std(data)
        self.bic = self.sigma ** 2 * np.log(self.T)

    def compute_aic(self):
        num_params = 2
        log_likelihood = -np.sum((self.data - np.mean(self.data))**2) / (2 * self.sigma**2)
        return 2 * num_params - 2 * log_likelihood

    def compute_penalty(self):
        if self.metric == 'aic':
            penalty = self.compute_aic()
        elif self.metric == 'bic':
            penalty = self.bic
        elif self.metric == 'var':
            penalty = self.sigma * self.sigma
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return penalty

    def detect(self, graph=False):
        start_time = time.time()
        penalty = self.compute_penalty()

        if self.method == 'pelt':
            model_instance = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump)
        elif self.method == 'binseg':
            model_instance = rpt.Binseg(model=self.model, min_size=self.min_size, jump=self.jump)
        elif self.method == 'kernelcpd':
            model_instance = rpt.KernelCPD()
        elif self.method == 'cumsum':
            result = cumsum_CPD(self.data, 'data', 0.01, 0.01, 1, penalty, graph=graph)[-1]
        elif self.method == 'sr':
            result = self.shiryaev_roberts()
        elif self.method == 'wbs':
            model_instance = rpt.Window(width=self.min_size, model=self.model)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.method not in ['cumsum', 'sr']:
            model_instance.fit(self.data)
            result = model_instance.predict(pen=penalty)

        if self.method not in ['cumsum', 'sr']:
            if result[-1] == len(self.data):
                result.pop()

        print(f'{time.time() - start_time:.3f} s')

        if graph:
            plt.figure(figsize=(15, 3))
            plt.plot(self.data)
            for cp in result:
                plt.axvline(x=cp, color='r', linestyle='--')
            plt.title(f'Method: {self.method}, Model: {self.model}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
        else:
            rpt.display(self.data, result)
            plt.show()

        return result

    def grid_search(self, param_grid):
        best_params = None
        best_score = float('inf')
        grid = ParameterGrid(param_grid)
        for params in grid:
            self.min_size = params.get('min_size', self.min_size)
            self.jump = params.get('jump', self.jump)
            detected_points = self.detect()
            try:
                if detected_points[-1] != len(self.data):
                    detected_points.append(len(self.data))

                bkps1 = [0] + detected_points
                bkps2 = [0, len(self.data)]
                score = hausdorff(bkps1, bkps2)

                if score < best_score:
                    best_score = score
                    best_params = params
            except ValueError as e:
                print(f"Skipping configuration due to bad partitions: {params}")
                continue

        self.min_size = best_params.get('min_size', self.min_size)
        self.jump = best_params.get('jump', self.jump)
        print(f"Best parameters: {best_params}, Score: {best_score}")

    def shiryaev_roberts(self):
        n = len(self.data)
        S = np.zeros(n)
        for t in range(1, n):
            S[t] = S[t - 1] + np.log(self.data[t] / self.data[t - 1])
        return np.where(S > np.mean(S) + 3 * np.std(S))[0]

    def compare_methods(self, real_change_points=[]):
        methods = ['pelt', 'binseg', 'kernelcpd', 'cumsum', 'sr', 'wbs']
        results = {}
        metrics = {}
        for method in methods:
            self.method = method
            print(f"Running method: {method}")
            result = self.detect()
            results[method] = result

            # Calculate metrics
            false_alarms = len([cp for cp in result if cp not in real_change_points])
            if len(real_change_points) > 0:
                mttr = np.mean([min([abs(cp - rcp) for rcp in real_change_points]) for cp in result])
            else:
                mttr = float('inf')
            metrics[method] = {'False Alarms': false_alarms, 'MTTR': mttr}

            # Plot results
            plt.figure(figsize=(15, 3))
            plt.plot(self.data, label='Data', color='blue')
            for cp in real_change_points:
                plt.axvline(x=cp, color='green', linestyle='-', label='Real CP' if cp == real_change_points[0] else "")
            for cp in result:
                plt.axvline(x=cp, color='red', linestyle='--', label=f'{method} CP' if cp == result[0] else "")
            plt.legend()
            plt.title(f'Method: {method}, Model: {self.model}\nFalse Alarms: {false_alarms}, MTTR: {mttr:.2f}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.show()

        # Собираем все точки изменений
        all_change_points = []
        for result in results.values():
            all_change_points.extend(result)

        # Находим точки, которые встречаются более чем в двух методах
        common_change_points = [point for point in set(all_change_points) if all_change_points.count(point) > 2]

        plt.figure(figsize=(15, 3))
        plt.plot(self.data, label='Data', color='blue')
        for method, result in results.items():
            for cp in result:
                plt.axvline(x=cp, linestyle='--', label=f'{method} CP' if method == list(results.keys())[0] else "")
        for cp in common_change_points:
            plt.axvline(x=cp, color='green', linestyle='-', label='Common CP' if cp == common_change_points[0] else "")
        plt.legend()
        plt.title('Comparison of Change Point Detection Methods')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()

        # Вывод метрик
        for method, metric in metrics.items():
            print(f"Method: {method}")
            print(f"  False Alarms: {metric['False Alarms']}")
            print(f"  MTTR: {metric['MTTR']:.2f}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, List, Dict


def variation_series(data: np.ndarray) -> np.ndarray:
    """
    Составить вариационный ряд.
    Здесь идёт сортировка данных значений
    """
    return np.sort(data)


def interval_statistics(data: np.ndarray, intervals: np.ndarray) -> pd.DataFrame:
    """
    Построить интервальный статистический ряд.
    Подсчитывается количество значений, встречающийхся на этом интервале
    """
    hist, bin_edges = np.histogram(data, bins=intervals)
    interval_df = pd.DataFrame({
        'Interval': [f"{bin_edges[i]} - {bin_edges[i + 1]}" for i in range(len(hist))],
        'Frequency': hist
    })
    return interval_df


def relative_frequency_polygon(data: np.ndarray, intervals: np.ndarray) -> None:
    """Построить полигон относительных частот и гистограмму."""
    hist, _ = np.histogram(data, bins=intervals) #вычисляем частоты на заданных интервалах
    relative_frequencies = hist / np.sum(hist) #делим выявленные ранее частоты на количество всех появлений значений

    plt.figure(figsize=(12, 6))

    # Гистограмма относительных частот
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=intervals, density=True, alpha=0.5, color='blue', edgecolor='black')
    plt.title('Гистограмма относительных частот')
    plt.xlabel('Значение')
    plt.ylabel('Относительная частота')

    # Полигон относительных частот
    plt.subplot(1, 2, 2)
    plt.plot(intervals[:-1], relative_frequencies, marker='o', linestyle='-', color='orange')
    plt.title('Полигон относительных частот')
    plt.xlabel('Значение')
    plt.ylabel('Относительная частота')

    plt.tight_layout()
    plt.show()


def empirical_distribution_function(data: np.ndarray) -> None:
    """Построить график эмпирической функции распределения."""
    ecdf = np.searchsorted(np.sort(data), np.sort(data)) / len(data)
    plt.figure(figsize=(8, 5))
    plt.step(np.sort(data), ecdf, where='post')
    plt.title('Эмпирическая функция распределения')
    plt.xlabel('Значение')
    plt.ylabel('F(x)')
    plt.grid()
    plt.show()


def descriptive_statistics(data: np.ndarray) -> Dict[str, float]:
    """Найти числовые характеристики выборки."""
    if data.size == 0:
        return {"mean": np.nan, "variance": np.nan, "mode": np.nan, "median": np.nan, "kurtosis": np.nan, "skewness": np.nan}

    mean = np.mean(data)
    variance = np.var(data, ddof=1)  # Неправильная выборочная дисперсия

    # Получение моды с обработкой исключений
    mode_result = stats.mode(data)
    mode = mode_result.mode if mode_result.count > 0 else np.nan
    median = np.median(data)
    kurtosis = stats.kurtosis(data, nan_policy='omit')
    skewness = stats.skew(data, nan_policy='omit')

    return mean, variance, mode, median, kurtosis, skewness


def fit_distribution_parameters(data: np.ndarray) -> Tuple[float, float]:
    """Fit parameters for a normal distribution."""
    mu = np.mean(data)
    std = np.std(data, ddof=1)
    return mu, std


def three_sigma_rule(data: np.ndarray) -> int:
    """Проверить выполнение правила «трех сигма»."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    within_three_sigma = np.sum((data >= (mean - 3 * std)) & (data <= (mean + 3 * std)))
    return within_three_sigma / len(data) * 100

def theoretical_graphics(data, intervals: np.ndarray):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Используем выборочное стандартное отклонение

    # Настройка графиков
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # 1. Гистограмма выборки
    axs[0].hist(data, bins=intervals, density=True, alpha=0.5, color='blue', edgecolor='black', label='Гистограмма выборки')
    axs[0].set_title('Гистограмма выборки')
    axs[0].set_xlabel('Значения')
    axs[0].set_ylabel('Плотность')
    axs[0].grid()
    axs[0].legend()

    # 2. Эмпирическая функция распределения
    ecdf_vals = np.searchsorted(np.sort(data), np.sort(data), side="right") / len(data)  # ECDF
    axs[1].step(np.sort(data), ecdf_vals, label='Эмпирическая функция распределения', color='blue', where='post')
    axs[1].set_title('Эмпирическая функция распределения')
    axs[1].set_xlabel('Значения')
    axs[1].set_ylabel('Вероятность')
    axs[1].grid()
    axs[1].legend()

    # 3. Теоретическая гистограмма (нормальное распределение)
    x = np.linspace(min(data), max(data), 100)
    p = stats.norm.pdf(x, mean, std_dev)  # Теоретическая функция плотности
    axs[0].plot(x, p, 'k', linewidth=2, label='Нормальное распределение (PDF)')
    axs[0].set_title('Теоретическая гистограмма')
    axs[0].set_xlabel('Значения')
    axs[0].set_ylabel('Плотность')
    axs[0].grid()
    axs[0].legend()

    # 4. Теоретическая функция распределения
    F_x = stats.norm.cdf(x, mean, std_dev)  # Функция распределения
    axs[1].plot(x, F_x, 'r-', label='Нормальное распределение (CDF)', linewidth=2)
    axs[1].set_title('Теоретическая функция распределения')
    axs[1].set_xlabel('Значения')
    axs[1].set_ylabel('Вероятность накопления')
    axs[1].grid()
    axs[1].legend()

    # Настройка общего оформления
    plt.tight_layout()
    plt.show()

# 1. Вычисление среднего и стандартного отклонения
def calculate_mean_std(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Используем ddof=1 для выборочного отклонения
    return mean, std

# 4. Критерий согласия Пирсона
def chisquare_test(data, alpha=0.05):
    mean, std_dev = fit_distribution_parameters(data)
    observed_freq, bins = np.histogram(data, bins=9)
    expected_freq = np.array([len(data) * (stats.norm.cdf(bins[i + 1], mean, std_dev) - stats.norm.cdf(bins[i], mean, std_dev)) for i in range(len(bins) - 1)])
    
    chi_squared = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
    
    # Вычисление степеней свободы
    df = len(observed_freq) - 1 - 2
    
    p_value = 1 - stats.chi2.cdf(chi_squared, df)
    
    # Вычисление критического значения хи-квадрат
    chi2_critical = stats.chi2.ppf(1 - alpha, df)
    
    return chi_squared, p_value, chi2_critical

# 5. Доверительные интервалы
def confidence_intervals(data, confidence=0.95):
    mean, std = calculate_mean_std(data)
    n = len(data)
    
    z_value = stats.norm.ppf((1 + confidence) / 2)
    mean_ci = (mean - z_value * std / np.sqrt(n), mean + z_value * std / np.sqrt(n))
    
    chi2_lower = stats.chi2.ppf(0.025, n-1)
    chi2_upper = stats.chi2.ppf(0.975, n-1)
    var_ci = ((n-1)*std**2 / chi2_upper, (n-1)*std**2 / chi2_lower)
    std_ci = (np.sqrt(var_ci[0]), np.sqrt(var_ci[1]))
    
    return mean_ci, std_ci

def generate_report(mean, variance, mode, median, kurtosis, skewness, std, 
                    mean_ci, variance_ci, chi2, p_value, percent, chi2_critical):
    report = f"""### Отчет по анализу выборки ###

Процент значений попадающих под "Правило трёх сигм": {percent}

1. **Основные статистические показатели:**
   - Среднее значение (mu): {mean:.2f}
   - Мода: {mode}
   - Медиана: {median:.2f}
   - Дисперсия: {variance:.2f}
   - Стандартное отклонение (std): {std:.2f}
   - Эксцесс: {kurtosis:.2f}
   - Асимметрия: {skewness:.2f}

2. **Доверительные интервалы:**
   - Доверительный интервал для средней: ({mean_ci[0]:.2f}, {mean_ci[1]:.2f})
   - Доверительный интервал для дисперсии: ({variance_ci[0]:.2f}, {variance_ci[1]:.2f})

3. **Критерий согласия Пирсона:**
   - χ² = {chi2:.2f}
   - p-значение = {p_value:.4f}
   - χ²-крит = {chi2_critical:.2f}

### Заключение:
На основании значения p-значения (< 0.05) можно сделать вывод о том, что выборка существенно отличается от нормального распределения, что подтверждается критерием согласия Пирсона.

Доверительные интервалы показывают, что, с вероятностью 95%, истинное среднее значение генеральной совокупности находится в пределах {mean_ci[0]:.2f}, {mean_ci[1]:.2f} и истинная дисперсия в пределах {variance_ci[0]:.2f}, {variance_ci[1]:.2f}.

### Общие рекомендации:
Для дальнейшего анализа целесообразно провести дополнительные исследования, например, исследовать нормальность данных с помощью других тестов или визуализировать распределение выборки.
"""
    return report
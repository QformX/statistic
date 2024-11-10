import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, List, Dict


def preprocess_text(text: str) -> str:
    russian_alphabet = 'абвгдежзийклмнопрстуфхцчшщыэюя'
    text = text.lower()
    return ''.join(c for c in text if c in russian_alphabet)


def variation_series(data: np.ndarray) -> np.ndarray:
    """Составить вариационный ряд."""
    return np.sort(data)


def interval_statistics(data: np.ndarray, intervals: np.ndarray) -> pd.DataFrame:
    """Построить интервальный статистический ряд."""
    hist, bin_edges = np.histogram(data, bins=intervals)
    interval_df = pd.DataFrame({
        'Interval': [f"{bin_edges[i]} - {bin_edges[i + 1]}" for i in range(len(hist))],
        'Frequency': hist
    })
    return interval_df


def relative_frequency_polygon(data: np.ndarray, intervals: np.ndarray) -> None:
    """Построить полигон относительных частот и гистограмму."""
    hist, _ = np.histogram(data, bins=intervals)
    relative_frequencies = hist / np.sum(hist)

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

    return {
        'mean': mean,
        'variance': variance,
        'mode': mode,
        'median': median,
        'kurtosis': kurtosis,
        'skewness': skewness
    }


def hypothesis_distribution(data: np.ndarray) -> None:
    """Выдвинуть гипотезу о распределении генеральной совокупности на основе гистограммы."""
    sns.histplot(data, kde=True)
    plt.title('Гистограмма и KDE для проверки распределения')
    plt.show()


def fit_distribution_parameters(data: np.ndarray) -> Tuple[float, float]:
    """Fit parameters for a normal distribution."""
    mu = np.mean(data)
    std = np.std(data, ddof=1)
    return mu, std


def theoretical_distribution(data: np.ndarray, mu: float, std: float, intervals: np.ndarray) -> None:
    """Построить теоретические аналоги гистограммы и эмпирической функции."""
    x = np.linspace(np.min(data), np.max(data), 100)
    pdf = stats.norm.pdf(x, mu, std)
    
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=intervals, density=True, alpha=0.5, color='blue', edgecolor='black', label='Гистограмма')
    plt.plot(x, pdf, color='red', label='Теоретическая PDF')
    plt.title('Сравнение гистограммы и теоретической PDF')
    plt.legend()
    plt.show()


def three_sigma_rule(data: np.ndarray) -> int:
    """Проверить выполнение правила «трех сигма»."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    within_three_sigma = np.sum((data >= (mean - 3 * std)) & (data <= (mean + 3 * std)))
    return within_three_sigma


def chi_square_test(data: np.ndarray, intervals: np.ndarray) -> Tuple[float, float]:
    """Apply Pearson's Chi-square test."""
    hist, _ = np.histogram(data, bins=intervals)
    mu, std = fit_distribution_parameters(data)

    # Calculate expected frequencies for a normal distribution
    bin_centers = (intervals[:-1] + intervals[1:]) / 2
    expected_frequencies = len(data) * stats.norm.pdf(bin_centers, mu, std) * np.diff(intervals)

    # Debug print to check the sums
    observed_sum = np.sum(hist)
    expected_sum = np.sum(expected_frequencies)
    print(f"Observed sum: {observed_sum}, Expected sum: {expected_sum}")

    # Check if the expected frequencies sum up correctly with tolerance
    if not np.isclose(observed_sum, expected_sum, rtol=1e-5):
        print(f"Warning: The sums do not match closely. Diff: {observed_sum - expected_sum}")

    # Add a condition to check the dimensions
    if hist.shape[0] != expected_frequencies.shape[0]:
        print("Warning: The dimensions of hist and expected_frequencies do not match.")

    # Perform Chi-Square test
    try:
        chi2_stat, p_value = stats.chisquare(f_obs=hist, f_exp=expected_frequencies)
    except ValueError as e:
        print(f"Error in Chi-Square calculation: {e}")
        print("Check the sums and dimensions of the observed/expected frequencies.")
        return None, None

    return chi2_stat, p_value


def confidence_intervals(data: np.ndarray, confidence_level: float = 0.95) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Find confidence intervals for the population mean and population variance."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    variance = np.var(data, ddof=1)  # Calculate variance
    n = len(data)
    h = std / np.sqrt(n)

    mean_interval = (mean - h * stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1),
                     mean + h * stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1))
    variance_interval = (variance * (n - 1) / stats.chi2.ppf(1 - confidence_level / 2, n - 1),
                         variance * (n - 1) / stats.chi2.ppf(confidence_level / 2, n - 1))

    return mean_interval, variance_interval
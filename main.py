from func import *

# Список чисел из предоставленной выборки
data = [
    127, 141, 121, 131, 145, 139, 141, 131, 139, 137, 128, 132,
    129, 134, 140, 143, 140, 127, 132, 136, 134, 133, 121, 133,
    136, 126, 138, 144, 138, 138, 137, 137, 127, 139, 140, 141,
    125, 136, 136, 129, 136, 135, 142, 132, 129, 128, 134, 139,
    127, 133, 140, 135, 140, 130, 130, 130, 131, 133, 133, 131, 
    129, 127, 138, 128, 122, 137, 136, 137, 140, 138, 117, 114,
    132, 137, 126, 137, 136, 130, 148, 130, 140, 132, 127, 125,
    134, 136, 116, 120, 125, 136, 124, 144, 133, 132, 123, 136,
    121, 130, 130, 134, 132, 131, 125, 138, 130, 146, 132, 131,
    138, 135, 133, 126, 134, 137, 141, 136, 126, 125, 135, 139,
    150, 130, 136, 132, 127, 131, 122, 125, 133, 133, 123, 136,
    131, 135, 130, 123, 133, 123, 140, 133, 138, 124, 129, 129,
    128, 126, 128, 138, 127, 120, 144, 135, 126, 126, 144, 125,
    123, 132, 138, 155, 139, 137, 133, 129, 124, 140, 131, 128,
    130, 130, 124, 142, 124, 129, 131, 143, 129, 127
]

data = np.array(data)

intervals = np.arange(110, 160, 5)

# Вызов функций
print(f'Вариационный ряд:\n {variation_series(data)}')
print(f'Интервальный статистический ряд:\n {interval_statistics(data, intervals)}')
relative_frequency_polygon(data, intervals)
empirical_distribution_function(data)
theoretical_graphics(data, intervals)
mean, variance, mode, median, kurtosis, skewness = descriptive_statistics(data)
mu, std = fit_distribution_parameters(data)
percent_three_sigma = three_sigma_rule(data)
mean_interval, variance_interval = confidence_intervals(data, confidence=0.95)

chi2, p_value, chi2_critical = chisquare_test(data)

mean_ci, std_ci = confidence_intervals(data)

print(generate_report(mean, variance, mode, median, kurtosis, skewness, std, mean_ci, std_ci, chi2, p_value, percent_three_sigma, chi2_critical))

_, p_value = stats.normaltest(data)
print(f"P-значение теста на нормальность: {p_value}")



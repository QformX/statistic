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
variation_series(data)
print(interval_statistics(data, intervals))
relative_frequency_polygon(data, intervals)
empirical_distribution_function(data)
print(descriptive_statistics(data))
hypothesis_distribution(data)
mu, std = fit_distribution_parameters(data)
print(f'Estimated parameters: mu={mu}, std={std}')
theoretical_distribution(data, mu, std, intervals)
print(three_sigma_rule(data))
chi2_stat, p_value = chi_square_test(data, intervals)
print(f'Chi-square statistic: {chi2_stat}, p-value: {p_value}')
mean_interval, variance_interval = confidence_intervals(data, confidence_level=0.95)
print(f'Mean Confidence Interval: {mean_interval}')
print(f'Variance Confidence Interval: {variance_interval}')



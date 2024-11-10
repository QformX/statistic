import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def interval_statistical_series(data, num_intervals):
    # Находим минимальное и максимальное значения в данных
    min_value = min(data)
    max_value = max(data)

    # Создаем интервалы
    intervals = np.linspace(min_value, max_value, num_intervals + 1)
    
    # Создаем список для хранения частот
    frequency = []
    
    # Подсчитываем частоту каждого интервала
    for i in range(len(intervals) - 1):
        count = sum((data >= intervals[i]) & (data < intervals[i + 1]))
        frequency.append(count)
    
    # Создаем DataFrame для удобного отображения
    intervals_labels = [f"[{intervals[i]:.2f}, {intervals[i + 1]:.2f})" for i in range(len(intervals) - 1)]
    
    result = pd.DataFrame({'Intervals': intervals_labels, 'Frequency': frequency})
    
    return result

def variation_series(data):
    # Удаляем дубликаты и сортируем данные
    unique_data = sorted(set(data))
    
    # Создаем вариационный ряд
    series = []
    for value in unique_data:
        count = data.count(value)
        series.append((value, count))  # Каждое уникальное значение с количеством его повторений

    return series

def plot_relative_frequency(data, num_intervals):
    # Находим минимальное и максимальное значения в данных
    min_value = min(data)
    max_value = max(data)

    # Создаем интервалы
    intervals = np.linspace(min_value, max_value, num_intervals + 1)
    
    # Подсчитываем частоту каждого интервала
    frequency = []
    
    for i in range(len(intervals) - 1):
        count = sum((data >= intervals[i]) & (data < intervals[i + 1]))
        frequency.append(count)

    # Рассчитываем относительные частоты
    total_count = sum(frequency)
    relative_frequency = [f / total_count for f in frequency]
    
    # Создаем DataFrame для удобного отображения
    intervals_labels = [f"[{intervals[i]:.2f}, {intervals[i + 1]:.2f})" for i in range(len(intervals) - 1)]
    
    result = pd.DataFrame({'Intervals': intervals_labels, 'Relative Frequency': relative_frequency})

    # Построение гистограммы
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=intervals, density=True, alpha=0.5, color='blue', edgecolor='black', label='Гистограмма')
    
    # Построение полигона относительных частот
    mid_points = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]
    plt.plot(mid_points, relative_frequency, marker='o', color='red', linestyle='-', label='Полигон относительных частот')
    
    plt.title('Полигон и гистограмма относительных частот')
    plt.xlabel('Значение')
    plt.ylabel('Относительная частота')
    plt.xticks(mid_points, rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

    return result
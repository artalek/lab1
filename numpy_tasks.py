"""Решения задач второй части лабораторной работы по NumPy."""

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    """Суммирует произведения соответствующих матриц и векторов."""
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)

    if len(matrices) != len(vectors):
        raise ValueError("Количество матриц и векторов должно совпадать.")

    if len(matrices) == 0:
        raise ValueError("Набор матриц и векторов не должен быть пустым.")

    # (p, n, n) @ (p, n, 1) -> (p, n, 1)
    products = matrices @ vectors

    # Складываем результаты по первой оси.
    return np.sum(products, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    """Заменяет элементы > threshold на 1, остальные на 0."""
    matrix = np.asarray(data.matrix)
    return (matrix > data.threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    """Возвращает уникальные значения каждой строки."""
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    """Возвращает уникальные значения каждого столбца."""
    matrix = np.asarray(data.matrix)
    return [
        np.unique(matrix[:, j]).tolist()
        for j in range(matrix.shape[1])
    ]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    """Создаёт случайную нормальную матрицу и вычисляет статистики."""
    rows = data.rows
    columns = data.columns
    mean = data.mean
    std = data.std
    seed = data.seed

    rng = np.random.default_rng(seed)
    matrix = rng.normal(
        loc=mean,
        scale=std,
        size=(rows, columns),
    )

    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    """Создаёт шахматную матрицу из двух заданных значений."""
    rows = data.rows
    columns = data.columns
    first = data.first
    second = data.second

    indexes = np.indices((rows, columns)).sum(axis=0)
    return np.where(indexes % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    """Рисует закрашенный прямоугольник в центре RGB-изображения."""
    width = data.width
    height = data.height
    image_height = data.image_height
    image_width = data.image_width
    shape_color = data.shape_color
    background_color = data.background_color

    if width < 0 or height < 0:
        raise ValueError("Ширина и высота должны быть неотрицательными.")

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    # Координаты левого верхнего угла.
    x0 = (image_width - width) // 2
    y0 = (image_height - height) // 2

    # Обрезаем координаты, если прямоугольник больше изображения.
    x1 = x0 + width
    y1 = y0 + height

    left = max(0, x0)
    right = min(image_width, x1)
    top = max(0, y0)
    bottom = min(image_height, y1)

    if left < right and top < bottom:
        image[top:bottom, left:right] = shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    """Рисует закрашенный эллипс в центре RGB-изображения."""
    a = data.semi_axis_x
    b = data.semi_axis_y
    image_height = data.image_height
    image_width = data.image_width
    shape_color = data.shape_color
    background_color = data.background_color

    if a <= 0 or b <= 0:
        raise ValueError("Полуоси эллипса должны быть положительными.")

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    # Центр изображения в координатах пикселей.
    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2

    y, x = np.ogrid[:image_height, :image_width]

    # Уравнение внутренней области эллипса.
    mask = (
        ((x - x0) / a) ** 2
        + ((y - y0) / b) ** 2
        <= 1
    )

    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    """Вычисляет статистики, экстремумы и скользящее среднее ряда."""
    values = np.asarray(data.values, dtype=float)
    window = data.window

    if values.ndim != 1:
        raise ValueError("Временной ряд должен быть одномерным.")

    if window < 1 or window > len(values):
        raise ValueError(
            "Размер окна должен быть от 1 до длины временного ряда."
        )

    # Строгие локальные экстремумы; крайние элементы не учитываем.
    local_maxima_indices = [
        i
        for i in range(1, len(values) - 1)
        if values[i] > values[i - 1]
        and values[i] > values[i + 1]
    ]

    local_minima_indices = [
        i
        for i in range(1, len(values) - 1)
        if values[i] < values[i - 1]
        and values[i] < values[i + 1]
    ]

    # Скользящее среднее всех участков длины window.
    moving_average = np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )

    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    """Преобразует метки классов в one-hot матрицу."""
    labels = np.asarray(data.labels, dtype=int)
    class_count = data.class_count

    if labels.ndim != 1:
        raise ValueError("Метки должны быть одномерным вектором.")

    if np.any(labels < 0):
        raise ValueError("Метки классов должны быть неотрицательными.")

    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0

    if class_count < 0:
        raise ValueError("Количество классов не может быть отрицательным.")

    if labels.size and np.any(labels >= class_count):
        raise ValueError("Метка выходит за пределы class_count.")

    result = np.zeros((len(labels), class_count), dtype=int)

    if labels.size:
        result[np.arange(len(labels)), labels] = 1

    return result


def plot_matrix_statistics(statistics: MatrixStatistics) -> None:
    """Строит гистограммы средних значений строк и столбцов."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(statistics.matrix)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(matrix.mean(axis=1), bins="auto")
    axes[0].set_title("Средние значения строк")
    axes[0].set_xlabel("Среднее")
    axes[0].set_ylabel("Частота")

    axes[1].hist(matrix.mean(axis=0), bins="auto")
    axes[1].set_title("Средние значения столбцов")
    axes[1].set_xlabel("Среднее")
    axes[1].set_ylabel("Частота")

    fig.tight_layout()
    plt.show()
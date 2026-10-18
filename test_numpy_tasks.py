import unittest
from types import SimpleNamespace as NS

import numpy as np

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)


class TestNumpyTasks(unittest.TestCase):

    # Задача 1. Сумма произведений матриц на векторы
    def test_sum_prod(self):
        data = NS(
            matrices=[
                np.array([[1, 0], [0, 2]]),
                np.array([[2, 1], [1, 1]]),
            ],
            vectors=[
                np.array([[3], [4]]),
                np.array([[1], [2]]),
            ],
        )

        # A1*x1 = [[3], [8]]
        # A2*x2 = [[4], [3]]
        # Сумма = [[7], [11]]
        result = sum_prod(data)
        np.testing.assert_array_equal(result, [[7], [11]])
        self.assertEqual(result.shape, (2, 1))

    def test_sum_prod_single_pair(self):
        data = NS(
            matrices=[np.eye(2)],
            vectors=[np.array([[7], [8]])],
        )
        np.testing.assert_array_equal(
            sum_prod(data),
            [[7], [8]],
        )

    # Задача 2. Бинаризация матрицы
    def test_binarize(self):
        original = np.array([
            [1, 2],
            [3, 2],
        ])

        result = binarize(NS(matrix=original, threshold=2))

        np.testing.assert_array_equal(
            result,
            [[0, 0], [1, 0]],
        )

        # Исходная матрица не должна измениться.
        np.testing.assert_array_equal(
            original,
            [[1, 2], [3, 2]],
        )
        self.assertIsNot(result, original)

    def test_binarize_negative_threshold(self):
        data = NS(
            matrix=np.array([[-3, -1, 0, 2]]),
            threshold=-1,
        )
        np.testing.assert_array_equal(
            binarize(data),
            [[0, 0, 1, 1]],
        )

    # Задача 3. Уникальные строки и столбцы
    def test_unique_rows(self):
        data = NS(matrix=np.array([
            [2, 1, 2],
            [4, 4, 3],
            [5, 5, 5],
        ]))

        self.assertEqual(
            unique_rows(data),
            [[1, 2], [3, 4], [5]],
        )

    def test_unique_columns(self):
        data = NS(matrix=np.array([
            [2, 1, 2],
            [4, 1, 3],
            [2, 5, 3],
        ]))

        self.assertEqual(
            unique_columns(data),
            [[2, 4], [1, 5], [2, 3]],
        )

    # Задача 4. Статистики случайной матрицы
    def test_matrix_statistics(self):
        data = NS(
            rows=3,
            columns=4,
            mean=2.0,
            std=0.5,
            seed=123,
        )

        result = matrix_statistics(data)

        self.assertEqual(result.matrix.shape, (3, 4))

        np.testing.assert_allclose(
            result.row_means,
            result.matrix.mean(axis=1),
        )
        np.testing.assert_allclose(
            result.column_means,
            result.matrix.mean(axis=0),
        )
        np.testing.assert_allclose(
            result.row_variances,
            result.matrix.var(axis=1),
        )
        np.testing.assert_allclose(
            result.column_variances,
            result.matrix.var(axis=0),
        )

    def test_matrix_statistics_reproducible(self):
        data = NS(
            rows=3,
            columns=3,
            mean=0.0,
            std=1.0,
            seed=42,
        )

        first = matrix_statistics(data)
        second = matrix_statistics(data)

        np.testing.assert_array_equal(
            first.matrix,
            second.matrix,
        )

    # Задача 5. Шахматная матрица
    def test_chess(self):
        result = chess(
            NS(rows=3, columns=4, first=0, second=1)
        )

        np.testing.assert_array_equal(
            result,
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
        )

    def test_chess_one_element(self):
        result = chess(
            NS(rows=1, columns=1, first=8, second=9)
        )
        np.testing.assert_array_equal(result, [[8]])

    # Задача 6. Прямоугольник и эллипс
    def test_draw_rectangle(self):
        data = NS(
            width=2,
            height=2,
            image_height=5,
            image_width=5,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )

        image = draw_rectangle(data)

        self.assertEqual(image.shape, (5, 5, 3))
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(image[2, 2], [255, 0, 0])

        # Для прямоугольника 2x2 должны быть ровно 4 цветных пикселя.
        colored_pixels = np.all(
            image == [255, 0, 0],
            axis=2,
        )
        self.assertEqual(int(colored_pixels.sum()), 4)

    def test_draw_ellipse(self):
        data = NS(
            semi_axis_x=2,
            semi_axis_y=1,
            image_height=5,
            image_width=5,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )

        image = draw_ellipse(data)

        self.assertEqual(image.shape, (5, 5, 3))
        np.testing.assert_array_equal(image[2, 2], [255, 0, 0])
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])

    # Задача 7. Анализ временного ряда
    def test_analyze_time_series(self):
        values = [1, 3, 1, 0, 2, 0]
        data = NS(values=values, window=2)

        result = analyze_time_series(data)

        self.assertAlmostEqual(result.mean, np.mean(values))
        self.assertAlmostEqual(result.variance, np.var(values))
        self.assertAlmostEqual(result.std, np.std(values))

        self.assertEqual(result.local_maxima_indices, [1, 4])
        self.assertEqual(result.local_minima_indices, [3])

        np.testing.assert_allclose(
            result.moving_average,
            [2, 2, 0.5, 1, 1],
        )

    def test_time_series_window_one(self):
        data = NS(values=[1, 2, 3], window=1)
        result = analyze_time_series(data)

        np.testing.assert_allclose(
            result.moving_average,
            [1, 2, 3],
        )

    # Задача 8. One-hot encoding
    def test_one_hot(self):
        data = NS(
            labels=[0, 2, 3, 0],
            class_count=None,
        )

        result = one_hot(data)

        np.testing.assert_array_equal(
            result,
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
        )

    def test_one_hot_explicit_class_count(self):
        data = NS(
            labels=[1, 0],
            class_count=3,
        )

        np.testing.assert_array_equal(
            one_hot(data),
            [
                [0, 1, 0],
                [1, 0, 0],
            ],
        )

    def test_one_hot_empty(self):
        data = NS(labels=[], class_count=None)
        self.assertEqual(one_hot(data).shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
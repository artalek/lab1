"""Тесты: запуск python -m unittest -v test_python_basics.py."""
import unittest
from types import SimpleNamespace
from python_basics import (
    count_vowels, has_unique_characters, count_one_bits,
    multiplicative_persistence, mse, prime_factorization,
    pyramid, is_balanced_number,
)

def text(value):
    return SimpleNamespace(value=value)

def positive(value):
    return SimpleNamespace(value=value)

def vectors(predicted, expected):
    return SimpleNamespace(predicted=predicted, expected=expected)

class TestPythonBasics(unittest.TestCase):
    def test_count_vowels(self):
        self.assertEqual(count_vowels(text("hi my name")), 3)
        self.assertEqual(count_vowels(text("AEIOU")), 5)
        self.assertEqual(count_vowels(text("rhythm")), 0)
        self.assertEqual(count_vowels(text("")), 0)

    def test_unique_characters(self):
        self.assertTrue(has_unique_characters(text("abc")))
        self.assertFalse(has_unique_characters(text("abca")))
        self.assertFalse(has_unique_characters(text("  ")))
        self.assertTrue(has_unique_characters(text("")))

    def test_one_bits(self):
        self.assertEqual(count_one_bits(positive(1)), 1)
        self.assertEqual(count_one_bits(positive(7)), 3)
        self.assertEqual(count_one_bits(positive(8)), 1)
        self.assertEqual(count_one_bits(positive(255)), 8)

    def test_persistence(self):
        self.assertEqual(multiplicative_persistence(positive(39)), 3)
        self.assertEqual(multiplicative_persistence(positive(4)), 0)
        self.assertEqual(multiplicative_persistence(positive(999)), 4)
        self.assertEqual(multiplicative_persistence(positive(10)), 1)

    def test_mse(self):
        self.assertAlmostEqual(mse(vectors([1, 2, 3], [1, 2, 4])), 1 / 3)
        self.assertEqual(mse(vectors([1, 2], [3, 4])), 4.0)
        self.assertEqual(mse(vectors([5], [5])), 0.0)
        with self.assertRaises(ValueError):
            mse(vectors([], []))

    def test_factorization(self):
        self.assertEqual(prime_factorization(positive(86240)), "(2**5)(5)(7**2)(11)")
        self.assertEqual(prime_factorization(positive(13)), "(13)")
        self.assertEqual(prime_factorization(positive(12)), "(2**2)(3)")
        self.assertEqual(prime_factorization(positive(1)), "")

    def test_pyramid(self):
        self.assertEqual(pyramid(positive(1)), 1)
        self.assertEqual(pyramid(positive(6)), "It is impossible")
        self.assertEqual(pyramid(positive(14)), 3)
        self.assertEqual(pyramid(positive(30)), 4)

    def test_balanced_number(self):
        self.assertTrue(is_balanced_number(positive(1234006)))
        self.assertFalse(is_balanced_number(positive(123456)))
        self.assertTrue(is_balanced_number(positive(121)))
        self.assertTrue(is_balanced_number(positive(11)))
        self.assertTrue(is_balanced_number(positive(7)))

if __name__ == "__main__":
    unittest.main()
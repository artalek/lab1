"""Решения задач первой части лабораторной работы по основам Python."""
from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(char in "aeiou" for char in text.lower())


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    return bin(data.value).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Векторы должны иметь одинаковую длину.")
    if not predicted:
        raise ValueError("Нельзя вычислить MSE для пустых векторов.")
    return sum((x - y) ** 2 for x, y in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power == 1:
            factors.append(f"({divisor})")
        elif power > 1:
            factors.append(f"({divisor}**{power})")
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        factors.append(f"({number})")
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    level = total = 0
    while total < cube_count:
        level += 1
        total += level * level
    return level if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)
    middle = len(digits) // 2
    if len(digits) % 2:
        left, right = digits[:middle], digits[middle + 1:]
    else:
        left, right = digits[:middle - 1], digits[middle + 1:]
    return sum(map(int, left)) == sum(map(int, right))
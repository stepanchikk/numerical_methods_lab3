"""
Метод Ньютона для системи нелінійних рівнянь.

Відповідальний: Олександр
Статус: TODO

Що потрібно зробити:
  Функція solve(x0, eps) -> (x, y, records)

  1. Взяти початкове наближення x(0) з області G (з localization.py).
  2. На кожному кроці k:
       - обчислити матрицю Якобі W(x(k-1)) і вектор F(x(k-1)) (з common.py);
       - розв'язати СЛАР 2x2:  W(x(k-1)) · Δ(k) = -F(x(k-1))
         (методом Крамера, оберненої матриці або Гаусса);
       - обчислити x(k) = x(k-1) + Δ(k).
  3. Умова зупинки: max(|x1(k) - x1(k-1)|, |x2(k) - x2(k-1)|) <= ε.
  4. Зберігати кожну ітерацію в records для таблиці.
  5. Якщо det W = 0 або перевищено MAX_ITER — повідомити про помилку.

Для звіту: розписати першу ітерацію вручну (W, F, Δ, нове наближення).
"""
import math

from src.common import EPS, MAX_ITER, F, jacobian_W, make_record


def _solve_linear_system(W, b):
    """
    Розв'язок СЛАР 2x2  W · Δ = b  методом Крамера.

    | a11 a12 | | d1 |   | b1 |
    | a21 a22 | | d2 | = | b2 |
    """
    a11, a12 = W[0]
    a21, a22 = W[1]
    det = a11 * a22 - a12 * a21
    if det == 0.0 or not math.isfinite(det):
        raise ZeroDivisionError("det W = 0: метод Ньютона не застосовний "
                                "(Якобі вироджений)")
    d1 = (b[0] * a22 - a12 * b[1]) / det
    d2 = (a11 * b[1] - b[0] * a21) / det
    return d1, d2, det


def solve(x0, eps=EPS):
    """
    Метод Ньютона для системи F(x) = 0.

    x0  — початкове наближення (x, y) з області G;
    eps — точність ε (умова зупинки по максимуму змін координат).

    Повертає (x, y, records), де records — список ітерацій
    з полями k, x, y, dx, dy, delta.
    """
    x, y = float(x0[0]), float(x0[1])
    records = [make_record(0, x, y)]

    for k in range(1, MAX_ITER + 1):
        W = jacobian_W(x, y)              # матриця Якобі W(x(k-1))
        F1, F2 = F(x, y)                  # вектор F(x(k-1))
        dx, dy, _ = _solve_linear_system(W, (-F1, -F2))   # W · Δ = -F
        x_new, y_new = x + dx, y + dy     # x(k) = x(k-1) + Δ(k)

        record = make_record(k, x_new, y_new, x, y)
        records.append(record)

        x, y = x_new, y_new

        # Умова зупинки: max(|Δx1|, |Δx2|) <= ε
        if record["delta"] <= eps:
            return x, y, records

    raise RuntimeError(f"Метод Ньютона не збігся за {MAX_ITER} ітерацій "
                       f"(поточне наближення: x = {x:.6f}, y = {y:.6f}, "
                       f"Δ = {records[-1]['delta']:.6f} > ε = {eps})")

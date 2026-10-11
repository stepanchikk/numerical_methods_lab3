"""
Метод Зейделя для системи нелінійних рівнянь.

Відповідальний: Тарас
"""

import math

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import MAX_ITER, make_record, phi1, phi2


def solve(x0, threshold, max_iter=MAX_ITER):
    """Повертає (x, y, records) для заданого початкового наближення.

    threshold — додатний поріг для max(|dx|, |dy|).
    records містить початковий рядок k=0 та всі виконані ітерації
    з полями k, x, y, dx, dy, delta.
    Якщо збіжності не досягнуто за max_iter кроків, піднімає RuntimeError.
    """
    x, y = float(x0[0]), float(x0[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Початкове наближення має містити скінченні числа")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Поріг зупинки має бути додатним скінченним числом")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Кількість ітерацій має бути додатним цілим числом")

    records = [make_record(0, x, y)]

    for k in range(1, max_iter + 1):
        x_new = phi1(x, y)
        y_new = phi2(x_new, y)
        if not math.isfinite(x_new) or not math.isfinite(y_new):
            raise RuntimeError(f"Метод Зейделя: нескінченне або невизначене значення на ітерації {k}")

        record = make_record(k, x_new, y_new, x, y)
        records.append(record)
        x, y = x_new, y_new
        if record["delta"] <= threshold:
            return x, y, records

    raise RuntimeError(f"Метод Зейделя не зійшовся за {max_iter} ітерацій")


def run_report():
    from src.common import EPS, X0, print_table, verify
    from src.localization import estimate_q_analytic, stop_threshold
    from src.simple_iteration import solve as solve_mpi

    q = estimate_q_analytic()[2]
    if not 0 <= q < 1:
        raise ValueError("Для цього порогу зупинки потрібна умова 0 <= q < 1")
    threshold = stop_threshold(q, EPS)
    x, y, records = solve(X0, threshold)
    x_mpi, y_mpi, records_mpi = solve_mpi(X0, threshold, max_iter=MAX_ITER)
    for name, result_x, result_y, history in (
        ("Зейдель", x, y, records),
    ):
        residual = max(abs(value) for value in verify(result_x, result_y))
        print(f"{name:<12} | {len(history) - 1:>8} | {result_x:>12.9f} | "
              f"{result_y:>12.9f} | {residual:>15.3e}")




if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_report()

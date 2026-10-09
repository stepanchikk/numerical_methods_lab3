"""
Лабораторна робота №3. Розв'язування систем нелінійних рівнянь. Варіант 7.

    x + cos(y) = 1,5
    2y - sin(x - 0,5) = 1          ε = 0,001

Відповідальний: Степан (тімлід)

Головний файл — точка входу. Запуск: python main.py

Що потрібно зробити (по черзі викликати частини всіх учасників):
  1. src/localization.py    — Степан:    локалізація, q, поріг зупинки
  2. src/newton.py          — Олександр: метод Ньютона    -> таблиця 1
  3. src/simple_iteration.py — Даниїл:   метод простої ітерації -> таблиця 2
  4. src/seidel.py          — Тарас:     метод Зейделя    -> таблиця 3
  5. Степан: перевірка розв'язків підстановкою у систему,
     порівняльна таблиця методів (кількість ітерацій, x*, y*, нев'язки),
     відповідь, округлена до 0,001.
"""

import sys

from src import localization, newton, simple_iteration, seidel
from src.common import X0, EPS, print_table, verify

# щоб українські символи коректно виводились у будь-якій консолі
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LINE = "=" * 86


def run_method(stage, title, author, module, table_no, **kwargs):
    """
    Запускає solve() одного учасника і друкує таблицю ітерацій.
    Повертає (x, y, records) або None, якщо метод ще не готовий / впав.
    """
    print(f"\nЕТАП {stage}. {title.upper()}  ({author})")

    if not hasattr(module, "solve"):
        print(f"  [TODO] {author}: у {module.__name__} ще немає функції solve()")
        return None

    try:
        x, y, records = module.solve(X0, **kwargs)
    except Exception as e:  # щоб помилка в одному методі не зупиняла інші
        print(f"  [ПОМИЛКА] {author}: {type(e).__name__}: {e}")
        return None

    threshold = kwargs.get("eps", kwargs.get("threshold"))
    print_table(f"Таблиця {table_no}. {title}", records, threshold)
    return x, y, records


def print_comparison(results):
    """Порівняльна таблиця: кількість ітерацій, розв'язок, нев'язки."""
    print(f"\nЕТАП 6. ПЕРЕВІРКА ТА ПОРІВНЯННЯ МЕТОДІВ  (Степан)")
    if not results:
        print("  Жоден метод ще не реалізовано — порівнювати нічого.")
        return

    line = "-" * 86
    print(line)
    print(f"{'Метод':<22} | {'Ітерацій':>8} | {'x*':>10} | {'y*':>10} | "
          f"{'f1(x*,y*)':>11} | {'f2(x*,y*)':>11}")
    print(line)
    for name, (x, y, records) in results.items():
        r1, r2 = verify(x, y)
        print(f"{name:<22} | {len(records) - 1:>8} | {x:>10.6f} | {y:>10.6f} | "
              f"{r1:>11.2e} | {r2:>11.2e}")
    print(line)


def print_answer(results):
    """Відповідь, округлена до ε, і перевірка округленого розв'язку."""
    if not results:
        return
    # беремо Ньютона як найточніший, якщо він уже готовий
    name = "Ньютона" if "Ньютона" in results else next(iter(results))
    x, y, _ = results[name]
    xr, yr = round(x, 3), round(y, 3)
    r1, r2 = verify(xr, yr)
    ok = max(abs(r1), abs(r2)) < EPS

    print(f"\nВІДПОВІДЬ (метод {name}, ε = 0,001):  x* ≈ {xr:.3f},  y* ≈ {yr:.3f}")
    print(f"Перевірка округленого розв'язку: f1 = {r1:.6f}, f2 = {r2:.6f}  "
          f"(|f| < ε: {'так' if ok else 'ні'})")

    if len(results) > 1:
        counts = ", ".join(f"{n} — {len(r[2]) - 1}" for n, r in results.items())
        print(f"Кількість ітерацій: {counts}.")


def main():
    print(LINE)
    print("Лабораторна робота №3. Розв'язування систем нелінійних рівнянь. Варіант 7")
    print("    x + cos(y) = 1,5")
    print("    2y - sin(x - 0,5) = 1          ε = 0,001")
    print(LINE)

    # Етапи 1–2: локалізація, оцінка q, поріг зупинки (Степан)
    q, threshold = localization.run()

    # Етапи 3–5: методи учасників
    results = {}
    methods = [
        (3, "Метод Ньютона",          "Олександр", newton,           1, {"eps": EPS}),
        (4, "Метод простої ітерації", "Даниїл",    simple_iteration, 2, {"threshold": threshold}),
        (5, "Метод Зейделя",          "Тарас",     seidel,           3, {"threshold": threshold}),
    ]
    for stage, title, author, module, table_no, kwargs in methods:
        res = run_method(stage, title, author, module, table_no, **kwargs)
        if res is not None:
            results[title.replace("Метод ", "").capitalize()] = res

    # Етап 6: перевірка, порівняння, відповідь (Степан)
    print_comparison(results)
    print_answer(results)


if __name__ == "__main__":
    main()
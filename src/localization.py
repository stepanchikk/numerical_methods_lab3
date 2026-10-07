"""
Локалізація розв'язку та аналіз збіжності.

Відповідальний: Степан (тімлід)
Статус: TODO

Що потрібно зробити:
  1. Побудувати графіки рівнянь x + cos(y) = 1,5 і 2y - sin(x - 0,5) = 1
     (matplotlib) і зберегти у results/localization.png.
  2. За графіком визначити прямокутник G, у якому лежить розв'язок,
     і обрати початкове наближення x(0) з G (спільне для всіх методів).
  3. Оцінити q = max ||dΦ/dx|| в області G (по рядках або по стовпцях),
     пояснюючи кожну оцінку (монотонність sin/cos на проміжку тощо).
  4. Переконатися, що q < 1 — достатня умова збіжності МПІ і Зейделя.
  5. Обчислити поріг зупинки для МПІ і Зейделя:
       (1 - q) / q · ε,  якщо q > 0,5
       ε,                 якщо q <= 0,5
  6. Функція run(), яку викликає main.py: виводить результати етапу
     і повертає q та поріг зупинки.
"""

import math
import os


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

from src.common import (X_MIN, X_MAX, Y_MIN, Y_MAX, X0, EPS,
                        phi1, phi2, jacobian_phi)


def plot_localization(path=os.path.join(RESULTS_DIR, "localization.png")):
    """Будує криві f1 = 0, f2 = 0 і прямокутник G. Повертає шлях до файлу або None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [!] matplotlib/numpy не встановлено — графік пропущено "
              "(pip install -r requirements.txt)")
        return None

    os.makedirs(os.path.dirname(path), exist_ok=True)

    y = np.linspace(-1.5, 2.5, 400)
    x_curve1 = 1.5 - np.cos(y)                       # x + cos y = 1,5
    x = np.linspace(-1.5, 3.0, 400)
    y_curve2 = 0.5 + 0.5 * np.sin(x - 0.5)           # 2y - sin(x - 0,5) = 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_curve1, y, color="tab:red", label=r"$x + \cos y = 1{,}5$")
    ax.plot(x, y_curve2, color="tab:green", label=r"$2y - \sin(x - 0{,}5) = 1$")
    ax.add_patch(plt.Rectangle((X_MIN, Y_MIN), X_MAX - X_MIN, Y_MAX - Y_MIN,
                               fill=False, ls="--", color="tab:blue", lw=1.5))
    ax.annotate("G", (X_MAX, Y_MAX), textcoords="offset points", xytext=(4, 4),
                color="tab:blue", fontsize=12)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlim(-1.5, 3.0)
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Рис. 1. Локалізація розв'язку системи (варіант 7)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def estimate_q_analytic():
    """
    Аналітична оцінка q у G = {0,6 < x < 0,8; 0,5 < y < 0,7}.

    dΦ/dx = | 0                 sin(y) |
            | 0,5·cos(x - 0,5)  0      |

    Рядок 1:  |sin y|  — sin зростає на (0; π/2), тому max при y = 0,7:
              q1 = sin(0,7)
    Рядок 2:  0,5·|cos(x - 0,5)|, де x - 0,5 ∈ (0,1; 0,3) — cos спадає,
              тому max при x = 0,6:  q2 = 0,5·cos(0,1)
    Для стовпців (друга норма) отримуємо ті самі два числа, бо в кожному
    рядку і стовпці лише один ненульовий елемент.
    """
    q1 = math.sin(Y_MAX)
    q2 = 0.5 * math.cos(X_MIN - 0.5)
    return q1, q2, max(q1, q2)


def estimate_q_numeric(n=200):
    """Перевірка: максимум сум модулів по рядках на сітці n x n у G."""
    q = 0.0
    for i in range(n + 1):
        x = X_MIN + (X_MAX - X_MIN) * i / n
        for j in range(n + 1):
            y = Y_MIN + (Y_MAX - Y_MIN) * j / n
            J = jacobian_phi(x, y)
            q = max(q, abs(J[0][0]) + abs(J[0][1]), abs(J[1][0]) + abs(J[1][1]))
    return q


def check_phi_maps_G_into_G():
    """
    φ1 залежить лише від y, φ2 — лише від x, обидві монотонні в G,
    тому досить перевірити значення на кінцях проміжків.
    """
    phi1_range = (phi1(0, Y_MIN), phi1(0, Y_MAX))   # 1,5 - cos y зростає
    phi2_range = (phi2(X_MIN, 0), phi2(X_MAX, 0))   # 0,5 + 0,5 sin(x-0,5) зростає
    inside = (X_MIN < phi1_range[0] and phi1_range[1] < X_MAX and
              Y_MIN < phi2_range[0] and phi2_range[1] < Y_MAX)
    return phi1_range, phi2_range, inside


def stop_threshold(q, eps=EPS):
    """Поріг зупинки для МПІ/Зейделя: (1-q)/q·ε, якщо q > 0,5, інакше ε."""
    return (1 - q) / q * eps if q > 0.5 else eps


def run():
    """Виконує весь етап локалізації та повертає q і поріг зупинки."""
    print("ЕТАП 1. ЛОКАЛІЗАЦІЯ РОЗВ'ЯЗКУ")
    path = plot_localization()
    if path:
        print(f"  Графік збережено: {os.path.relpath(path)}")
    print(f"  Область ізоляції: G = ({X_MIN}; {X_MAX}) x ({Y_MIN}; {Y_MAX})")
    print(f"  Початкове наближення: x(0) = {X0}")

    print("\nЕТАП 2. ДОСТАТНЯ УМОВА ЗБІЖНОСТІ МПІ ТА ЗЕЙДЕЛЯ")
    q1, q2, q = estimate_q_analytic()
    print(f"  q1 = max|sin y|          = sin(0,7)     ≈ {q1:.6f}")
    print(f"  q2 = max 0,5|cos(x-0,5)| = 0,5·cos(0,1) ≈ {q2:.6f}")
    print(f"  q  = max(q1, q2)         ≈ {q:.6f} < 1  -> умова збіжності виконується")
    print(f"  Перевірка на сітці 201x201: max ||dΦ/dx|| ≈ {estimate_q_numeric():.6f}")

    r1, r2, inside = check_phi_maps_G_into_G()
    print(f"  φ1(G) ⊂ [{r1[0]:.4f}; {r1[1]:.4f}],  φ2(G) ⊂ [{r2[0]:.4f}; {r2[1]:.4f}]"
          f"  -> Φ(G) ⊂ G: {'так' if inside else 'ні'}")

    thr = stop_threshold(q)
    print(f"  Оскільки q > 0,5, умова зупинки: Δ(k) <= (1-q)/q·ε = {thr:.6f}")
    return q, thr
import math

# Ітераційні функції та їхні частинні похідні

def phi1(x, y):
    return 1.5 - math.cos(y)


def phi2(x, y):
    return (1 + math.sin(x - 0.5)) / 2


def jacobian_phi(x, y):
    return [[0.0, math.sin(y)],
            [math.cos(x - 0.5) / 2, 0.0]]


def estimate_q(x_range, y_range, steps=200):
    q = 0.0
    for i in range(steps + 1):
        x = x_range[0] + (x_range[1] - x_range[0]) * i / steps
        for j in range(steps + 1):
            y = y_range[0] + (y_range[1] - y_range[0]) * j / steps
            J = jacobian_phi(x, y)
            row_norm = max(abs(J[0][0]) + abs(J[0][1]), abs(J[1][0]) + abs(J[1][1]))
            q = max(q, row_norm)
    return q


def stop_tolerance(q, eps):
    return eps if q <= 0.5 else (1 - q) / q * eps


# метод

def solve(x0, tol, max_iter=100):
    """
    Метод простої ітерації.
    x0       - початкове наближення (x, y), спільне для всіх методів;
    tol      - поріг зупинки (з етапу 4);
    max_iter - запобіжник від нескінченного циклу, якщо метод розбігається.

    Повертає (x, y, records), де records - список словників
    з полями k, x, y, dx, dy, delta (для таблиці ітерацій).
    """
    x, y = float(x0[0]), float(x0[1])
    records = [{"k": 0, "x": x, "y": y, "dx": None, "dy": None, "delta": None}]

    for k in range(1, max_iter + 1):
        # обидві координати рахуємо через СТАРІ x, y
        x_new = phi1(x, y)
        y_new = phi2(x, y)

        dx = abs(x_new - x)
        dy = abs(y_new - y)
        delta = max(dx, dy)
        records.append({"k": k, "x": x_new, "y": y_new, "dx": dx, "dy": dy, "delta": delta})

        x, y = x_new, y_new
        if delta <= tol:
            return x, y, records

    raise RuntimeError(f"МПІ не зійшовся за {max_iter} ітерацій")

# друк таблиці ітерацій і нев'язки

def print_table(records, tol):
    print(f"{'k':>3} | {'x1(k)':>10} | {'x2(k)':>10} | {'|dx|':>10} | {'|dy|':>10} | {'delta':>10} | Висновок")
    print("-" * 82)
    for r in records:
        if r["delta"] is None:
            print(f"{r['k']:>3} | {r['x']:>10.6f} | {r['y']:>10.6f} | {'-':>10} | {'-':>10} | {'-':>10} | -")
        else:
            verdict = "<= tol, стоп" if r["delta"] <= tol else "> tol"
            print(f"{r['k']:>3} | {r['x']:>10.6f} | {r['y']:>10.6f} | {r['dx']:>10.6f} | "
                  f"{r['dy']:>10.6f} | {r['delta']:>10.6f} | {verdict}")


def residual(x, y):
    """Нев'язка: підставляємо (x, y) у вихідну систему."""
    r1 = x + math.cos(y) - 1.5
    r2 = 2 * y - math.sin(x - 0.5) - 1
    return max(abs(r1), abs(r2))

if __name__ == "__main__":
    EPS = 1e-3
    G_X, G_Y = (0.6, 0.8), (0.5, 0.7)
    X0 = (0.7, 0.6)

    q = estimate_q(G_X, G_Y)
    tol = stop_tolerance(q, EPS)
    print(f"G = [{G_X[0]}; {G_X[1]}] x [{G_Y[0]}; {G_Y[1]}],  q = {q:.4f} < 1 -> МПІ збігається")
    print(f"Поріг зупинки: delta <= {tol:.6f}\n")

    x, y, records = solve(X0, tol)
    print_table(records, tol)
    print(f"\nРозв'язок: x = {x:.4f}, y = {y:.4f}  (ітерацій: {len(records) - 1})")
    print(f"Перевірка: max нев'язка = {residual(x, y):.2e}")
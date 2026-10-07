import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Загрузка данных (разделитель ';', заголовка нет, BOM в начале)
data = np.genfromtxt("data2.csv", delimiter=";", encoding="utf-8-sig")
x, y = data[:, 0], data[:, 1]
n = x.size
print(f"Точек: {n}, скидка от {x.min()} до {x.max()}")


def fit_poly(idx, label):
    xs, ys = x[idx], y[idx]
    deg = len(idx) - 1
    # 1. СЛУ: матрица
    A = np.vander(xs, N=deg + 1)
    # 2. Решение СЛУ
    coef = linalg.solve(A, ys)
    # 3. Значения полинома во всех точках
    y_fit = np.polyval(coef, x)
    # 5. RSS
    rss = np.sum((y - y_fit) ** 2)
    print(f"\n{label}: точки x = {xs}")
    print("  СЛУ, матрица A:\n", A)
    print("  правая часть:", ys)
    print("  коэффициенты (старший -> a0):", coef)
    print(f"  RSS = {rss:.4f}")
    return coef, y_fit, rss


idx2 = [0, n // 2, n - 1]
idx3 = [0, n // 3, 2 * n // 3, n - 1]

coef2, fit2, rss2 = fit_poly(idx2, "Полином 2-й степени")
coef3, fit3, rss3 = fit_poly(idx3, "Полином 3-й степени")

# 7. Задание по желанию: перебор всех наборов точек
from itertools import combinations

def best_subset(k):
    best = (np.inf, None)
    for idx in combinations(range(n), k):
        A = np.vander(x[list(idx)], N=k)
        try:
            c = linalg.solve(A, y[list(idx)])
        except Exception:
            continue
        r = np.sum((y - np.polyval(c, x)) ** 2)
        if r < best[0]:
            best = (r, idx, c)
    return best

rss2b, idx2b, coef2b = best_subset(3)
rss3b, idx3b, coef3b = best_subset(4)
print("\n7. Лучший подбор точек перебором:")
print(f"  степень 2: x = {x[list(idx2b)]}, RSS = {rss2b:.4f}")
print(f"  степень 3: x = {x[list(idx3b)]}, RSS = {rss3b:.4f}")

# 4. Графики
for fit, idx, deg, rss in ((fit2, idx2, 2, rss2), (fit3, idx3, 3, rss3)):
    plt.figure(figsize=(7, 4.8))
    plt.plot(x, y, "o-", label="данные из файла")
    plt.plot(x, fit, "s--", label=f"полином {deg}-й степени")
    plt.scatter(x[idx], y[idx], s=140, facecolors="none", edgecolors="red",
                label="опорные точки")
    plt.title(f"Степень {deg}, RSS = {rss:.3f}")
    plt.xlabel("Скидка, %")
    plt.ylabel("Прибыль")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graph{deg}.png", dpi=130)
plt.show()

# 8. Лучший вариант (из заданных в шагах 1-6) и прогноз при скидке 6 и 8
if rss2 <= rss3:
    coef, deg = coef2, 2
else:
    coef, deg = coef3, 3
print(f"\n8. Лучший вариант: полином {deg}-й степени")
for s in (6, 8):
    print(f"  прибыль при скидке {s}% = {np.polyval(coef, s):.4f}")

# то же для лучшего подбора из п. 7
cb, db = (coef2b, 2) if rss2b <= rss3b else (coef3b, 3)
print(f"\n  (лучший подбор точек из п.7: степень {db})")
for s in (6, 8):
    print(f"  прибыль при скидке {s}% = {np.polyval(cb, s):.4f}")

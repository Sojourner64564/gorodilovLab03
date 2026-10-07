import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1 Загрузка данных
GEN_FILE = "global-electricity-generation.csv"
CON_FILE = "global-electricity-consumption.csv"

years = np.arange(1992, 2022)

def load(path):
    names = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=0,
                          dtype=str, encoding="utf-8-sig")
    names = np.char.strip(names)
    data = np.genfromtxt(path, delimiter=",", skip_header=1,
                         usecols=range(1, len(years) + 1),
                         dtype=float, encoding="utf-8-sig")
    return names, data

countries, gen = load(GEN_FILE)
countries2, con = load(CON_FILE)
assert np.array_equal(countries, countries2), "порядок стран различается"

print("Стран:", countries.size, "| лет:", years.size)
print("Форма gen:", gen.shape, "| форма con:", con.shape)
print("nan в производстве:", np.isnan(gen).sum(), "| в потреблении:", np.isnan(con).sum())

# 2. Среднегодовое производство/потребление за последние 5 лет
gen_avg5 = np.nanmean(gen[:, -5:], axis=1)
con_avg5 = np.nanmean(con[:, -5:], axis=1)
print("\n2. Среднее за 2017–2021 (первые 5 стран):")
for n, g, c in zip(countries[:5], gen_avg5[:5], con_avg5[:5]):
    print(f"   {n:25s} произв.={g:9.3f}  потр.={c:9.3f}")

# 3 Вопросы
# 3.1 Суммарное потребление по всем странам за каждый год
total_con_by_year = np.nansum(con, axis=0)
print("\n3.1 Суммарное потребление по годам (млрд кВт*ч):")
for y, v in zip(years, total_con_by_year):
    print(f"   {y}: {v:10.2f}")

# 3.2 Максимум, произведённый одной страной за один год
max_gen = np.nanmax(gen)
i, j = np.unravel_index(np.nanargmax(gen), gen.shape)
print(f"\n3.2 Максимальное производство: {max_gen:.2f} млрд кВт*ч "
      f"({countries[i]}, {years[j]})")

# 3.3 Страны с производством > 500 млрд кВт*ч в среднем за последние 5 лет
mask33 = gen_avg5 > 500
print("\n3.3 Страны с производством > 500 в среднем за 5 лет:")
print("   ", countries[mask33])

# 3.4 Топ-10 стран по потреблению (в среднем за последние 5 лет)
q90 = np.nanquantile(con_avg5, 0.9)
mask34 = con_avg5 >= q90
order = np.argsort(-con_avg5[mask34])
print(f"\n3.4 90%-квантиль = {q90:.3f}. Топ-10% по потреблению ({mask34.sum()} стран):")
for n, v in zip(countries[mask34][order], con_avg5[mask34][order]):
    print(f"   {n:25s} {v:10.2f}")

# 3.5 Страны, увеличившие производство в 2021 vs 1992 более чем в 10 раз
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = gen[:, -1] / gen[:, 0]
mask35 = ratio > 10
mask35 &= np.isfinite(ratio)
print("\n3.5 Производство выросло более чем в 10 раз (2021 к 1992):")
for n, r in zip(countries[mask35], ratio[mask35]):
    print(f"   {n:25s} x{r:.1f}")

# 3.6 Суммарное потребление за все годы > 100 и производство < потребления
sum_con = np.nansum(con, axis=1)
sum_gen = np.nansum(gen, axis=1)
mask36 = (sum_con > 100) & (sum_gen < sum_con)
print(f"\n3.6 Потратили в сумме > 100 и произвели меньше, чем потратили ({mask36.sum()} стран):")
print("   ", countries[mask36])

# 3.7 Страна с наибольшим потреблением в 2020 году
k = np.nanargmax(con[:, years == 2020].ravel())
print(f"\n3.7 Наибольшее потребление в 2020: {countries[k]} "
      f"({con[k, years == 2020][0]:.2f} млрд кВт*ч)")

# Графики
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(years, total_con_by_year, marker="o")
    ax[0].set(title="Суммарное мировое потребление", xlabel="Год", ylabel="млрд кВт·ч")
    ax[0].grid(alpha=.3)
    top = np.argsort(-np.nan_to_num(con_avg5))[:10][::-1]
    ax[1].barh(countries[top], con_avg5[top])
    ax[1].set(title="Топ-10 по потреблению (среднее 2017–2021)", xlabel="млрд кВт·ч")
    plt.tight_layout()
    plt.savefig("electricity_plots.png", dpi=130)
except ImportError:
    pass

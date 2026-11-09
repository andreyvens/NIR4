"""Численные эксперименты для проверки расчётного ядра (результаты для отчёта)."""
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import channel_geometry as g

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "font.size": 14.4, "axes.labelsize": 14.4, "legend.fontsize": 14.4,
                     "xtick.labelsize": 14.4, "ytick.labelsize": 14.4})
W = 6.3  # ширина рисунка, дюймы (около 16 см)

# 1. трапеция
c = g.characteristics(g.trapezoid(4, 2.2, 0.75, 1.0))
print("TRAP", {k: round(v, 4) for k, v in c.items()})

# 2. парабола, сходимость по числу точек
h, B = 1.5, 6.0
chi_ex = quad(lambda y: np.sqrt(1 + (4 * h / B * (2 * y / B - 1)) ** 2), 0, B)[0]
om_ex = 2 / 3 * B * h
ns = [5, 9, 17, 33, 65, 129]
rows = []
for n in ns:
    r = g.characteristics(g.parabola(h, B, n))
    rows.append((n, r["omega"], abs(r["omega"] - om_ex) / om_ex * 100,
                 r["chi"], abs(r["chi"] - chi_ex) / chi_ex * 100))
print("PARAB exact", om_ex, chi_ex)
for r in rows:
    print("PARAB", r[0], *(f"{v:.4f}" if i in (1, 3) else f"{v:.4f}" for i, v in enumerate(r[1:], 1)))

fig, ax = plt.subplots(figsize=(W, 3.9))
ax.loglog(ns, [r[2] for r in rows], "ko-", label="площадь")
ax.loglog(ns, [r[4] for r in rows], "ks--", label="смоченный периметр")
ax.set_xlabel("Число точек профиля $n$"); ax.set_ylabel("Погрешность, %")
ax.grid(True, color="0.85"); ax.legend()
fig.tight_layout(); fig.savefig("proto_err.png", dpi=200); plt.close(fig)

# 3. объём
s1, s2 = g.trapezoid(4, 2.0, .75, 1.0), g.trapezoid(8, 3.0, .75, 1.0)
Aex = lambda x: (4 + .04 * x) * (2 + .01 * x) + .875 * (2 + .01 * x) ** 2
Vex = quad(Aex, 0, 100)[0]
print("VOL exact", Vex)
Ks = [1, 2, 4, 8, 16]
vr = []
for K in Ks:
    Vv = g.volume(s1, 0, s2, 100, K, "vertex")
    Va = g.volume(s1, 0, s2, 100, K, "arc")
    vr.append((K, Vv, abs(Vv - Vex) / Vex * 100, Va, abs(Va - Vex) / Vex * 100))
    print("VOL", K, f"{Vv:.2f}", f"{vr[-1][2]:.4f}", f"{Va:.2f}", f"{vr[-1][4]:.4f}")
fig, ax = plt.subplots(figsize=(W, 3.9))
ax.loglog(Ks, [r[2] for r in vr], "ko-", label="по вершинам")
ax.loglog(Ks, [r[4] for r in vr], "ks--", label="по длине дуги")
ax.set_xlabel("Число интервалов между створами $K$"); ax.set_ylabel("Погрешность объёма, %")
ax.grid(True, color="0.85"); ax.legend()
fig.tight_layout(); fig.savefig("proto_vol.png", dpi=200); plt.close(fig)

# 4. примеры контуров
fig, ax = plt.subplots(figsize=(W, 3.9))
t = g.trapezoid(4, 2.2, .75, 1.0); p = g.wet_contour(t)
ax.plot(p[:, 0], p[:, 1], "ko-", label="трапеция")
q = g.parabola(2.2, 6.0, 9); pq = g.wet_contour(q)
ax.plot(pq[:, 0] + 6.5, pq[:, 1], "ks--", label="парабола, $n=9$")
ax.axhline(2.2, color="0.4", ls=":")
ax.set_xlabel("Поперечная координата $y$, м"); ax.set_ylabel("Отметка $z$, м")
ax.set_ylim(-0.1, 3.1); ax.grid(True, color="0.85"); ax.legend(loc="upper center", ncol=2)
fig.tight_layout(); fig.savefig("proto_sections.png", dpi=200); plt.close(fig)

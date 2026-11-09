"""Прототип расчётного ядра системы формирования параметрической геометрии
открытых русел. Реализует только численную часть: построение контура сечения
по параметрам, проверку координатного профиля, расчёт характеристик сечения,
линейную интерполяцию сечений между створами и оценку объёма участка.
Гидродинамические расчёты не выполняются."""
from dataclasses import dataclass
import numpy as np


class GeometryError(ValueError):
    """Ошибка допустимости исходных данных."""


@dataclass
class Section:
    y: np.ndarray      # поперечные координаты, не убывают слева направо
    z: np.ndarray      # высотные отметки
    level: float       # отметка свободной поверхности (локальная система сечения)


def _validate_profile(y, z, level):
    y, z = np.asarray(y, float), np.asarray(z, float)
    if y.ndim != 1 or y.shape != z.shape or len(y) < 3:
        raise GeometryError("нужно не менее трёх точек и массивы одной длины")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(z)) and np.isfinite(level)):
        raise GeometryError("координаты должны быть конечными числами")
    if np.any(np.diff(y) < 0):
        raise GeometryError("поперечная координата должна не убывать")
    if np.any((np.diff(y) == 0) & (np.diff(z) == 0)):
        raise GeometryError("найдены дублирующиеся точки")
    if level <= z.min() or level > min(z[0], z[-1]) + 1e-12:
        raise GeometryError("уровень воды вне допустимого диапазона")
    return y, z


def polyline_section(y, z, level):
    y, z = _validate_profile(y, z, level)
    return Section(y, z, float(level))


def trapezoid(b, h, m1, m2):
    if not (b > 0 and h > 0 and m1 >= 0 and m2 >= 0):
        raise GeometryError("требуется b > 0, h > 0, m1 >= 0, m2 >= 0")
    y = [-m1 * h, 0.0, b, b + m2 * h]
    z = [h, 0.0, 0.0, h]
    return polyline_section(y, z, h)


def rectangle(b, h):
    return trapezoid(b, h, 0.0, 0.0)


def triangle(h, m1, m2):
    if not (h > 0 and m1 >= 0 and m2 >= 0 and m1 + m2 > 0):
        raise GeometryError("требуется h > 0, m1, m2 >= 0, m1 + m2 > 0")
    return polyline_section([-m1 * h, 0.0, m2 * h], [h, 0.0, h], h)


def parabola(h, B, n):
    if not (h > 0 and B > 0 and n >= 3):
        raise GeometryError("требуется h > 0, B > 0, n >= 3")
    y = np.linspace(0.0, B, n)
    return polyline_section(y, h * (2 * y / B - 1) ** 2, h)


def wet_contour(s):
    """Точки смоченной части профиля с учётом пересечения с уровнем воды."""
    pts = []
    for i in range(len(s.y) - 1):
        y1, z1, y2, z2 = s.y[i], s.z[i], s.y[i + 1], s.z[i + 1]
        in1, in2 = z1 <= s.level, z2 <= s.level
        if in1:
            pts.append((y1, z1))
        if in1 != in2:
            t = (s.level - z1) / (z2 - z1)
            pts.append((y1 + t * (y2 - y1), s.level))
    if s.z[-1] <= s.level:
        pts.append((s.y[-1], s.z[-1]))
    return np.array(pts)


def characteristics(s):
    p = wet_contour(s)
    y, z = p[:, 0], p[:, 1]
    chi = float(np.sum(np.hypot(np.diff(y), np.diff(z))))
    yc, zc = np.roll(y, -1), np.roll(z, -1)
    omega = 0.5 * abs(float(np.sum(y * zc - yc * z)))
    B = float(y.max() - y.min())
    return {"B": B, "omega": omega, "chi": chi, "R": omega / chi}


def resample(s, m):
    """Передискретизация профиля до m точек по нормированной длине дуги."""
    p = np.column_stack([s.y, s.z])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0.0, d[-1], m)
    return np.column_stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])])


def interpolate(s1, x1, zb1, s2, x2, zb2, x, mode="auto", m=64):
    """Сечение в створе x между двумя створами; возвращает Section и отметку дна.
    mode="vertex" — соответствие вершин по номеру (при одинаковом числе точек),
    mode="arc" — передискретизация по нормированной длине дуги до m точек,
    mode="auto" — "vertex" при равном числе точек, иначе "arc"."""
    lam = (x - x1) / (x2 - x1)
    if mode == "auto":
        mode = "vertex" if len(s1.y) == len(s2.y) else "arc"
    if mode == "vertex":
        if len(s1.y) != len(s2.y):
            raise GeometryError("для режима vertex число точек должно совпадать")
        p = (1 - lam) * np.column_stack([s1.y, s1.z]) + lam * np.column_stack([s2.y, s2.z])
    else:
        p = (1 - lam) * resample(s1, m) + lam * resample(s2, m)
    level = (1 - lam) * s1.level + lam * s2.level
    return Section(p[:, 0], p[:, 1], level), (1 - lam) * zb1 + lam * zb2


def volume(s1, x1, s2, x2, intervals, mode="auto", m=64):
    """Объём по формуле средних площадей на равномерной сетке створов."""
    xs = np.linspace(x1, x2, intervals + 1)
    areas = [characteristics(interpolate(s1, x1, 0, s2, x2, 0, x, mode, m)[0])["omega"] for x in xs]
    return float(np.sum((np.array(areas[:-1]) + np.array(areas[1:])) / 2 * np.diff(xs)))


def run_checks():
    """Самопроверка: возвращает число пройденных проверок."""
    n = 0
    c = characteristics(trapezoid(4, 2.2, 0.75, 1.0))
    ref = {"B": 4 + 1.75 * 2.2, "omega": 4 * 2.2 + 0.875 * 2.2 ** 2,
           "chi": 4 + 2.2 * 1.25 + 2.2 * 2 ** 0.5}
    for k, v in ref.items():
        assert abs(c[k] - v) < 1e-12; n += 1
    r = characteristics(rectangle(3, 1))
    assert abs(r["omega"] - 3) < 1e-12 and abs(r["chi"] - 5) < 1e-12; n += 2
    t = characteristics(triangle(2, 1, 1))
    assert abs(t["omega"] - 4) < 1e-12 and abs(t["B"] - 4) < 1e-12; n += 2
    bad = [lambda: trapezoid(0, 1, 1, 1), lambda: trapezoid(1, -1, 1, 1),
           lambda: trapezoid(1, 1, -0.1, 1), lambda: triangle(1, 0, 0),
           lambda: parabola(1, 1, 2),
           lambda: polyline_section([0, 2, 1], [1, 0, 1], 1),
           lambda: polyline_section([0, 1, 1, 2], [1, 0, 0, 1], 1),
           lambda: polyline_section([0, 1, float("nan")], [1, 0, 1], 1)]
    for f in bad:
        try:
            f()
        except GeometryError:
            n += 1
        else:
            raise AssertionError("ошибка не обнаружена")
    # уровень ниже бровки: пересечение с откосами
    s = polyline_section([-1.5, 0, 4, 5.5], [3, 0, 0, 3], 3)
    s.level = 1.5
    assert abs(characteristics(s)["B"] - 4 - 1.5) < 1e-12; n += 1
    return n


if __name__ == "__main__":
    print("пройдено проверок:", run_checks())

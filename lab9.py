import sys
import math
import random
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QGridLayout, QMessageBox,
    QFileDialog, QTableWidget, QTableWidgetItem
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


# Биты кода Сазерленда-Коэна (порядок как в методических указаниях):
# 1-й бит - левее окна, 2-й - выше, 3-й - ниже, 4-й - правее.
LEFT = 1
TOP = 2
BOTTOM = 4
RIGHT = 8

EPS = 0.5   # точность алгоритма разбиения средней точкой (в пикселях)

COLOR_WINDOW = [0, 0, 0]
COLOR_SCENE = [90, 90, 90]
COLOR_ORIGINAL = [200, 200, 200]
COLOR_SIMPLE = [255, 0, 0]
COLOR_COHEN = [0, 160, 0]
COLOR_MIDPOINT = [0, 0, 255]


def numpy_to_qpixmap(arr):
    arr = np.ascontiguousarray(np.clip(np.round(arr), 0, 255).astype(np.uint8))
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def draw_line(img, x1, y1, x2, y2, color):
    # целочисленный алгоритм Брезенхема
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    h, w = img.shape[:2]
    while True:
        if 0 <= x1 < w and 0 <= y1 < h:
            img[y1, x1] = color
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def draw_segment(img, p1, p2, color, thick=False):
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    draw_line(img, x1, y1, x2, y2, color)
    if thick:
        draw_line(img, x1 + 1, y1, x2 + 1, y2, color)
        draw_line(img, x1, y1 + 1, x2, y2 + 1, color)


def draw_window(img, win):
    xmin, ymin, xmax, ymax = win
    a = (xmin, ymin)
    b = (xmax, ymin)
    c = (xmax, ymax)
    d = (xmin, ymax)
    for p, q in ((a, b), (b, c), (c, d), (d, a)):
        draw_segment(img, p, q, COLOR_WINDOW)


# ---------------------------------------------------------------- коды концов отрезка
def outcode(x, y, win):
    # 4-битовый код точки относительно окна win = (xmin, ymin, xmax, ymax);
    # (xmin, ymin) - верхний левый угол, (xmax, ymax) - нижний правый
    xmin, ymin, xmax, ymax = win
    code = 0
    if x < xmin:
        code |= LEFT
    if y < ymin:
        code |= TOP
    if y > ymax:
        code |= BOTTOM
    if x > xmax:
        code |= RIGHT
    return code


def code_to_str(code):
    # код в виде строки из 4 бит: левее, выше, ниже, правее
    return "".join("1" if code & bit else "0" for bit in (LEFT, TOP, BOTTOM, RIGHT))


# ---------------------------------------------------------------- простой алгоритм
def clip_simple(p1, p2, win):
    # Простой алгоритм: полная видимость и тривиальная невидимость определяются по
    # кодам концов, затем отрезок пересекается с каждой стороной окна; точка
    # пересечения принимается, если она принадлежит и стороне окна, и отрезку.
    xmin, ymin, xmax, ymax = win
    x1, y1 = p1
    x2, y2 = p2
    c1 = outcode(x1, y1, win)
    c2 = outcode(x2, y2, win)

    if c1 == 0 and c2 == 0:
        return (x1, y1), (x2, y2)
    if c1 & c2:
        return None

    dx = x2 - x1
    dy = y2 - y1
    pts = []   # (параметр t вдоль отрезка, x, y)

    if dx != 0:                      # наклон не бесконечен: левая и правая стороны
        for xs in (xmin, xmax):
            t = (xs - x1) / dx
            if 0 <= t <= 1:
                y = y1 + t * dy
                if ymin <= y <= ymax:
                    pts.append((t, xs, y))
    if dy != 0:                      # наклон не равен нулю: нижняя и верхняя стороны
        for ys in (ymin, ymax):
            t = (ys - y1) / dy
            if 0 <= t <= 1:
                x = x1 + t * dx
                if xmin <= x <= xmax:
                    pts.append((t, x, ys))

    if c1 == 0:                      # конец внутри окна остаётся без изменений
        pts.append((0.0, x1, y1))
    if c2 == 0:
        pts.append((1.0, x2, y2))

    if not pts:
        return None
    pts.sort()
    a = pts[0]
    b = pts[-1]
    return (a[1], a[2]), (b[1], b[2])


# ---------------------------------------------------------------- алгоритм Сазерленда-Коэна
def clip_cohen_sutherland(p1, p2, win):
    # Для каждой стороны окна (левая, правая, нижняя, верхняя): проверяются полная
    # видимость и тривиальная невидимость; если P1 не вне стороны, концы меняются
    # местами; P1 заменяется точкой пересечения отрезка со стороной окна.
    xmin, ymin, xmax, ymax = win
    x1, y1 = p1
    x2, y2 = p2

    for side in (LEFT, RIGHT, BOTTOM, TOP):
        c1 = outcode(x1, y1, win)
        c2 = outcode(x2, y2, win)

        if c1 == 0 and c2 == 0:
            return (x1, y1), (x2, y2)
        if c1 & c2:
            return None
        if (c1 & side) == (c2 & side):
            continue                  # отрезок не пересекает эту сторону

        if not (c1 & side):           # P1 не вне окна: меняем P1 и P2 местами
            x1, y1, x2, y2 = x2, y2, x1, y1

        if side == LEFT:
            y1 = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
            x1 = xmin
        elif side == RIGHT:
            y1 = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
            x1 = xmax
        elif side == BOTTOM:
            x1 = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
            y1 = ymax
        else:
            x1 = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
            y1 = ymin

    if outcode(x1, y1, win) == 0 and outcode(x2, y2, win) == 0:
        return (x1, y1), (x2, y2)
    return None


# ---------------------------------------------------------------- разбиение средней точкой
def nearest_visible(pa, pb, win, eps):
    # Ближайшая к pa видимая точка отрезка pa-pb (или None) с точностью eps.
    ca = outcode(pa[0], pa[1], win)
    cb = outcode(pb[0], pb[1], win)

    if ca == 0:
        return pa                     # pa видима
    if ca & cb:
        return None                   # тривиально невидим

    if cb == 0:                       # pa вне окна, pb внутри: двоичный поиск границы
        lo, hi = pa, pb
        while math.hypot(hi[0] - lo[0], hi[1] - lo[1]) > eps:
            m = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
            if outcode(m[0], m[1], win) == 0:
                hi = m
            else:
                lo = m
        return hi

    if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) <= eps:
        return None                   # отрезок выродился в невидимую точку

    m = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
    res = nearest_visible(pa, m, win, eps)
    if res is not None:               # видимая точка найдена в ближней половине
        return res
    return nearest_visible(m, pb, win, eps)


def clip_midpoint(p1, p2, win, eps=EPS):
    # Алгоритм разбиения средней точкой: два двоичных поиска - ближайшей к P1
    # и ближайшей к P2 видимых точек отрезка.
    a = nearest_visible(p1, p2, win, eps)
    if a is None:
        return None
    b = nearest_visible(p2, p1, win, eps)
    if b is None:
        return None
    return a, b


ALGORITHMS = {
    "simple": ("Простой алгоритм", clip_simple, COLOR_SIMPLE),
    "cohen": ("Алгоритм Сазерленда-Коэна", clip_cohen_sutherland, COLOR_COHEN),
    "midpoint": ("Разбиение средней точкой", clip_midpoint, COLOR_MIDPOINT),
}


def classify(p1, p2, win, result):
    # 0 - полностью видим, 1 - видим частично, 2 - невидим
    if result is None:
        return 2
    if outcode(p1[0], p1[1], win) == 0 and outcode(p2[0], p2[1], win) == 0:
        return 0
    return 1


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЛР №9 — Алгоритмы отсечения отрезков (вариант 5)")
        self.resize(1500, 900)

        self.canvas_size = None      # (ширина, высота) холста
        self.images = {}             # изображения панелей
        self.last_key = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        params = QGridLayout()
        params.setHorizontalSpacing(6)
        params.setVerticalSpacing(8)

        params.addWidget(QLabel("Width:"), 0, 0)
        self.edit_w = QLineEdit("500")
        self.edit_w.setFixedWidth(60)
        params.addWidget(self.edit_w, 0, 1)

        params.addWidget(QLabel("Height:"), 0, 2)
        self.edit_h = QLineEdit("500")
        self.edit_h.setFixedWidth(60)
        params.addWidget(self.edit_h, 0, 3)

        self.btn_create = QPushButton("Создать")
        params.addWidget(self.btn_create, 0, 4)

        params.addWidget(QLabel("Окно: x1:"), 1, 0)
        self.edit_x1 = QLineEdit("150")
        self.edit_x1.setFixedWidth(60)
        params.addWidget(self.edit_x1, 1, 1)

        params.addWidget(QLabel("y1:"), 1, 2)
        self.edit_y1 = QLineEdit("150")
        self.edit_y1.setFixedWidth(60)
        params.addWidget(self.edit_y1, 1, 3)

        params.addWidget(QLabel("x2:"), 2, 0)
        self.edit_x2 = QLineEdit("350")
        self.edit_x2.setFixedWidth(60)
        params.addWidget(self.edit_x2, 2, 1)

        params.addWidget(QLabel("y2:"), 2, 2)
        self.edit_y2 = QLineEdit("350")
        self.edit_y2.setFixedWidth(60)
        params.addWidget(self.edit_y2, 2, 3)

        params.addWidget(QLabel("Отрезков N:"), 3, 0)
        self.edit_n = QLineEdit("7")
        self.edit_n.setFixedWidth(60)
        params.addWidget(self.edit_n, 3, 1)

        self.btn_table = QPushButton("Таблица")
        params.addWidget(self.btn_table, 3, 2)

        self.btn_random = QPushButton("Случайные")
        params.addWidget(self.btn_random, 3, 3)

        self.btn_example = QPushButton("Пример")
        params.addWidget(self.btn_example, 3, 4)

        left.addLayout(params)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["x1", "y1", "x2", "y2"])
        self.table.setMinimumWidth(360)
        self.table.setMaximumWidth(420)
        left.addWidget(self.table, 1)

        algs = QGridLayout()
        self.btn_scene = QPushButton("Сцена")
        algs.addWidget(self.btn_scene, 0, 0)
        self.btn_all = QPushButton("Все алгоритмы")
        algs.addWidget(self.btn_all, 0, 1)
        self.btn_simple = QPushButton("Простой")
        algs.addWidget(self.btn_simple, 1, 0)
        self.btn_cohen = QPushButton("Сазерленд-Коэн")
        algs.addWidget(self.btn_cohen, 1, 1)
        self.btn_mid = QPushButton("Средняя точка")
        algs.addWidget(self.btn_mid, 2, 0)
        self.btn_save = QPushButton("Сохранить")
        algs.addWidget(self.btn_save, 2, 1)
        left.addLayout(algs)

        root.addLayout(left)

        grid = QGridLayout()
        self.lb_scene = self.make_label("Исходная сцена")
        self.lb_simple = self.make_label("Простой алгоритм")
        self.lb_cohen = self.make_label("Алгоритм Сазерленда-Коэна")
        self.lb_mid = self.make_label("Разбиение средней точкой")
        self.labels = {
            "scene": self.lb_scene, "simple": self.lb_simple,
            "cohen": self.lb_cohen, "midpoint": self.lb_mid,
        }
        grid.addWidget(self.lb_scene, 0, 0)
        grid.addWidget(self.lb_simple, 0, 1)
        grid.addWidget(self.lb_cohen, 1, 0)
        grid.addWidget(self.lb_mid, 1, 1)
        root.addLayout(grid, 1)

        self.btn_create.clicked.connect(self.on_create)
        self.btn_table.clicked.connect(self.on_table)
        self.btn_random.clicked.connect(self.on_random)
        self.btn_example.clicked.connect(self.on_example)
        self.btn_scene.clicked.connect(self.on_scene)
        self.btn_all.clicked.connect(self.on_all)
        self.btn_simple.clicked.connect(lambda: self.run_algorithm("simple"))
        self.btn_cohen.clicked.connect(lambda: self.run_algorithm("cohen"))
        self.btn_mid.clicked.connect(lambda: self.run_algorithm("midpoint"))
        self.btn_save.clicked.connect(self.on_save)

    def make_label(self, text):
        lb = QLabel(text)
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lb.setMinimumSize(350, 350)
        lb.setStyleSheet("QLabel { border: 1px solid #888; }")
        return lb

    def show_image(self, key):
        arr = self.images.get(key)
        if arr is None:
            return
        label = self.labels[key]
        pix = numpy_to_qpixmap(arr)
        label.setPixmap(
            pix.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def new_canvas(self):
        w, h = self.canvas_size
        return np.full((h, w, 3), 255, dtype=np.uint8)

    # ---------------------------------------------------------------- ввод данных
    def on_create(self):
        try:
            w = int(self.edit_w.text())
            h = int(self.edit_h.text())
        except ValueError:
            QMessageBox.warning(self, "Ошибка",
                                "Ширина и высота должны быть целыми числами")
            return
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "Ошибка",
                                "Ширина и высота должны быть больше нуля")
            return

        self.canvas_size = (w, h)
        self.images = {"scene": self.new_canvas()}
        self.last_key = None
        for key in ("simple", "cohen", "midpoint"):
            self.labels[key].clear()
            self.labels[key].setText(ALGORITHMS[key][0])
        self.show_image("scene")

    def set_table_size(self, n):
        old = self.table.rowCount()
        self.table.setRowCount(n)
        for r in range(old, n):
            for c in range(4):
                self.table.setItem(r, c, QTableWidgetItem("0"))

    def on_table(self):
        try:
            n = int(self.edit_n.text())
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "N должно быть целым числом")
            return
        if n <= 0 or n > 1000:
            QMessageBox.warning(self, "Ошибка", "N должно быть от 1 до 1000")
            return
        self.set_table_size(n)

    def fill_table(self, segments):
        self.edit_n.setText(str(len(segments)))
        self.table.setRowCount(len(segments))
        for r, (p1, p2) in enumerate(segments):
            for c, v in enumerate((p1[0], p1[1], p2[0], p2[1])):
                self.table.setItem(r, c, QTableWidgetItem(f"{v:g}"))

    def on_random(self):
        if self.canvas_size is None:
            QMessageBox.warning(self, "Внимание", "Сначала создайте холст")
            return
        try:
            n = int(self.edit_n.text())
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "N должно быть целым числом")
            return
        if n <= 0 or n > 1000:
            QMessageBox.warning(self, "Ошибка", "N должно быть от 1 до 1000")
            return

        w, h = self.canvas_size
        segments = []
        for _ in range(n):
            p1 = (random.randint(0, w - 1), random.randint(0, h - 1))
            p2 = (random.randint(0, w - 1), random.randint(0, h - 1))
            segments.append((p1, p2))
        self.fill_table(segments)

    def on_example(self):
        # Набор отрезков для окна (150, 150) - (350, 350): полностью видимый,
        # пересекающие одну и две стороны окна, тривиально невидимые и
        # невидимый, но не отвергаемый по кодам (проходит рядом с углом окна)
        segments = [
            ((200, 250), (300, 200)),
            ((280, 300), (420, 260)),
            ((110, 230), (230, 310)),
            ((80, 300), (400, 100)),
            ((420, 120), (450, 400)),
            ((60, 60), (300, 100)),
            ((100, 180), (180, 100)),
        ]
        self.edit_x1.setText("150")
        self.edit_y1.setText("150")
        self.edit_x2.setText("350")
        self.edit_y2.setText("350")
        self.fill_table(segments)

    def read_scene(self):
        # Возвращает окно (xmin, ymin, xmax, ymax) и список отрезков или None
        if self.canvas_size is None:
            QMessageBox.warning(self, "Внимание", "Сначала создайте холст")
            return None
        try:
            x1 = float(self.edit_x1.text().replace(",", "."))
            y1 = float(self.edit_y1.text().replace(",", "."))
            x2 = float(self.edit_x2.text().replace(",", "."))
            y2 = float(self.edit_y2.text().replace(",", "."))
        except ValueError:
            QMessageBox.warning(self, "Ошибка",
                                "Координаты окна должны быть числами")
            return None
        if x1 >= x2 or y1 >= y2:
            QMessageBox.warning(self, "Ошибка",
                                "Должно быть x1 < x2 и y1 < y2 "
                                "(верхний левый и нижний правый углы окна)")
            return None

        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "Внимание",
                                "Задайте отрезки: кнопки «Таблица», «Случайные» или «Пример»")
            return None

        segments = []
        for r in range(self.table.rowCount()):
            vals = []
            for c in range(4):
                item = self.table.item(r, c)
                try:
                    vals.append(float(item.text().replace(",", ".")))
                except (ValueError, AttributeError):
                    QMessageBox.warning(self, "Ошибка",
                                        f"Некорректное значение в строке {r + 1}, "
                                        f"столбце {c + 1}")
                    return None
            segments.append(((vals[0], vals[1]), (vals[2], vals[3])))
        return (x1, y1, x2, y2), segments

    # ---------------------------------------------------------------- отрисовка
    def draw_scene(self, win, segments):
        img = self.new_canvas()
        draw_window(img, win)
        for p1, p2 in segments:
            draw_segment(img, p1, p2, COLOR_SCENE)
        self.images["scene"] = img
        self.show_image("scene")

    def on_scene(self):
        scene = self.read_scene()
        if scene is None:
            return
        self.draw_scene(*scene)

    def clip_and_draw(self, key, win, segments):
        name, func, color = ALGORITHMS[key]
        img = self.new_canvas()
        draw_window(img, win)
        for p1, p2 in segments:
            draw_segment(img, p1, p2, COLOR_ORIGINAL)

        counts = [0, 0, 0]
        for p1, p2 in segments:
            res = func(p1, p2, win)
            counts[classify(p1, p2, win, res)] += 1
            if res is not None:
                draw_segment(img, res[0], res[1], color, thick=True)

        self.images[key] = img
        self.last_key = key
        self.show_image(key)
        return name, counts

    def run_algorithm(self, key):
        scene = self.read_scene()
        if scene is None:
            return
        win, segments = scene
        self.draw_scene(win, segments)
        name, counts = self.clip_and_draw(key, win, segments)
        QMessageBox.information(
            self, "Готово",
            f"{name}\nПолностью видимых: {counts[0]}\n"
            f"Частично видимых: {counts[1]}\nНевидимых: {counts[2]}"
        )

    def on_all(self):
        scene = self.read_scene()
        if scene is None:
            return
        win, segments = scene
        self.draw_scene(win, segments)
        for key in ("simple", "cohen", "midpoint"):
            self.clip_and_draw(key, win, segments)

    def on_save(self):
        if self.last_key is None:
            QMessageBox.warning(self, "Внимание", "Нечего сохранять")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", f"result_{self.last_key}.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            numpy_to_qpixmap(self.images[self.last_key]).save(path)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

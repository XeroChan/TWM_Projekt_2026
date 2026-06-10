"""
Sterowanie:  →/y akceptuj   ←/n odrzuć   b wstecz   q wyjście
Czerwień = maska. Akceptuj gdy siedzi na dachach.
Zaakceptowane pary są pomijane przy ponownym uruchomieniu (resume).
"""
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

SRC_IMG = "data/processed/images"
SRC_MASK = "data/processed/masks"
DST_IMG = "data/fitted/images"
DST_MASK = "data/fitted/masks"


def overlay(img_rgb, mask):
    """Obraz z maską jako półprzezroczysta czerwień + żółty kontur."""
    out = img_rgb.copy()
    red = np.zeros_like(out)
    red[mask > 127] = [255, 0, 0]
    blended = cv2.addWeighted(out, 0.65, red, 0.35, 0)
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 0), 1)
    return blended


class Reviewer:
    def __init__(self):
        os.makedirs(DST_IMG, exist_ok=True)
        os.makedirs(DST_MASK, exist_ok=True)
        done = set(os.listdir(DST_IMG))
        done.discard(".gitkeep")
        self.accepted = len(done)

        self.files = []
        for f in os.listdir(SRC_IMG):
            if not f.endswith(".png"):
                continue
            if not os.path.exists(os.path.join(SRC_MASK, f)):
                continue
            if f in done:
                continue
            self.files.append(f)
        self.files.sort()
        if not self.files:
            print("Brak par do przeglądu (wszystko już przejrzane?).")
            return
        print(f"Do przeglądu: {len(self.files)} par | już zaakceptowanych: {self.accepted}")
        self.idx = 0
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show()
        plt.show()

    def show(self):
        name = self.files[self.idx]
        img = cv2.imread(os.path.join(SRC_IMG, name))
        mask = cv2.imread(os.path.join(SRC_MASK, name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"Pomijam (nie można wczytać): {name}")
            self.advance()
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cov = (mask > 127).mean() * 100
        self.ax.clear()
        self.ax.imshow(overlay(img, mask))
        is_acc = os.path.exists(os.path.join(DST_IMG, name))
        status = "  [ZAAKCEPTOWANE]" if is_acc else ""
        self.ax.set_title(
            f"[{self.idx+1}/{len(self.files)}] {name}{status}\n"
            f"pokrycie maską: {cov:.1f}%   |   zaakceptowane łącznie: {self.accepted}\n"
            f"→/y = AKCEPTUJ    ←/n = ODRZUĆ    b = wstecz    q = wyjście",
            fontsize=9)
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def accept(self):
        name = self.files[self.idx]
        if not os.path.exists(os.path.join(DST_IMG, name)):
            shutil.copy(os.path.join(SRC_IMG, name), os.path.join(DST_IMG, name))
            shutil.copy(os.path.join(SRC_MASK, name), os.path.join(DST_MASK, name))
            self.accepted += 1
            print(f"  + {name}  (razem: {self.accepted})")

    def unaccept(self, name):
        for d in (DST_IMG, DST_MASK):
            p = os.path.join(d, name)
            if os.path.exists(p):
                os.remove(p)
                if d == DST_IMG:
                    self.accepted -= 1

    def advance(self):
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self.show()
        else:
            print(f"\nKoniec. Zaakceptowanych: {self.accepted}. Folder: {DST_IMG}")
            plt.close(self.fig)

    def on_key(self, event):
        if event.key in ("right", "y", " "):
            self.accept()
            self.advance()
        elif event.key in ("left", "n"):
            self.unaccept(self.files[self.idx])
            self.advance()
        elif event.key == "b":
            if self.idx > 0:
                self.idx -= 1
                self.show()
        elif event.key == "q":
            print(f"\nPrzerwano. Zaakceptowanych: {self.accepted}. Folder: {DST_IMG}")
            plt.close(self.fig)


if __name__ == "__main__":
    Reviewer()

"""Main window: pick a folder, see the brackets, fuse them on a background thread.

Uses the throwaway core/prototype.py for scanning and fusion.

Threading rule: workers never touch widgets. They emit signals, and Qt delivers
those to slots on the main (GUI) thread, which is the only place widgets change.
Preview decoding runs on its own thread, separate from the scan/fuse worker, so
clicking around the list never locks the toolbar or collides with a running job.

Contrast and saturation are post-processing: they run on the fused result, not the
brackets. The fuse_to_image() result is kept in memory for the last
FUSED_CACHE_SIZE brackets, so moving a slider re-runs only enhance_and_save() on the
cached image, on a third thread. A bracket is re-fused only when its cached image is gone
or was fused with different upstream options (half size, align, straighten).

EchoLearn: each fused bracket is classified by SceneSense on the worker thread.
Selecting a classified bracket sets the sliders to echolearn.get_defaults(scene),
without re-running post-processing; releasing a slider logs the value against
the global default. Brackets that aren't classified (not fused yet, or
classification failed) leave the sliders alone and log nothing.
"""

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QProgressBar, QPushButton, QSizePolicy, QSlider,
    QSplitter, QVBoxLayout, QWidget,
)

from core import echolearn
from core.prototype import (
    CLAHE_CLIP, JPEG_EXTS, SATURATION, TIFF_EXTS, brightness, describe_exposure,
    enhance_and_save, exposure_order, find_frames, fuse_to_image, group_brackets,
)
from processing.scenesense import classify_scene

WARNING_COLOR = QColor("#b36b00")
OK_COLOR = QColor("#2e7d32")
FAIL_COLOR = QColor("#c62828")
PREVIEW_LONG_EDGE = 1200
FUSED_CACHE_SIZE = 4        # a 24MP fused RGB image is ~72 MB
SLIDER_SCALE = 100          # QSlider holds ints; value 150 means 1.50
CONTRAST_RANGE = (1.0, 3.0, 0.1)    # min, max, step
SATURATION_RANGE = (1.0, 2.0, 0.05)
SLIDER_SETTLE_MS = 300      # merges bursts of wheel/arrow-key steps into one run


@dataclass(frozen=True)
class FuseSettings:
    """Upstream options. A cached fused image is only valid for the settings it was made with."""
    half_size: bool
    align: bool
    straighten: bool


# --- preview image loading ------------------------------------------------------

def load_before_image(path, half_size=True):
    """The middle exposure rendered the way the camera would.

    Returns (RGB uint8 image, (width, height) of the frame at full resolution).
    The size is reported separately because a RAW is decoded at half_size for
    speed, so the array is half the dimensions the photographer actually shot.

    Deliberately NOT core.prototype.load_image. That one passes no_auto_bright so
    fusion sees the true exposure differences between frames, which leaves a single
    frame looking dark and flat; next to the fused result it would flatter the
    pipeline rather than show it honestly. Here rawpy's defaults are what we want:
    the same ~1% histogram stretch and sRGB gamma (2.222, 4.5) a camera applies to
    its own JPEG. Camera white balance matches what fusion used.
    """
    # TIFF and JPEG are already developed, so they need no rendering decisions:
    # what is in the file is what the camera (or the previous tool) produced.
    if path.suffix.lower() in TIFF_EXTS | JPEG_EXTS:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"could not read {path.name}")
        if img.dtype != np.uint8:  # same rescale as the prototype; these are already sRGB
            img = (img.astype(np.float32) * (255.0 / np.iinfo(img.dtype).max)).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0])
    with rawpy.imread(str(path)) as raw:
        # iwidth/iheight are the full-resolution postprocessed dimensions (verified
        # to match a full decode exactly), so they survive the half_size shortcut.
        full_size = (raw.sizes.iwidth, raw.sizes.iheight)
        # half_size keeps a 24MP decode quick; it still lands well above PREVIEW_LONG_EDGE.
        return raw.postprocess(use_camera_wb=True, output_bps=8, half_size=half_size), full_size


def downscale(img, long_edge=PREVIEW_LONG_EDGE):
    """Shrink to `long_edge` on the long side. Never enlarges."""
    scale = min(1.0, long_edge / max(img.shape[:2]))
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def to_qimage(rgb):
    """RGB uint8 array -> QImage that owns its pixels (the array may be freed after)."""
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def describe_sizes(index, before_size, after_size, half_size):
    """Caption for a before/after pair, in real pixels rather than display pixels.

    Dimensions can differ for two unrelated reasons -- the half-size decode option
    and the perspective crop -- so only name the crop when the output is smaller
    than half-size alone would explain. A couple of pixels of rounding slack keeps
    an odd-numbered sensor dimension from reading as a crop.
    """
    (before_w, before_h), (after_w, after_h) = before_size, after_size
    if (before_w, before_h) == (after_w, after_h):
        return f"Bracket {index + 1}: {before_w}×{before_h}, unchanged"

    expected_w, expected_h = (before_w // 2, before_h // 2) if half_size else (before_w, before_h)
    notes = []
    if half_size:
        notes.append("half size")
    if after_w < expected_w - 2 or after_h < expected_h - 2:
        notes.append("cropped by perspective correction")
    suffix = f" ({', '.join(notes)})" if notes else ""
    return f"Bracket {index + 1}: {before_w}×{before_h} → {after_w}×{after_h}{suffix}"


class ImagePane(QLabel):
    """Shows a QImage scaled to fit whatever size the layout gives it.

    Size policy is Ignored on purpose: a QLabel sized to its own pixmap grows the
    layout, which resizes the label, which rescales the pixmap, so previews creep
    larger on every resize. Ignored breaks that loop.
    """

    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._placeholder = placeholder
        self._image = None

    def set_image(self, image):
        self._image = image
        self._redraw()

    def clear_image(self):
        self._image = None
        self.setText(self._placeholder)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self):
        if self._image is None:
            return
        self.setPixmap(QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class PreviewWorker(QObject):
    """Decodes one before/after pair off the GUI thread."""

    # index, before QImage, after QImage, (before size, after size) in real pixels, error
    ready = Signal(int, object, object, object, str)
    finished = Signal()

    def __init__(self, index, before_path, after_path):
        super().__init__()
        self.index = index
        self.before_path = before_path
        self.after_path = after_path

    @Slot()
    def run(self):
        try:
            before_img, before_size = load_before_image(self.before_path)
            before = to_qimage(downscale(before_img))
            fused = cv2.imread(str(self.after_path), cv2.IMREAD_COLOR)
            if fused is None:
                raise ValueError(f"could not read {self.after_path.name}")
            # Measure the fused file as written, before any preview downscaling.
            after_size = (fused.shape[1], fused.shape[0])
            after = to_qimage(downscale(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)))
            self.ready.emit(self.index, before, after, (before_size, after_size), "")
        except Exception as e:
            self.ready.emit(self.index, None, None, None, str(e))
        finally:
            self.finished.emit()


class ScanWorker(QObject):
    """Finds files and groups them into brackets."""

    finished = Signal(object, object, str)  # brackets, warnings, error message ("" if none)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    @Slot()
    def run(self):
        try:
            frames = find_frames(self.folder)
            if not frames:
                self.finished.emit([], ["no RAW or TIFF files found"], "")
                return
            brackets, warnings = group_brackets(frames)
            self.finished.emit(brackets, warnings, "")
        except Exception as e:
            self.finished.emit([], [], str(e))


class FuseWorker(QObject):
    """Fuses brackets one at a time, reporting progress through signals.

    `indices` limits the run to those brackets (re-fusing one after a cache miss);
    signals always carry the bracket's real index.
    """

    bracket_started = Signal(int)
    # index, fused RGB before post-processing, FuseSettings it was made with
    fused_ready = Signal(int, object, object)
    # index, success, output name or error, middle-exposure path, output path
    bracket_done = Signal(int, bool, str, object, object)
    scene_ready = Signal(int, str)     # index, SceneSense label
    finished = Signal(int, int, bool)  # fused, failed, cancelled

    def __init__(self, brackets, out_dir, settings, clahe_clip, saturation, indices=None):
        super().__init__()
        self.brackets = brackets
        self.out_dir = out_dir
        self.settings = settings
        self.clahe_clip = clahe_clip
        self.saturation = saturation
        self.indices = list(range(len(brackets))) if indices is None else list(indices)
        self._cancelled = False

    def cancel(self):
        # Called directly from the GUI thread, not through a signal: this worker's
        # thread is busy inside run() and would not process a queued call until it ended.
        self._cancelled = True

    @Slot()
    def run(self):
        fused = failed = 0
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            for i in self.indices:
                self.bracket_done.emit(i, False, str(e), None, None)
            self.finished.emit(0, len(self.indices), False)
            return

        for i in self.indices:
            bracket = self.brackets[i]
            if self._cancelled:
                break
            # Same ordering as prototype.main(): dark to bright when EXIF allows it.
            if all(brightness(f) is not None for f in bracket):
                bracket = sorted(bracket, key=exposure_order)
            # Only here is the bracket in exposure order, so only here do we know
            # which frame is the normal exposure. Tell the UI rather than let it guess.
            middle_path = bracket[len(bracket) // 2]["path"]
            out_path = self.out_dir / f"{bracket[0]['path'].stem}_fused.jpg"
            self.bracket_started.emit(i)
            try:
                image, _note = fuse_to_image(bracket, self.settings.half_size, self.settings.align,
                                             self.settings.straighten)
                # The array is handed over to the GUI thread's cache and never modified
                # again here: enhance_and_save() leaves its input untouched.
                self.fused_ready.emit(i, image, self.settings)
                enhance_and_save(image, out_path, self.clahe_clip, self.saturation)
            except Exception as e:
                failed += 1
                self.bracket_done.emit(i, False, str(e), None, None)
            else:
                try:
                    # SceneSense expects BGR; fuse_to_image returns RGB.
                    label, _conf = classify_scene(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), source=out_path.name)
                    self.scene_ready.emit(i, label)
                except Exception:
                    pass  # unclassified: EchoLearn just skips this bracket
                fused += 1
                self.bracket_done.emit(i, True, out_path.name, middle_path, out_path)
        self.finished.emit(fused, failed, self._cancelled)


class PostWorker(QObject):
    """Re-runs only enhance_and_save() on a cached fused image, then builds the After preview."""

    done = Signal(int, object, float, str)  # index, after QImage, seconds taken, error ("" if none)
    finished = Signal()

    def __init__(self, index, image, clahe_clip, saturation, out_path):
        super().__init__()
        self.index = index
        self.image = image
        self.clahe_clip = clahe_clip
        self.saturation = saturation
        self.out_path = out_path

    @Slot()
    def run(self):
        start = time.perf_counter()
        try:
            out = enhance_and_save(self.image, self.out_path, self.clahe_clip, self.saturation)
            # Build the preview from the array we already have instead of re-reading the JPEG.
            after = to_qimage(downscale(out))
            self.done.emit(self.index, after, time.perf_counter() - start, "")
        except Exception as e:
            self.done.emit(self.index, None, 0.0, str(e))
        finally:
            self.finished.emit()


def make_slider(value_range, value):
    """Horizontal QSlider over `value_range` (min, max, step), in SLIDER_SCALE units."""
    minimum, maximum, step = (round(v * SLIDER_SCALE) for v in value_range)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setSingleStep(step)
    slider.setPageStep(step * 5)
    slider.setValue(round(value * SLIDER_SCALE))
    slider.setFixedWidth(140)
    return slider


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EstateLens")
        self.resize(1100, 700)

        self.input_dir = None
        self.output_dir = None
        self.brackets = []
        self.bracket_items = []  # header QListWidgetItem per bracket
        self.results = {}        # bracket index -> (middle exposure path, fused path)
        self._thread = None
        self._worker = None
        self._busy = False
        self._close_pending = False

        # Previews get their own thread so decoding a RAW never locks the toolbar.
        self._preview_cache = {}     # bracket index -> (before QImage, after QImage)
        self._preview_thread = None
        self._preview_worker = None
        self._preview_wanted = None  # index the user is currently asking to see
        self._preview_pending = None # queued while a decode is in flight
        self._results_half_size = {}     # bracket index -> half-size option its output was fused with
        self._scene_types = {}           # bracket index -> SceneSense label

        # Fused images before post-processing, most recently used last.
        self._fused_cache = OrderedDict()  # bracket index -> (FuseSettings, RGB uint8)
        self._post_thread = None
        self._post_worker = None
        self._post_pending = None    # bracket to post-process again once the running pass ends
        self._post_timer = QTimer(self)
        self._post_timer.setSingleShot(True)
        self._post_timer.setInterval(SLIDER_SETTLE_MS)
        self._post_timer.timeout.connect(self.reprocess_selected)

        self.input_label = QLabel("No folder chosen")
        self.output_label = QLabel("-")
        for label in (self.input_label, self.output_label):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.input_button = QPushButton("Choose input…")
        self.output_button = QPushButton("Choose output…")
        self.output_button.setEnabled(False)

        self.list = QListWidget()
        self.half_size = QCheckBox("Half size (faster, less memory)")
        self.align = QCheckBox("Align frames")
        self.align.setChecked(True)
        self.straighten = QCheckBox("Straighten verticals")
        self.straighten.setChecked(True)
        self.contrast = make_slider(CONTRAST_RANGE, CLAHE_CLIP)
        self.saturation = make_slider(SATURATION_RANGE, SATURATION)
        self.contrast_value = QLabel()
        self.saturation_value = QLabel()
        for label in (self.contrast_value, self.saturation_value):
            label.setMinimumWidth(32)
        self.update_slider_labels()
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m brackets")
        self.progress.setValue(0)
        self.status = QLabel("Choose a folder of RAW files.")

        self.before_pane = ImagePane("No preview")
        self.after_pane = ImagePane("No preview")
        self.before_caption = QLabel("Before")
        self.after_caption = QLabel("After")
        self.preview_status = QLabel("Fuse a bracket, or click one that is already done.")
        self.preview_status.setWordWrap(True)
        for caption in (self.before_caption, self.after_caption):
            caption.setAlignment(Qt.AlignCenter)

        rows = QVBoxLayout()
        for label_text, label, button in (("Input:", self.input_label, self.input_button),
                                          ("Output:", self.output_label, self.output_button)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addWidget(label, 1)
            row.addWidget(button)
            rows.addLayout(row)
        panes = QHBoxLayout()
        for caption, pane in ((self.before_caption, self.before_pane),
                              (self.after_caption, self.after_pane)):
            column = QVBoxLayout()
            column.addWidget(caption)
            column.addWidget(pane, 1)
            panes.addLayout(column, 1)
        preview_rows = QVBoxLayout()
        preview_rows.addLayout(panes, 1)
        preview_rows.addWidget(self.preview_status)
        preview_panel = QWidget()
        preview_panel.setLayout(preview_rows)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.list)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 780])
        rows.addWidget(splitter, 1)
        options = QHBoxLayout()
        options.addWidget(self.half_size)
        options.addWidget(self.align)
        options.addWidget(self.straighten)
        options.addSpacing(16)
        for text, slider, value in (("Contrast", self.contrast, self.contrast_value),
                                    ("Saturation", self.saturation, self.saturation_value)):
            options.addWidget(QLabel(text))
            options.addWidget(slider)
            options.addWidget(value)
        options.addStretch(1)
        options.addWidget(self.process_button)
        rows.addLayout(options)
        rows.addWidget(self.progress)
        rows.addWidget(self.status)
        central = QWidget()
        central.setLayout(rows)
        self.setCentralWidget(central)

        self.input_button.clicked.connect(self.choose_input)
        self.output_button.clicked.connect(self.choose_output)
        self.process_button.clicked.connect(self.process_or_cancel)
        self.list.currentItemChanged.connect(self.on_list_selection)
        for slider, value_range in ((self.contrast, CONTRAST_RANGE), (self.saturation, SATURATION_RANGE)):
            slider.valueChanged.connect(
                lambda value, s=slider, step=round(value_range[2] * SLIDER_SCALE): self.on_slider_changed(s, value, step))
            slider.sliderReleased.connect(self._post_timer.start)
            slider.sliderReleased.connect(lambda s=slider: self.on_slider_released(s))

    # --- folder selection and scanning ---------------------------------------

    def choose_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder of RAW files")
        if folder:
            self.set_input(Path(folder))

    def set_input(self, folder):
        self.input_dir = folder
        self.input_label.setText(str(folder))
        self.output_dir = folder / "fused"
        self.output_label.setText(str(self.output_dir))
        self.list.clear()
        self.brackets = []
        self.clear_previews()
        self.status.setText("Scanning…")
        self.start_worker(ScanWorker(folder))

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if folder:
            self.output_dir = Path(folder)
            self.output_label.setText(folder)

    @Slot(object, object, str)
    def on_scan_finished(self, brackets, warnings, error):
        self.brackets = brackets
        self.bracket_items = []
        self.list.clear()
        if error:
            self.add_item(f"Error: {error}", FAIL_COLOR)
        for w in warnings:
            self.add_item(f"⚠ {w}", WARNING_COLOR)
        for i, bracket in enumerate(brackets, 1):
            exposures = ", ".join(describe_exposure(f, bracket) for f in bracket)
            # Tag the header and its file rows with the bracket index, so clicking
            # either one selects the same preview. Warning rows stay untagged.
            self.bracket_items.append(self.add_item(f"Bracket {i}  ({exposures})", index=i - 1))
            for f in bracket:
                self.add_item(f"      {f['path'].name}", index=i - 1)
        self.progress.setMaximum(max(len(brackets), 1))
        self.progress.setValue(0)
        self.status.setText(f"{len(brackets)} bracket(s) ready." if brackets else "Nothing to process.")

    def add_item(self, text, color=None, index=None):
        item = QListWidgetItem(text)
        if color is not None:
            item.setForeground(color)
        if index is not None:
            item.setData(Qt.UserRole, index)
        self.list.addItem(item)
        return item

    # --- processing ------------------------------------------------------------

    def process_or_cancel(self):
        if self._busy:
            self._worker.cancel()
            self.process_button.setEnabled(False)
            self.status.setText("Cancelling after the current bracket…")
            return
        for i, item in enumerate(self.bracket_items):
            item.setText(item.text().split("  —")[0])
            item.setForeground(self.list.palette().text().color())
        self.clear_previews()  # these outputs are about to be overwritten
        self.progress.setMaximum(len(self.brackets))
        self.progress.setValue(0)
        self.start_worker(FuseWorker(self.brackets, self.output_dir, self.fuse_settings(),
                                     *self.post_settings()))

    def fuse_settings(self):
        return FuseSettings(self.half_size.isChecked(), self.align.isChecked(), self.straighten.isChecked())

    def post_settings(self):
        """(clahe_clip, saturation) from the sliders."""
        return self.contrast.value() / SLIDER_SCALE, self.saturation.value() / SLIDER_SCALE

    @Slot(int)
    def on_bracket_started(self, i):
        self.status.setText(f"Fusing bracket {i + 1} of {len(self.brackets)}…")
        item = self.bracket_items[i]
        item.setText(f"{item.text().split('  —')[0]}  — working…")

    @Slot(int, object, object)
    def on_fused_ready(self, i, image, settings):
        self._fused_cache[i] = (settings, image)
        self._fused_cache.move_to_end(i)
        while len(self._fused_cache) > FUSED_CACHE_SIZE:
            self._fused_cache.popitem(last=False)
        self._results_half_size[i] = settings.half_size

    @Slot(int, bool, str, object, object)
    def on_bracket_done(self, i, ok, message, before_path, after_path):
        item = self.bracket_items[i]
        base = item.text().split("  —")[0]
        item.setText(f"{base}  — ✓ {message}" if ok else f"{base}  — ✗ failed: {message}")
        item.setForeground(OK_COLOR if ok else FAIL_COLOR)
        self.progress.setValue(self.progress.value() + 1)
        self._preview_cache.pop(i, None)  # the file on disk was just rewritten
        if ok:
            self.results[i] = (before_path, after_path)
            self.show_preview(i)
        else:
            self._fused_cache.pop(i, None)

    # --- contrast / saturation -----------------------------------------------------

    def update_slider_labels(self):
        clahe_clip, saturation = self.post_settings()
        self.contrast_value.setText(f"{clahe_clip:.1f}")
        self.saturation_value.setText(f"{saturation:.2f}")

    def on_slider_changed(self, slider, value, step):
        # A mouse drag lands on any integer; snap to the step. setValue re-enters this
        # method with the snapped value, which then does the real work.
        snapped = round(value / step) * step
        if snapped != value:
            slider.setValue(snapped)
            return
        self.update_slider_labels()  # cheap, so live while dragging
        # While dragging, wait for sliderReleased. Keyboard, wheel and track clicks
        # never press the handle, so they only arrive here; the timer merges bursts.
        if not slider.isSliderDown():
            self._post_timer.start()

    # --- EchoLearn ------------------------------------------------------------------

    def learned_sliders(self):
        """(slider, value range, echolearn param name) for each learnable slider."""
        return ((self.contrast, CONTRAST_RANGE, "clahe_clip"),
                (self.saturation, SATURATION_RANGE, "saturation"))

    @Slot(int, str)
    def on_scene_ready(self, index, label):
        self._scene_types[index] = label

    def on_slider_released(self, slider):
        index = self.selected_index()
        scene = self._scene_types.get(index)
        if scene is None:
            return
        for s, _range, param in self.learned_sliders():
            if s is slider:
                # Measured from the global default, the same base get_defaults adds to.
                echolearn.log_adjustment(scene, param, echolearn.GLOBAL_DEFAULTS[param],
                                         slider.value() / SLIDER_SCALE)

    def apply_learned_defaults(self, index):
        """Set the sliders to EchoLearn's defaults for bracket `index`'s scene, clamped and snapped."""
        scene = self._scene_types.get(index)
        if scene is None:
            return
        defaults = echolearn.get_defaults(scene)
        for slider, (low, high, step), param in self.learned_sliders():
            value = min(max(defaults[param], low), high)
            value = low + round((value - low) / step) * step
            # Blocked so selecting a bracket doesn't rewrite its JPEG or trigger a re-fuse.
            slider.blockSignals(True)
            slider.setValue(round(value * SLIDER_SCALE))
            slider.blockSignals(False)
        self.update_slider_labels()

    def selected_index(self):
        item = self.list.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    @Slot()
    def reprocess_selected(self):
        index = self.selected_index()
        if index is None or index not in self.results:
            self.status.setText("Contrast and saturation will apply to the next Process.")
            return
        self.reprocess(index)

    def reprocess(self, index):
        """Apply the sliders to bracket `index`: post-process only if its cached image is still valid."""
        if self._busy:
            return  # a fuse is running; it already reads the sliders for brackets it hasn't written
        if self._post_thread is not None:
            # Don't start a second writer on the same JPEG; run once more with the latest values.
            self._post_pending = index
            return
        cached = self._fused_cache.get(index)
        if cached is None or cached[0] != self.fuse_settings():
            reason = "not in memory" if cached is None else "options changed since it was fused"
            self.status.setText(f"Re-fusing bracket {index + 1} ({reason})…")
            self.progress.setMaximum(1)
            self.progress.setValue(0)
            self.start_worker(FuseWorker(self.brackets, self.output_dir, self.fuse_settings(),
                                         *self.post_settings(), indices=[index]))
            return
        self._fused_cache.move_to_end(index)
        clahe_clip, saturation = self.post_settings()
        worker = PostWorker(index, cached[1], clahe_clip, saturation, self.results[index][1])
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self.on_post_done)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.on_post_thread_finished)
        self._post_thread, self._post_worker = thread, worker
        self.process_button.setEnabled(False)
        self.status.setText(f"Applying contrast {clahe_clip:.1f}, saturation {saturation:.2f} "
                            f"to bracket {index + 1}…")
        thread.start()

    @Slot(int, object, float, str)
    def on_post_done(self, index, after, seconds, error):
        if error:
            self.status.setText(f"Contrast/saturation failed on bracket {index + 1}: {error}")
            return
        if index in self._preview_cache:
            before, _old_after, sizes = self._preview_cache[index]
            self._preview_cache[index] = (before, after, sizes)  # same pixels size, so sizes still hold
        if index == self._preview_wanted:
            self.show_preview(index)
        self.status.setText(f"Bracket {index + 1}: contrast and saturation applied in {seconds:.2f} s.")

    @Slot()
    def on_post_thread_finished(self):
        self._post_thread = self._post_worker = None
        self.set_busy(self._busy)
        if self._close_pending:
            self.close()
            return
        pending, self._post_pending = self._post_pending, None
        if pending is not None:
            self.reprocess(pending)

    # --- before/after preview --------------------------------------------------

    @Slot()
    def on_list_selection(self):
        item = self.list.currentItem()
        if item is None:
            return
        index = item.data(Qt.UserRole)
        if index is not None:
            self.apply_learned_defaults(index)
            self.show_preview(index)

    def show_preview(self, index):
        """Display bracket `index`, decoding it first if it is not cached yet."""
        self._preview_wanted = index
        if index not in self.results:
            self.before_pane.clear_image()
            self.after_pane.clear_image()
            self.preview_status.setText(f"Bracket {index + 1} has not been fused yet.")
            return
        if index in self._preview_cache:
            self.display_preview(index, *self._preview_cache[index])
            return
        self.preview_status.setText(f"Loading preview for bracket {index + 1}…")
        if self._preview_thread is not None:
            # A decode is already running and cannot be interrupted; queue this one.
            self._preview_pending = index
            return
        self.start_preview(index)

    def start_preview(self, index):
        before_path, after_path = self.results[index]
        worker = PreviewWorker(index, before_path, after_path)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(self.on_preview_ready)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.on_preview_thread_finished)
        self._preview_thread, self._preview_worker = thread, worker
        thread.start()

    @Slot(int, object, object, object, str)
    def on_preview_ready(self, index, before, after, sizes, error):
        if error:
            if index == self._preview_wanted:
                self.before_pane.clear_image()
                self.after_pane.clear_image()
                self.preview_status.setText(f"Preview failed: {error}")
            return
        self._preview_cache[index] = (before, after, sizes)
        # A slow decode may land after the user clicked elsewhere; cache it either
        # way, but only paint it if it is still the bracket they are looking at.
        if index == self._preview_wanted:
            self.display_preview(index, before, after, sizes)

    def display_preview(self, index, before, after, sizes):
        before_path, after_path = self.results[index]
        self.before_pane.set_image(before)
        self.after_pane.set_image(after)
        self.before_caption.setText(f"Before — {before_path.name} (middle exposure)")
        self.after_caption.setText(f"After — {after_path.name}")
        # Real file dimensions, not the scaled-to-fit preview the panes are showing.
        self.preview_status.setText(describe_sizes(index, *sizes, self._results_half_size.get(index, False)))

    def clear_previews(self):
        self.results = {}
        self._preview_cache = {}
        self._fused_cache = OrderedDict()
        self._preview_wanted = self._preview_pending = None
        self._post_pending = None
        self._results_half_size = {}
        self._scene_types = {}
        self.before_pane.clear_image()
        self.after_pane.clear_image()
        self.before_caption.setText("Before")
        self.after_caption.setText("After")
        self.preview_status.setText("Fuse a bracket, or click one that is already done.")

    @Slot()
    def on_preview_thread_finished(self):
        self._preview_thread = self._preview_worker = None
        pending, self._preview_pending = self._preview_pending, None
        if pending is not None and pending not in self._preview_cache:
            self.start_preview(pending)
        elif pending is not None:
            self.show_preview(pending)

    @Slot(int, int, bool)
    def on_fuse_finished(self, fused, failed, cancelled):
        summary = f"{fused} fused, {failed} failed"
        self.status.setText(f"Cancelled: {summary}." if cancelled else f"Done: {summary}. Output in {self.output_dir}")

    # --- thread management -----------------------------------------------------

    def start_worker(self, worker):
        """Run a worker on a fresh QThread; the UI stays locked until it finishes."""
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.on_thread_finished)

        if isinstance(worker, ScanWorker):
            worker.finished.connect(self.on_scan_finished)
        else:
            worker.bracket_started.connect(self.on_bracket_started)
            worker.fused_ready.connect(self.on_fused_ready)
            worker.bracket_done.connect(self.on_bracket_done)
            worker.scene_ready.connect(self.on_scene_ready)
            worker.finished.connect(self.on_fuse_finished)
        worker.finished.connect(self.on_worker_finished)

        # Keep Python references, or the garbage collector deletes them mid-run.
        self._thread, self._worker = thread, worker
        self.set_busy(True, processing=isinstance(worker, FuseWorker))
        thread.start()

    @Slot()
    def on_worker_finished(self):
        self.set_busy(False)
        if self._close_pending:
            self.close()

    @Slot()
    def on_thread_finished(self):
        self._thread = self._worker = None

    def set_busy(self, busy, processing=False):
        self._busy = busy
        self.input_button.setEnabled(not busy)
        self.output_button.setEnabled(not busy and self.input_dir is not None)
        self.half_size.setEnabled(not busy)
        self.align.setEnabled(not busy)
        self.straighten.setEnabled(not busy)
        self.contrast.setEnabled(not busy)
        self.saturation.setEnabled(not busy)
        self.process_button.setText("Cancel" if processing else "Process")
        # A running post-processing pass is writing a JPEG that Process would overwrite.
        self.process_button.setEnabled(
            processing or (not busy and bool(self.brackets) and self._post_thread is None))

    def closeEvent(self, event):
        self._post_timer.stop()
        self._post_pending = None
        if self._post_thread is not None:
            # Let the JPEG finish writing rather than leave a truncated file; closes when done.
            self._close_pending = True
            event.ignore()
            return
        if self._busy:
            # Closing now would destroy a running QThread and crash. Cancel, and
            # close automatically once the current bracket finishes.
            if self._worker is not None and hasattr(self._worker, "cancel"):
                self._worker.cancel()
            self._close_pending = True
            self.status.setText("Closing after the current step finishes…")
            event.ignore()
            return
        if self._thread is not None:  # worker done, thread still winding down
            self._thread.quit()
            self._thread.wait()
        if self._preview_thread is not None:
            # quit() only ends the event loop; wait() is what blocks until a decode
            # in progress returns. Destroying a running QThread would crash.
            self._preview_pending = None
            self._preview_thread.quit()
            self._preview_thread.wait()
        event.accept()


def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())

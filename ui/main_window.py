"""Main window: pick a folder, see the brackets, fuse them on a background thread.

Uses the throwaway core/prototype.py for scanning and fusion.

Threading rule: workers never touch widgets. They emit signals, and Qt delivers
those to slots on the main (GUI) thread, which is the only place widgets change.
"""

import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from core.prototype import brightness, describe_exposure, find_frames, fuse_bracket, group_brackets

WARNING_COLOR = QColor("#b36b00")
OK_COLOR = QColor("#2e7d32")
FAIL_COLOR = QColor("#c62828")


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
    """Fuses brackets one at a time, reporting progress through signals."""

    bracket_started = Signal(int)
    bracket_done = Signal(int, bool, str)  # index, success, output name or error
    finished = Signal(int, int, bool)  # fused, failed, cancelled

    def __init__(self, brackets, out_dir, half_size, do_align):
        super().__init__()
        self.brackets = brackets
        self.out_dir = out_dir
        self.half_size = half_size
        self.do_align = do_align
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
            for i in range(len(self.brackets)):
                self.bracket_done.emit(i, False, str(e))
            self.finished.emit(0, len(self.brackets), False)
            return

        for i, bracket in enumerate(self.brackets):
            if self._cancelled:
                break
            # Same ordering as prototype.main(): dark to bright when EXIF allows it.
            if all(brightness(f) is not None for f in bracket):
                bracket = sorted(bracket, key=brightness)
            out_path = self.out_dir / f"{bracket[0]['path'].stem}_fused.jpg"
            self.bracket_started.emit(i)
            try:
                fuse_bracket(bracket, out_path, self.half_size, self.do_align)
            except Exception as e:
                failed += 1
                self.bracket_done.emit(i, False, str(e))
            else:
                fused += 1
                self.bracket_done.emit(i, True, out_path.name)
        self.finished.emit(fused, failed, self._cancelled)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EstateLens")
        self.resize(760, 520)

        self.input_dir = None
        self.output_dir = None
        self.brackets = []
        self.bracket_items = []  # header QListWidgetItem per bracket
        self._thread = None
        self._worker = None
        self._busy = False
        self._close_pending = False

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
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m brackets")
        self.progress.setValue(0)
        self.status = QLabel("Choose a folder of RAW files.")

        rows = QVBoxLayout()
        for label_text, label, button in (("Input:", self.input_label, self.input_button),
                                          ("Output:", self.output_label, self.output_button)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addWidget(label, 1)
            row.addWidget(button)
            rows.addLayout(row)
        rows.addWidget(self.list, 1)
        options = QHBoxLayout()
        options.addWidget(self.half_size)
        options.addWidget(self.align)
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
            exposures = " ".join(describe_exposure(f) for f in bracket)
            self.bracket_items.append(self.add_item(f"Bracket {i}  ({exposures})"))
            for f in bracket:
                self.add_item(f"      {f['path'].name}")
        self.progress.setMaximum(max(len(brackets), 1))
        self.progress.setValue(0)
        self.status.setText(f"{len(brackets)} bracket(s) ready." if brackets else "Nothing to process.")

    def add_item(self, text, color=None):
        item = QListWidgetItem(text)
        if color is not None:
            item.setForeground(color)
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
        self.progress.setMaximum(len(self.brackets))
        self.progress.setValue(0)
        self.start_worker(FuseWorker(self.brackets, self.output_dir,
                                     self.half_size.isChecked(), self.align.isChecked()))

    @Slot(int)
    def on_bracket_started(self, i):
        self.status.setText(f"Fusing bracket {i + 1} of {len(self.brackets)}…")
        self.bracket_items[i].setText(f"{self.bracket_items[i].text()}  — working…")

    @Slot(int, bool, str)
    def on_bracket_done(self, i, ok, message):
        item = self.bracket_items[i]
        base = item.text().split("  —")[0]
        item.setText(f"{base}  — ✓ {message}" if ok else f"{base}  — ✗ failed: {message}")
        item.setForeground(OK_COLOR if ok else FAIL_COLOR)
        self.progress.setValue(self.progress.value() + 1)

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
            worker.bracket_done.connect(self.on_bracket_done)
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
        self.process_button.setText("Cancel" if processing else "Process")
        self.process_button.setEnabled(processing or (not busy and bool(self.brackets)))

    def closeEvent(self, event):
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
        event.accept()


def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())

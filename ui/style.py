"""
ui/style.py — EstateLens visual theme.

Contains ONLY a Qt stylesheet (QSS) string and a function to apply it.
Zero processing logic lives here. Nothing in this file touches
ChromaRaw, LumaMerge, TrueVertical, AeroSwap, SceneSense, ToneSync,
EchoLearn, IronCore, or DataShield.

Usage (already wired into main_window.py):
    from ui.style import apply_theme
    app = QApplication(sys.argv)
    apply_theme(app)
"""

_QSS = """
QMainWindow, QWidget {
    background-color: #1c1e26;
    color: #e4e6eb;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

/* ---------- Header bar ---------- */
#AppHeader {
    background-color: #14151b;
    border-bottom: 1px solid #2a2d3a;
    min-height: 46px;
    max-height: 46px;
}

#AppTitleLabel {
    color: #ffffff;
    font-size: 16px;
    font-weight: 600;
}

/* ---------- Bracket list ---------- */
QListWidget#BracketList {
    background-color: #20222c;
    border: none;
    outline: none;
    padding: 4px;
}

QListWidget#BracketList::item {
    border-radius: 4px;
    padding: 4px 6px;
}

QListWidget#BracketList::item:selected {
    background-color: #2f3346;
}

QListWidget#BracketList::item:hover:!selected {
    background-color: #262834;
}

/* ---------- Preview panel ---------- */
#PreviewPanel {
    background-color: #1c1e26;
}

QLabel#PreviewImageLabel {
    background-color: #0f1015;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
}

QLabel#PaneCaption {
    color: #c4c7d1;
    font-weight: 600;
    font-size: 12px;
    padding: 4px 0px;
}

QLabel#PreviewCaptionLabel {
    color: #8b8fa3;
    font-size: 11px;
    padding: 4px 0px;
}

/* ---------- Status label (bottom) ---------- */
QLabel#StatusLabel {
    color: #8b8fa3;
    font-size: 11px;
    padding-top: 4px;
}

/* ---------- Splitter ---------- */
QSplitter#MainSplitter::handle {
    background-color: #2a2d3a;
    width: 2px;
}

/* ---------- Buttons ---------- */
QPushButton {
    background-color: #2a2d3a;
    color: #e4e6eb;
    border: none;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #353849;
}

QPushButton:pressed {
    background-color: #40445a;
}

QPushButton:disabled {
    background-color: #21232d;
    color: #4a4d5c;
}

QPushButton#PrimaryButton {
    background-color: #5865f2;
    color: #ffffff;
    font-weight: 600;
    padding: 7px 20px;
}

QPushButton#PrimaryButton:hover {
    background-color: #6b74f5;
}

QPushButton#PrimaryButton:disabled {
    background-color: #33395c;
    color: #6b6f85;
}

/* ---------- Progress bar ---------- */
QProgressBar#MainProgress {
    background-color: #17181f;
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: #8b8fa3;
    font-size: 10px;
}

QProgressBar#MainProgress::chunk {
    background-color: #5865f2;
    border-radius: 5px;
}

/* ---------- Sliders ---------- */
QSlider::groove:horizontal {
    background: #2a2d3a;
    height: 4px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #5865f2;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #6b74f5;
}

QSlider::sub-page:horizontal {
    background: #5865f2;
    border-radius: 2px;
}

/* ---------- Checkboxes ---------- */
QCheckBox {
    color: #c4c7d1;
    font-size: 12px;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #454962;
    background-color: #17181f;
}

QCheckBox::indicator:checked {
    background-color: #5865f2;
    border: 1px solid #5865f2;
}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
}

QScrollBar::handle:vertical {
    background: #353849;
    border-radius: 4px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #454962;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


def apply_theme(app) -> None:
    """
    Applies the EstateLens visual theme to the whole application.
    Purely cosmetic — cannot alter any widget's behavior, signals,
    or slots. If anything goes wrong reading the stylesheet, the app
    still runs, just unstyled, rather than crashing the demo.
    """
    try:
        app.setStyleSheet(_QSS)
    except Exception as e:
        print(f"[style] Could not apply theme: {e}")
"""The Settings and Scheduler pages reached from the left rail."""
from __future__ import annotations

import dataclasses
from typing import Callable

from PySide6.QtCore import QTime, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QTimeEdit, QVBoxLayout, QWidget)

from ..engine.recommend.learn import available_trainers
from ..engine.scheduler import Schedule
from ..engine.settings import Settings

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
FOLDERS = (("folder_general", "Files"), ("folder_video", "Videos"), ("folder_playlist", "Playlists"))


def _page(title: str, intro: str) -> tuple[QWidget, QVBoxLayout]:
    page = QWidget(objectName="page")
    col = QVBoxLayout(page)
    col.setContentsMargins(24, 22, 24, 22)
    col.setSpacing(18)
    col.addWidget(QLabel(title, objectName="h1"))
    hint = QLabel(intro, objectName="muted")
    hint.setWordWrap(True)
    col.addWidget(hint)
    return page, col


class SettingsPage(QWidget):
    def __init__(self, settings: Settings, on_change: Callable[[Settings], None], parent=None):
        super().__init__(parent, objectName="page")
        self._settings, self._on_change = settings, on_change
        page, col = _page("Settings", "Where downloads are saved, and how videos and playlists are named. "
                                      "Speed, connections and theme are in the Queue panel on the right.")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        self._fields: dict[str, QLineEdit] = {}
        for key, label in FOLDERS:
            col.addWidget(QLabel(f"Save {label.lower()} to", objectName="panelLabel"))
            row = QHBoxLayout()
            field = QLineEdit(getattr(settings, key), objectName=key)
            field.setProperty("role", "field")
            field.editingFinished.connect(lambda k=key, f=field: self._set(**{k: f.text().strip()}))
            browse = QPushButton("Browse…", objectName="secondaryButton")
            browse.clicked.connect(lambda _=False, k=key, f=field: self._browse(k, f))
            row.addWidget(field, 1)
            row.addWidget(browse)
            col.addLayout(row)
            self._fields[key] = field

        self._number = QCheckBox("Number playlist files (01 - Title.mp4)", objectName="number_playlist_files")
        self._number.setChecked(settings.number_playlist_files)
        self._number.toggled.connect(lambda on: self._set(number_playlist_files=on))
        self._subs = QCheckBox("Download subtitles when a video has them", objectName="subtitles")
        self._subs.setChecked(settings.subtitles)
        self._subs.toggled.connect(lambda on: self._set(subtitles=on))
        self._suggest = QCheckBox("Suggest videos on the For you page", objectName="suggest_videos")
        self._suggest.setChecked(settings.suggest_videos)
        self._suggest.toggled.connect(lambda on: self._set(suggest_videos=on))
        self._learned = QCheckBox("Use the learned ranker on For you when it's ready", objectName="rec_learned")
        self._learned.setChecked(settings.rec_learned)
        self._learned.toggled.connect(lambda on: self._set(rec_learned=on))
        col.addWidget(self._number)
        col.addWidget(self._subs)
        col.addWidget(self._suggest)
        col.addWidget(self._learned)
        self._meaning = QCheckBox("Match by meaning", objectName="rec_meaning")
        self._meaning.setToolTip("For you compares what videos are about, not only the words they share")
        self._meaning.setChecked(settings.rec_meaning)
        self._meaning.toggled.connect(lambda on: self._set(rec_meaning=on))
        self._files = QCheckBox("Use file downloads for suggestions", objectName="rec_files")
        self._files.setToolTip("Programs, archives and course videos you download shape the suggestions; "
                               "documents and pictures never do")
        self._files.setChecked(settings.rec_files)
        self._files.toggled.connect(lambda on: self._set(rec_files=on))
        self._two_tower = QCheckBox("Try the two-tower ranker (v7, experimental)", objectName="rec_two_tower")
        self._two_tower.setToolTip("A small two-tower network learns from your own lists; it ranks For you only "
                                   "when it beats the current ranker. Needs Match by meaning")
        self._two_tower.setChecked(settings.rec_two_tower)
        self._two_tower.toggled.connect(lambda on: self._set(rec_two_tower=on))
        self._sasrec = QCheckBox("Try the sequence ranker (v8, experimental)", objectName="rec_sasrec")
        self._sasrec.setToolTip("A small attention network reads your downloads in order and learns from your own "
                                "lists; it ranks For you only when it beats the current ranker. Needs Match by meaning")
        self._sasrec.setChecked(settings.rec_sasrec)
        self._sasrec.toggled.connect(lambda on: self._set(rec_sasrec=on))
        col.addWidget(self._meaning)
        col.addWidget(self._two_tower)
        col.addWidget(self._sasrec)
        col.addWidget(self._files)
        trainer_row = QHBoxLayout()
        trainer_row.addWidget(QLabel("Train it with", objectName="panelLabel"))
        self._trainer = QComboBox(objectName="rec_trainer")
        have = available_trainers()
        for key, text in (("hand", "Hand-written gradient descent"), ("sklearn", "scikit-learn")):
            if key not in have:
                text += " (install scikit-learn to use this)"
            self._trainer.addItem(text, key)
            self._trainer.model().item(self._trainer.count() - 1).setEnabled(key in have)
        self._trainer.setCurrentIndex(max(0, self._trainer.findData(settings.rec_trainer if
                                                                    settings.rec_trainer in have else "hand")))
        self._trainer.currentIndexChanged.connect(lambda _i: self._set(rec_trainer=self._trainer.currentData()))
        trainer_row.addWidget(self._trainer)
        trainer_row.addStretch(1)
        col.addLayout(trainer_row)
        discovery_row = QHBoxLayout()
        discovery_row.addWidget(QLabel("Discovery card", objectName="panelLabel"))
        self._discovery = QComboBox(objectName="rec_discovery")
        self._discovery.setToolTip("Who picks the \"Something a little different\" card on For you")
        for key, text in (("random", "Random"), ("source", "Learn which search works"),
                          ("contextual", "Learn from every input")):
            self._discovery.addItem(text, key)
        self._discovery.setCurrentIndex(max(0, self._discovery.findData(settings.rec_discovery)))
        self._discovery.currentIndexChanged.connect(
            lambda _i: self._set(rec_discovery=self._discovery.currentData()))
        discovery_row.addWidget(self._discovery)
        discovery_row.addStretch(1)
        col.addLayout(discovery_row)
        col.addStretch(1)

    def set_settings(self, settings: Settings) -> None:
        self._settings = settings
        for box, on in ((self._meaning, settings.rec_meaning), (self._files, settings.rec_files)):
            box.blockSignals(True)  # changed elsewhere (the For you banner): show it, don't send it back
            box.setChecked(on)
            box.blockSignals(False)
        for key, field in self._fields.items():
            if not field.hasFocus():
                field.setText(getattr(settings, key))

    def _browse(self, key: str, field: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder", field.text())
        if folder:
            field.setText(folder)
            self._set(**{key: folder})

    def _set(self, **changes) -> None:
        if all(getattr(self._settings, k) == v for k, v in changes.items()):
            return
        self._settings = dataclasses.replace(self._settings, **changes)
        self._on_change(self._settings)


class SchedulerPage(QWidget):
    def __init__(self, settings: Settings, on_change: Callable[[Settings], None], parent=None):
        super().__init__(parent, objectName="page")
        self._settings, self._on_change = settings, on_change
        s = settings.schedule
        page, col = _page("Scheduler", "Start the queue at a set time, for example at night, "
                                       "and stop it again in the morning if you like.")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        self._enabled = QCheckBox("Start the queue on a schedule", objectName="enabled")
        self._enabled.setChecked(s.enabled)
        col.addWidget(self._enabled)

        times = QHBoxLayout()
        times.setSpacing(12)
        times.addWidget(QLabel("Start at", objectName="panelLabel"))
        self._start = QTimeEdit(QTime.fromString(s.start, "HH:mm"), objectName="start")
        self._start.setDisplayFormat("HH:mm")
        times.addWidget(self._start)
        times.addSpacing(18)
        self._use_stop = QCheckBox("Stop at", objectName="use_stop")
        self._use_stop.setChecked(s.stop is not None)
        self._stop = QTimeEdit(QTime.fromString(s.stop or "07:00", "HH:mm"), objectName="stop")
        self._stop.setDisplayFormat("HH:mm")
        self._stop.setEnabled(s.stop is not None)
        times.addWidget(self._use_stop)
        times.addWidget(self._stop)
        times.addStretch(1)
        col.addLayout(times)

        col.addWidget(QLabel("On these days", objectName="panelLabel"))
        days = QHBoxLayout()
        days.setSpacing(4)
        self._days: list[QPushButton] = []
        for i, name in enumerate(DAYS):
            b = QPushButton(name, objectName="segButton", checkable=True)
            b.setChecked(i in s.days)
            b.setCursor(Qt.PointingHandCursor)
            b.toggled.connect(self._changed)
            self._days.append(b)
            days.addWidget(b)
        days.addStretch(1)
        col.addLayout(days)
        col.addStretch(1)

        self._enabled.toggled.connect(self._changed)
        self._start.timeChanged.connect(self._changed)
        self._use_stop.toggled.connect(self._changed)
        self._stop.timeChanged.connect(self._changed)

    def _changed(self, *_args) -> None:
        self._stop.setEnabled(self._use_stop.isChecked())
        schedule = Schedule(
            enabled=self._enabled.isChecked(),
            start=self._start.time().toString("HH:mm"),
            stop=self._stop.time().toString("HH:mm") if self._use_stop.isChecked() else None,
            days=[i for i, b in enumerate(self._days) if b.isChecked()],
        )
        self._settings = dataclasses.replace(self._settings, schedule=schedule)
        self._on_change(self._settings)

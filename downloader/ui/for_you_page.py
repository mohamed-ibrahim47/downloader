"""The For you page: suggested videos, why each one is there, and the buttons that teach the recommender.

Searching YouTube is slow, so `Recommender.refresh` runs in the pool. Everything else here reads the local
database and runs on the GUI thread. A refresh result carries the token it was started with; an older one
that arrives late is ignored, so it can never overwrite a newer list.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from ..engine.recommend.learn import Weights
from ..engine.recommend.rank import Suggestion
from ..engine.recommend.service import Recommender
from ..engine.timecode import fmt_clock
from .elided import ElidedLabel
from .pages import _page
from .workers import run_in_pool

COLD = "Download a few videos and suggestions will show up here."
EMPTY = "No new suggestions. Try Refresh later."
LOADING = "Looking for new videos…"
WORDS = 15  # profile words shown in "Why this order?"


def _plain(text: str, name: str) -> QLabel:
    label = QLabel(text, objectName=name)
    label.setTextFormat(Qt.PlainText)  # server text is never read as HTML
    label.setWordWrap(True)
    return label


def _menu_text(text: str) -> str:
    return text.replace("&", "&&")  # a lone & would turn the next letter into a keyboard shortcut


def _score_text(s: Suggestion) -> str:
    if s.learned:  # v2's score is a probability; similarity and channel are only two of its inputs
        return f"chance {s.score:.0%} (learned) · similarity {s.similarity:.2f} · channel {s.boost:.2f}"
    return f"score {s.score:.2f} = similarity {s.similarity:.2f} + channel {s.boost:.2f}"


TRAINER_NAMES = {"hand": "hand-written trainer", "sklearn": "scikit-learn"}


def ranker_text(model: dict | None, ranker: str) -> str:
    """The "Why this order?" block: which ranker is on, how learning goes, and the learned weights."""
    lines = []
    weights = Weights.from_json(model["weights"]) if model and model.get("weights") else None
    if ranker == "v2" and weights:
        lines.append(f"Ranked by: Learned (v2), {TRAINER_NAMES.get(weights.trainer, weights.trainer)}")
    else:
        lines.append("Ranked by: Hand-tuned (v1)")
    if model:
        reason = model.get("reason", "")
        if ranker == "v1" and model.get("active"):
            reason += ". Turned off in Settings"
        lines.append(reason)
    compare = {name: Weights.from_json(text) for name, text in (model or {}).get("compare", {}).items()}
    if compare:
        names = [n for n in ("hand", "sklearn") if n in compare]
        lines.append("Learned weights (" + " / ".join(TRAINER_NAMES[n] for n in names) + "):")
        first = compare[names[0]]
        for i, feature in enumerate(first.names):
            lines.append(f"  {feature}  " + " / ".join(f"{compare[n].w[i]:+.2f}" for n in names))
    return "\n".join(lines)


def _ago(seconds: float) -> str:
    if seconds < 60:
        return "Updated just now"
    if seconds < 3600:
        return f"Updated {int(seconds // 60)} min ago"
    return f"Updated {int(seconds // 3600)} h ago"


class SuggestionCard(QFrame):
    def __init__(self, s: Suggestion, show_score: bool, parent=None):
        super().__init__(parent, objectName="suggestionCard")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # never shorter than its contents
        c = s.candidate
        col = QVBoxLayout(self)
        col.setContentsMargins(16, 12, 16, 12)
        col.setSpacing(4)

        col.addWidget(ElidedLabel(c.title, objectName="title"))
        meta = " · ".join(x for x in (c.channel, fmt_clock(c.duration) if c.duration else None) if x)
        # One line each, shortened with "…": a wrapping label's height depends on its width, and the scroll
        # area doesn't count that, so wrapped cards got squashed when drawn while the page was on screen.
        col.addWidget(ElidedLabel(meta, objectName="meta"))
        if s.discovery:
            reason = s.because or ""
        elif s.because:
            # Say what actually matched: shared words, or only the channel.
            why = ", ".join(s.shared) if s.shared else ("same channel" if s.boost > 0 else "")
            reason = f"Because you downloaded {s.because}" + (f" · {why}" if why else "")
        else:
            reason = ""
        self.reason = ElidedLabel(reason, objectName="reason")
        self.reason.setProperty("discovery", s.discovery)
        self.reason.setVisible(bool(reason))
        col.addWidget(self.reason)
        self.score = ElidedLabel(_score_text(s), objectName="score")
        self.score.setVisible(show_score)
        col.addWidget(self.score)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.setContentsMargins(0, 6, 0, 0)
        self.download = QPushButton("Download", objectName="download")
        self.dismiss = QPushButton("Dismiss", objectName="dismiss")
        self.more = QToolButton(objectName="more", text="…")  # "⋯" is missing from the app's fonts
        self.more.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.more)
        self.block = menu.addAction("Not this channel")
        self.block.setEnabled(bool(c.channel))
        self.more.setMenu(menu)
        for b in (self.download, self.dismiss, self.more):
            b.setCursor(Qt.PointingHandCursor)
            row.addWidget(b)
        row.addStretch(1)
        col.addLayout(row)


class ForYouPage(QWidget):
    download_requested = Signal(str)  # the video's URL

    def __init__(self, recommender: Recommender, parent=None):
        super().__init__(parent, objectName="page")
        self._rec = recommender
        self._token = 0
        self._cards: list[SuggestionCard] = []
        self._shown: list[Suggestion] = []
        self.state = "cold"
        self.offline = False
        self._busy = False  # a refresh is running in the pool
        self._active = False  # the page is the one on screen (open() until leave())

        page, col = _page("For you", "Videos picked from what you download, by this app's own ranking. "
                                     "Dismiss what you don't want and the list learns.")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self._updated = QLabel("", objectName="muted")
        self._refresh = QPushButton("Refresh", objectName="secondaryButton")
        self._refresh.clicked.connect(self.refresh)
        self._blocked = QToolButton(objectName="blocked", text="Blocked channels")
        self._blocked.setPopupMode(QToolButton.InstantPopup)
        self._blocked.setMenu(QMenu(self._blocked))
        self._why = QCheckBox("Why this order?", objectName="why")
        self._why.toggled.connect(self._show_why)
        bar.addWidget(self._updated)
        bar.addStretch(1)
        for w in (self._why, self._blocked, self._refresh):
            bar.addWidget(w)
        col.addLayout(bar)

        self._offline = _plain("", "offline")
        self._offline.hide()
        self._ranker = _plain("", "rankerInfo")
        self._ranker.hide()
        self._words = _plain("", "profileWords")
        self._words.hide()
        self._message = _plain("", "muted")
        col.addWidget(self._offline)
        col.addWidget(self._ranker)
        col.addWidget(self._words)
        col.addWidget(self._message)

        self._list = QVBoxLayout()
        self._list.setSpacing(10)
        holder = QWidget()
        holder_col = QVBoxLayout(holder)
        holder_col.setContentsMargins(0, 0, 0, 0)
        holder_col.addLayout(self._list)
        holder_col.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        col.addWidget(scroll, 1)

    # ---- for tests and the window ------------------------------------------------

    def cards(self) -> list[QWidget]:
        return list(self._cards)

    def suggestion(self, i: int) -> Suggestion:
        return self._shown[i]

    # ---- opening and refreshing ----------------------------------------------------

    def open(self) -> None:
        """Called when the page is shown. Searches only when the cache is older than REFRESH_AFTER."""
        self._active = True
        if self._busy:
            return  # a search is already running; its result will draw the list
        if not self._rec.ready():
            self._show_cold()
        elif self._rec.needs_refresh():
            self.refresh()
        else:
            self._render()

    def leave(self) -> None:
        """Called when the owner moves to another page: a refresh that finishes now isn't drawn or logged."""
        self._active = False

    def refresh(self) -> None:
        self._token += 1
        self._busy = True
        self.state = "loading"  # the old cards stay up until the new list arrives
        self._updated.setText(LOADING)
        self._refresh.setEnabled(False)
        task = run_in_pool(self._rec.refresh)
        task.tag = self._token
        task.tagged_done.connect(self._on_refreshed)
        task.tagged_failed.connect(self._on_failed)

    def _on_refreshed(self, token, _result) -> None:
        if token != self._token:
            return  # a newer refresh is running; this result is stale
        self._busy = False
        self.offline = False
        self._finish()

    def _on_failed(self, token, message: str) -> None:
        if token != self._token:
            return
        self._busy = False
        self.offline = True
        self._offline.setText(message)
        self._finish()

    def _finish(self) -> None:
        self._refresh.setEnabled(True)
        if self._active:
            self._render()  # otherwise open() draws the fresh cache when the owner comes back

    # ---- drawing ---------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        self.state = "loading" if self._busy else state
        self._refresh.setEnabled(not self._busy)
        self._offline.setVisible(self.offline)

    def _show_cold(self) -> None:
        self._clear_cards()
        self._shown = []
        self._message.setText(COLD)
        self._message.setVisible(True)
        self._words.hide()
        self._updated.setText("")
        self._set_state("cold")

    def _render(self) -> None:
        """One read of the downloads per render (Recommender.view)."""
        why = self._why.isChecked()
        view = self._rec.view(words=WORDS if why else 0)
        if not view.ready:
            self._show_cold()
            return
        self._shown = view.suggestions
        self._clear_cards()
        for s in self._shown:
            card = SuggestionCard(s, why)
            card.download.clicked.connect(lambda _=False, url=s.candidate.url: self.download_requested.emit(url))
            card.dismiss.clicked.connect(lambda _=False, x=s: self._dismiss(x))
            card.block.triggered.connect(lambda _=False, ch=s.candidate.channel: self._block(ch))
            self._list.addWidget(card)
            self._cards.append(card)
        # Offline with nothing to show: the red line says it all, so no second "try later" message.
        self._message.setText("" if self._shown else EMPTY)
        self._message.setVisible(not self._shown and not self.offline)
        age = self._rec.cache_age()
        self._updated.setText(_ago(age) if age is not None else "")
        if why:
            self._set_words(view.words)
            self._set_ranker()
        self._words.setVisible(why and bool(self._shown))
        self._ranker.setVisible(why and bool(self._shown))
        self._fill_blocked()
        self._set_state("list" if self._shown else "empty")

    def _set_ranker(self) -> None:
        self._ranker.setText(ranker_text(self._rec.model(), self._rec.ranker()))

    def _set_words(self, words: list[tuple[str, float]]) -> None:
        self._words.setText("Your words: " + " · ".join(f"{w} {v:.2f}" for w, v in words))

    def _clear_cards(self) -> None:
        for card in self._cards:
            card.setParent(None)  # off the page now, not at the next event-loop turn
            card.deleteLater()
        self._cards = []

    def _fill_blocked(self) -> None:
        menu = self._blocked.menu()
        menu.clear()
        names = self._rec.blocked_channels()
        for name in names:
            menu.addAction(f"Unblock {_menu_text(name)}").triggered.connect(
                lambda _=False, n=name: self._unblock(n))
        self._blocked.setVisible(bool(names))

    def _show_why(self, on: bool) -> None:
        if on and self._shown:
            self._set_words(self._rec.words(WORDS))  # worked out only when asked for
            self._set_ranker()
        self._words.setVisible(on and bool(self._shown))
        self._ranker.setVisible(on and bool(self._shown))
        for card in self._cards:
            card.score.setVisible(on)

    # ---- feedback: re-rank from the cache, no new search ----------------------------

    def _dismiss(self, s: Suggestion) -> None:
        self._rec.dismiss(s)
        self._render()

    def _block(self, channel: str | None) -> None:
        if channel:
            self._rec.block_channel(channel)
            self._render()

    def _unblock(self, channel: str) -> None:
        self._rec.unblock_channel(channel)
        self._render()

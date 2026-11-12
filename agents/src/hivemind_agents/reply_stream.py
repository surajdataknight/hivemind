"""what the customer has been shown of a reply that answers several problems, part by part, in order.

a part the communicator writes arrives sentence by sentence; a template part arrives whole. when a written part
starts, the parts before it are shown first, so the screen never shows them out of order; a `replace` from the
communicator replaces only its own part. a reply of one part is sent exactly as before: a template part only with
`done`.
"""

from typing import Callable

from hivemind_agents.replies import PART_BREAK

Emit = Callable[[dict], None]


class ReplyStream:
    def __init__(self) -> None:
        self.shown = ""                 # the text sent to the screen so far
        self._unshown: list[str] = []   # finished parts not sent yet
        self._parts = 0                 # finished parts
        self._streaming = False         # the part in progress has sent something

    def writer(self, emit: Emit) -> Emit:
        """the emit for a part that streams."""
        before: str | None = None

        def write(event: dict) -> None:
            nonlocal before
            if before is None:
                self._catch_up(emit, gap_after=True)
                before = self.shown
                self._streaming = True
            if event["type"] == "replace":
                self.shown = before + event["text"]
                emit({**event, "text": self.shown})
            else:
                self.shown += event["text"]
                emit(event)

        return write

    def part_done(self, text: str) -> None:
        if not self._streaming:
            self._unshown.append(text)
        self._streaming = False
        self._parts += 1

    def finish(self, emit: Emit, tail: list[str]) -> None:
        """a reply of several parts: sends what is not shown yet, the question at the end included."""
        self._unshown.extend(tail)
        self._catch_up(emit, gap_after=False)

    def _catch_up(self, emit: Emit, gap_after: bool) -> None:
        text = PART_BREAK.join(self._unshown)
        self._unshown = []
        if text and self.shown:
            text = PART_BREAK + text
        if gap_after and (self.shown or text):
            text += PART_BREAK
        if text:
            self.shown += text
            emit({"type": "token", "text": text})

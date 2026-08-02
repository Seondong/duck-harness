"""Render a window of frames as motion the model can see.

A single board says what is there; it never says what moved. Analogy recall is
visual-motor -- "the piece rotates into a slot", "the blocks fall" -- so the
model has to be shown a sequence, not a state. These renderers turn a window of
history into two images: a filmstrip of the frames in time order, and a single
board with every change in the window overlaid.

Nothing here knows about games, agents, or models. Frames in, data URLs out.
"""
from __future__ import annotations

import base64
import io
import os
import re
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.vision_context import ARC_COLOR_MAP

WINDOW_FRAMES = 6
# Many actions do nothing, and a strip of them teaches nothing: ka59 produced
# five near-identical panels this way. One dead transition is itself a finding
# worth showing; more than one and the window is mostly silence.
MAX_DEAD_TRANSITIONS = 1
# How far back to look for a window worth showing.
SEARCH_FRAMES = 24

# One visual token covers 32x32 px (patch_size 16, merge_size 2), so no upscale
# resolves a single cell -- the vision channel is for gestalt, and segmentation
# stays the precise one. Upscale 4 is still worth paying for: at upscale 2 the
# model called a 45-degree rotation "90 degrees", and at 4 it did not.
STRIP_UPSCALE = 4
# The overlay is one panel rather than six. At the strip's upscale it lands
# under the processor's 256x256 floor and gets rescaled anyway, so draw it big.
TRAIL_UPSCALE = 8

_LABEL_H = 14
_GUTTER = 26
_ARROW_W = 12
_TRAIL_FADE = 0.72


def _env_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def strip_upscale() -> int:
    return _env_int("FRAMING_STRIP_UPSCALE", STRIP_UPSCALE)


def trail_upscale() -> int:
    return _env_int("FRAMING_TRAIL_UPSCALE", TRAIL_UPSCALE)


def short_action(action: Any) -> str:
    """`MOUSE(row=40, col=22)` is too wide to sit under a panel; `M(40,22)` fits."""
    text = str(action or "").strip()
    if not text:
        return "?"
    match = re.search(r"row\s*=\s*(\d+).*?col\s*=\s*(\d+)", text)
    if match:
        return f"M({match.group(1)},{match.group(2)})"
    return text


def _changed_cells(before: Frame, after: Frame) -> int:
    rows = min(len(before.grid), len(after.grid))
    total = 0
    for r in range(rows):
        row_a, row_b = before.grid[r], after.grid[r]
        for c in range(min(len(row_a), len(row_b))):
            if row_a[c] != row_b[c]:
                total += 1
    return total


def select_window(
    history_entries: Sequence[HistoryEntry],
    current_frame: Frame | None,
    *,
    size: int = WINDOW_FRAMES,
) -> list[tuple[Frame, str]] | None:
    """The most recent window worth looking at, or None.

    Three rules, all learned from real recordings: one level only, no RESET
    inside the window, and at most one transition that changed nothing.
    """
    frames: list[tuple[Frame, str]] = []
    for entry in history_entries:
        if entry.frame is None or not entry.frame.grid:
            continue
        frames.append((entry.frame, short_action(entry.action)))
    if current_frame is not None and current_frame.grid:
        if not frames or frames[-1][0] is not current_frame:
            if not frames or frames[-1][0].step != current_frame.step:
                frames.append((current_frame, frames[-1][1] if frames else "?"))
    if len(frames) < size:
        return None

    # Only the recent past is worth framing from, and scoring every window ever
    # recorded would cost more than it is worth.
    frames = frames[-SEARCH_FRAMES:]
    deltas = [
        _changed_cells(frames[i - 1][0], frames[i][0])
        for i in range(1, len(frames))
    ]

    best: tuple[int, int, list[tuple[Frame, str]]] | None = None
    for start in range(len(frames) - size, -1, -1):
        window = frames[start:start + size]
        if len({frame.level for frame, _ in window}) != 1:
            continue
        # RESET rewinds the level. The frames on either side belong to different
        # attempts, so the overlay would draw a jump that never happened.
        if any(action.upper().startswith("RESET") for _, action in window[1:]):
            continue
        window_deltas = deltas[start:start + size - 1]
        if sum(1 for d in window_deltas if d == 0) > MAX_DEAD_TRANSITIONS:
            continue
        # Prefer the busiest recent window, not merely the latest acceptable
        # one. A HUD timer ticking one cell a step keeps every transition
        # technically alive while the board stands still; the window where
        # something actually happened carries far more changed cells.
        score = sum(window_deltas)
        if best is None or score > best[0]:
            best = (score, start, window)
    return best[2] if best is not None else None


def _board_image(frame: Frame, scale: int) -> Image.Image:
    rows = len(frame.grid)
    cols = max((len(row) for row in frame.grid), default=0)
    if rows <= 0 or cols <= 0:
        raise ValueError("cannot render an empty grid")
    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
    pixels = image.load()
    for r, row in enumerate(frame.grid):
        for c in range(cols):
            value = row[c] if c < len(row) else 0
            pixels[c, r] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
    return image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)


def _blend(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    target = (250, 250, 250)
    return tuple(int(rgb[i] + (target[i] - rgb[i]) * amount) for i in range(3))  # type: ignore[return-value]


def _font() -> Any:
    for path in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, 12)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def filmstrip_data_url(window: Sequence[tuple[Frame, str]], *, upscale: int | None = None) -> str:
    """The window in one row, oldest first, each panel labelled with its action."""
    scale = upscale or strip_upscale()
    panels = [_board_image(frame, scale) for frame, _ in window]
    last = len(window) - 1
    labels = ["start"] + [
        f"{action} -> " + ("now" if i == last else f"t-{last - i}")
        for i, (_, action) in enumerate(window)
        if i > 0
    ]

    pw, ph = panels[0].size
    count = len(panels)
    width = count * pw + (count + 1) * _GUTTER + (count - 1) * _ARROW_W
    height = ph + _LABEL_H + 2 * _GUTTER
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = _font()

    x = _GUTTER
    for i, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel, (x, _GUTTER))
        draw.rectangle([x - 1, _GUTTER - 1, x + pw, _GUTTER + ph], outline=(120, 120, 120))
        draw.text((x + 2, _GUTTER + ph + 2), label, fill=(20, 20, 20), font=font)
        if i < count - 1:
            draw.text((x + pw + _GUTTER // 2, _GUTTER + ph // 2 - 7), ">", fill=(80, 80, 80), font=font)
        x += pw + _GUTTER + _ARROW_W
    return _to_data_url(canvas)


def trail_data_url(window: Sequence[tuple[Frame, str]], *, upscale: int | None = None) -> str:
    """Every change in the window on one board, faint = earlier, solid = later.

    Good for translation, poor for rotation in place, where a shape overlaps
    itself into a blob. It is handed over as its own image and introduced as a
    summary rather than a frame: joined onto the filmstrip, the model read it as
    a seventh timestep and invented an arrow in the blob.
    """
    scale = upscale or trail_upscale()
    frames = [frame for frame, _ in window]
    last = frames[-1]
    rows = len(last.grid)
    cols = max((len(row) for row in last.grid), default=0)
    if rows <= 0 or cols <= 0:
        raise ValueError("cannot render an empty grid")

    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
    pixels = image.load()
    for r, row in enumerate(last.grid):
        for c in range(cols):
            value = row[c] if c < len(row) else 0
            pixels[c, r] = _blend(ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0]), _TRAIL_FADE)

    span = max(1, len(frames) - 1)
    for i in range(1, len(frames)):
        before, after = frames[i - 1], frames[i]
        fade = 0.70 - 0.70 * (i / span)
        for r in range(min(len(before.grid), len(after.grid))):
            row_a, row_b = before.grid[r], after.grid[r]
            for c in range(min(len(row_a), len(row_b))):
                if row_a[c] != row_b[c]:
                    pixels[c, r] = _blend(ARC_COLOR_MAP.get(int(row_b[c]), ARC_COLOR_MAP[0]), fade)

    panel = image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)
    pw, ph = panel.size
    canvas = Image.new("RGB", (pw + 2 * _GUTTER, ph + 2 * _GUTTER + _LABEL_H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(panel, (_GUTTER, _GUTTER))
    draw.rectangle([_GUTTER - 1, _GUTTER - 1, _GUTTER + pw, _GUTTER + ph], outline=(120, 120, 120))
    draw.text(
        (_GUTTER, _GUTTER + ph + 3),
        "every change in the window; faint = earlier, solid = later",
        fill=(20, 20, 20),
        font=_font(),
    )
    return _to_data_url(canvas)


def window_actions(window: Iterable[tuple[Frame, str]]) -> list[str]:
    return [action for i, (_, action) in enumerate(window) if i > 0]

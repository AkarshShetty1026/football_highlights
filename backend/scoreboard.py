"""Goal detection from the broadcast scoreboard, instead of from what was said.

The commentary is a poor clock for goals: measured on a real match, the words
"in the back of the net" arrived 50 seconds after the ball crossed the line,
because the commentator says nothing during the roar and only describes it over
the replay. The score bug burned into the picture has no such lag -- it flips
within a few seconds and, unlike a sentence, it cannot be ambiguous about
whether a goal actually counted.

Two stages, because neither alone is enough:

1. Sample the score region on keyframes only. Decoding every frame of a 50fps
   match takes over ten minutes; keyframes take twenty seconds and land every
   ~5 seconds, finer than the graphic's own update latency.
2. Find changes that PERSIST. A goal changes the digits for the rest of the
   match; replays, graphics and the panel hiding itself all revert. Comparing a
   median of the frames before against a median of the frames after ignores
   everything transient -- which matters, because a raw frame-to-frame diff
   ranks both real goals below ~38 pieces of noise.
"""

import subprocess
from dataclasses import dataclass

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


@dataclass(frozen=True)
class PixelRegion:
    """A concrete crop box, resolved against one video's actual dimensions."""
    x: int
    y: int
    w: int
    h: int

    def crop_filter(self) -> str:
        # Explicit scale: ffmpeg silently rounds odd crop dimensions, which
        # corrupts the fixed-size buffer the frames are read back into.
        return f"crop={self.w}:{self.h}:{self.x}:{self.y},scale={self.w}:{self.h},format=gray"


@dataclass(frozen=True)
class ScoreRegion:
    """Where the score digits sit, as FRACTIONS of frame width and height.

    Broadcasters keep the scoreboard in the same place relative to the frame
    regardless of the resolution they ship, so fractions survive a 480p feed
    becoming 720p or 1080p -- pixel coordinates do not, and a region measured at
    480p would land in the middle of the pitch at 1080p.

    Excludes the match clock deliberately: it ticks every second and would
    swamp any change detection."""
    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, frame_w: int, frame_h: int) -> PixelRegion:
        # Round the size up and keep it even: a sub-pixel rounding difference
        # between the crop and the read-back buffer corrupts every frame.
        w = max(int(round(self.w * frame_w)) // 2 * 2, 8)
        h = max(int(round(self.h * frame_h)) // 2 * 2, 6)
        x = min(max(int(round(self.x * frame_w)), 0), max(frame_w - w, 0))
        y = min(max(int(round(self.y * frame_h)), 0), max(frame_h - h, 0))
        return PixelRegion(x=x, y=y, w=w, h=h)

    @classmethod
    def from_pixels(cls, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> "ScoreRegion":
        return cls(x=x / frame_w, y=y / frame_h, w=w / frame_w, h=h / frame_h)


# Measured on a Sky Sports broadcast at 854x480 (x=66 y=30 w=48 h=14), stored as
# fractions so the same numbers hold at any resolution that broadcaster ships.
DEFAULT_REGION = ScoreRegion(x=66 / 854, y=30 / 480, w=48 / 854, h=14 / 480)


def video_dimensions(video_path: str) -> tuple[int, int]:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
        capture_output=True, text=True,
    )
    parts = result.stdout.strip().splitlines()[0].split("x") if result.stdout.strip() else []
    if len(parts) < 2:
        raise RuntimeError(f"could not read video dimensions from {video_path}")
    return int(parts[0]), int(parts[1])


@dataclass
class ScoreChange:
    time: float       # seconds into the recording, when the digits changed
    strength: float   # how much the scoring side's digit changed
    side: str         # "home" (left digit) or "away" (right digit)
    quiet_side: float # how much the OTHER digit changed -- the discriminator


def keyframe_times(video_path: str) -> list[float]:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v", "-skip_frame", "nokey",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    times = []
    for line in result.stdout.splitlines():
        field = line.split(",")[0].strip()
        if field:
            try:
                times.append(float(field))
            except ValueError:
                pass
    return times


def extract_strip(
    video_path: str, region: ScoreRegion = DEFAULT_REGION
) -> tuple[np.ndarray, np.ndarray, PixelRegion]:
    """Return (times, frames, resolved_region) for the score region on every keyframe."""
    frame_w, frame_h = video_dimensions(video_path)
    box = region.to_pixels(frame_w, frame_h)
    times = keyframe_times(video_path)
    result = subprocess.run(
        [FFMPEG, "-v", "error", "-skip_frame", "nokey", "-i", video_path,
         "-vf", box.crop_filter(), "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        capture_output=True,
    )
    px = box.w * box.h
    count = len(result.stdout) // px
    frames = np.frombuffer(result.stdout[: count * px], np.uint8).reshape(count, box.h, box.w)
    n = min(count, len(times))
    return np.asarray(times[:n], dtype=float), frames[:n].astype(np.int16), box


def _panel_present(frames: np.ndarray) -> np.ndarray:
    """The bug is on screen for most of a match, so the per-pixel median across
    the whole match IS the panel. Frames far from it are ones where a replay or
    a graphic has taken the corner over."""
    reference = np.median(frames, axis=0)
    distance = np.abs(frames - reference).mean(axis=(1, 2))
    # Bimodal: present frames cluster near zero, absent ones far away.
    return distance < max(np.median(distance) * 3.0, 8.0)


def detect_score_changes(
    times: np.ndarray,
    frames: np.ndarray,
    window: int = 12,
    min_present: int = 5,
    digit_change_min: float = 12.0,
    quiet_side_max: float = 6.0,
    min_gap_sec: float = 60.0,
) -> list[ScoreChange]:
    """Find keyframes where exactly one score digit changed, permanently.

    Magnitude alone is useless here -- measured on a real match, every false
    positive (kickoff, half-time, full-time graphics) produced a LARGER change
    than either actual goal. What separates them is locality: a goal repaints
    one digit and leaves the other pixel-identical, while a graphic repaints the
    whole panel. The 'quiet' side reads 1.5 on a real goal and 10-47 on noise,
    so requiring one side to stay still is what makes this work."""
    n = len(frames)
    if n < 2 * window + 2:
        return []

    present = _panel_present(frames)
    third = frames.shape[2] // 3          # left digit | separator | right digit
    scored = np.zeros(n)
    quiet = np.full(n, np.inf)
    side = [""] * n

    for i in range(window, n - window):
        before = frames[i - window : i][present[i - window : i]]
        after = frames[i + 1 : i + 1 + window][present[i + 1 : i + 1 + window]]
        if len(before) < min_present or len(after) < min_present:
            continue
        delta = np.abs(np.median(before, axis=0) - np.median(after, axis=0))
        left, right = delta[:, :third].mean(), delta[:, -third:].mean()
        scored[i] = max(left, right)
        quiet[i] = min(left, right)
        side[i] = "home" if left >= right else "away"

    ok = (scored >= digit_change_min) & (quiet <= quiet_side_max)

    # Non-maximum suppression: one goal must not report as five adjacent hits.
    changes: list[ScoreChange] = []
    for i in np.argsort(np.where(ok, scored, -1))[::-1]:
        if not ok[i]:
            break
        if any(abs(times[i] - c.time) < min_gap_sec for c in changes):
            continue
        if not _persists(times, frames, present, float(times[i]), third,
                         digit_change_min, quiet_side_max):
            continue
        changes.append(ScoreChange(
            time=float(times[i]), strength=float(scored[i]),
            side=side[i], quiet_side=float(quiet[i]),
        ))
    changes.sort(key=lambda c: c.time)
    return changes


def _median_between(times, frames, present, t0, t1, min_frames=5):
    mask = (times >= t0) & (times <= t1) & present
    if mask.sum() < min_frames:
        return None
    return np.median(frames[mask], axis=0)


def _persists(
    times, frames, present, t, third, digit_change_min, quiet_side_max,
) -> bool:
    """Confirm the change is still there minutes later.

    A goal holds the new score for the rest of the half; a graphic sliding
    through the corner does not. This is what rejects pre-match idents, which
    otherwise pass the locality test -- the Premier League logo animating across
    an empty corner changes one 'digit' region while the other stays blank."""
    before = _median_between(times, frames, present, t - 90, t - 15)
    if before is None:
        return True  # no evidence to reject on -- keep it

    after = _median_between(times, frames, present, t + 180, t + 420)
    if after is None:
        # Late goal: not enough match left for the preferred horizon, so use
        # whatever follows. A 90th-minute winner must never be dropped here.
        after = _median_between(times, frames, present, t + 60, times[-1], min_frames=3)
    if after is None:
        return True

    delta = np.abs(before - after)
    left, right = delta[:, :third].mean(), delta[:, -third:].mean()
    # This is a CONFIRMATION, not a second detection, so the bar for the changed
    # side is deliberately half the detection threshold: anything that actually
    # reverted reads near zero here, while a real change can read low simply
    # because some digit pairs differ by fewer pixels than others (1->2 is a
    # smaller change than 0->1, and measured 10.3 against a 12.0 detection bar,
    # which silently dropped a real goal). The quiet side still does the
    # discriminating -- a passing graphic reads 40+ there.
    return max(left, right) >= digit_change_min * 0.5 and min(left, right) <= quiet_side_max


def find_goals(video_path: str, region: ScoreRegion = DEFAULT_REGION, **kwargs) -> list[ScoreChange]:
    times, frames, box = extract_strip(video_path, region)
    print(f"scoreboard: reading {box.w}x{box.h} at ({box.x},{box.y}) across {len(frames)} keyframes")
    return detect_score_changes(times, frames, **kwargs)


# ---------------------------------------------------------------------------
# Automatic location of the score region
#
# A hand-measured region only works for the broadcast it was measured on. These
# three signals find it on any feed, and each exists because the one before it
# was not enough on its own:
#
# 1. WHERE IS THE GRAPHIC? A scoreboard is rendered, not filmed, so it comes out
#    pixel-identical frame after frame; grass and crowd never do, because of
#    sensor noise. Measured on a real match: 825 identical pixels in the top
#    band of the frame, 0 in the bottom band of pure pitch.
# 2. WHICH ROWS ARE THE SCORE? The clock shares the panel and ticks every
#    second, so counting how often each ROW changes splits them cleanly --
#    measured 13-16 changes for the score rows against 32-91 for the clock.
# 3. WHICH COLUMNS ARE THE DIGITS? Counting per column finds them, but ONLY
#    among frames where the panel is actually on screen. Without that filter
#    every column shows ~12 changes (the panel being covered by replays and
#    coming back) and the digits are invisible. With it, the crests read 0 and
#    the digits read 3-4.
# ---------------------------------------------------------------------------

LOCALIZE_WIDTH = 854            # analyse at this width; digits stay legible
LOCALIZE_BANDS = ((0.0, 0.25), (0.75, 1.0))   # scoreboards live near an edge
MIN_OVERLAY_PIXELS = 50


def _extract_band(video_path: str, y0: int, h: int, frame_w: int, frame_h: int):
    """Full-width horizontal slice of the frame, on keyframes, scaled to a
    consistent analysis width so thresholds hold at any resolution."""
    w = min(frame_w, LOCALIZE_WIDTH)
    aw, ah = w // 2 * 2, max(h * w // frame_w // 2 * 2, 8)
    result = subprocess.run(
        [FFMPEG, "-v", "error", "-skip_frame", "nokey", "-i", video_path,
         "-vf", f"crop={frame_w}:{h}:0:{y0},scale={aw}:{ah},format=gray",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        capture_output=True,
    )
    px = aw * ah
    count = len(result.stdout) // px
    if count < 40:
        return None, aw, ah
    frames = np.frombuffer(result.stdout[: count * px], np.uint8).reshape(count, ah, aw)
    return frames.astype(np.int16), aw, ah


def _overlay_bbox(frames: np.ndarray):
    """Bounding box of the rendered graphic: pixels that repeat exactly."""
    identical = (np.abs(np.diff(frames, axis=0)) <= 2).mean(axis=0)
    overlay = identical > 0.5
    if overlay.sum() < MIN_OVERLAY_PIXELS:
        return None
    # Keep the dominant block rather than scattered stragglers.
    cols, rows = overlay.sum(axis=0), overlay.sum(axis=1)
    cx = np.nonzero(cols >= max(cols.max() * 0.2, 1))[0]
    ry = np.nonzero(rows >= max(rows.max() * 0.2, 1))[0]
    if not len(cx) or not len(ry):
        return None
    return int(cx.min()), int(ry.min()), int(cx.max() - cx.min() + 1), int(ry.max() - ry.min() + 1)


def _block_medians(frames: np.ndarray, block: int = 12):
    return [np.median(frames[i : i + block], axis=0) for i in range(0, len(frames) - block, block)]


def _refine_to_digits(panel: np.ndarray):
    """Drop the ticking clock rows, then keep the columns that change rarely."""
    reference = np.median(panel, axis=0)
    distance = np.abs(panel - reference).mean(axis=(1, 2))
    present = distance < max(np.median(distance) * 3.0, 8.0)
    if present.sum() < 60:
        return None

    blocks = _block_medians(panel)
    if len(blocks) < 8:
        return None
    transitions = len(blocks) - 1

    row_counts = np.array([
        sum(1 for a, b in zip(blocks, blocks[1:]) if np.abs(a[y] - b[y]).mean() > 8)
        for y in range(panel.shape[1])
    ])
    # A ticking row changes far more often than a score row (measured: 32-91
    # against 13-16), so judge each row against the panel's own median rather
    # than an absolute count, which would not survive a different broadcast.
    quiet_row = row_counts <= max(np.median(row_counts) * 1.8, 3)
    # Take the longest CONTIGUOUS run: using min/max instead lets one stray
    # quiet row on the far side of the clock stretch the band back over it.
    best_run, run = (0, 0, 0), None
    for y, ok in enumerate(np.append(quiet_row, False)):
        if ok and run is None:
            run = y
        elif not ok and run is not None:
            if y - run > best_run[0]:
                best_run = (y - run, run, y - 1)
            run = None
    if best_run[0] < 4:
        return None
    y0, y1 = best_run[1], best_run[2]

    # Columns, among present frames only -- this filter is what makes it work.
    sub = panel[present][:, y0 : y1 + 1, :]
    sub_blocks = _block_medians(sub)
    if len(sub_blocks) < 8:
        return None
    sub_transitions = len(sub_blocks) - 1
    col_counts = np.array([
        sum(1 for a, b in zip(sub_blocks, sub_blocks[1:]) if np.abs(a[:, x] - b[:, x]).mean() > 10)
        for x in range(panel.shape[2])
    ])
    hi = max(4, int(sub_transitions * 0.08))
    digit_cols = np.nonzero((col_counts >= 1) & (col_counts <= hi))[0]
    if len(digit_cols) < 8:
        return None
    return int(digit_cols.min()), y0, int(digit_cols.max() - digit_cols.min() + 1), y1 - y0 + 1


def locate_score_region(video_path: str) -> ScoreRegion | None:
    """Find the score digits on an unseen broadcast. None if nothing looks like
    a scoreboard, in which case the caller should fall back to a known region."""
    frame_w, frame_h = video_dimensions(video_path)
    best = None
    for y0f, y1f in LOCALIZE_BANDS:
        y0 = int(y0f * frame_h)
        h = max(int((y1f - y0f) * frame_h) // 2 * 2, 8)
        frames, aw, ah = _extract_band(video_path, y0, h, frame_w, frame_h)
        if frames is None:
            continue
        box = _overlay_bbox(frames)
        if box is None:
            continue
        bx, by, bw, bh = box
        refined = _refine_to_digits(frames[:, by : by + bh, bx : bx + bw])
        if refined is None:
            continue
        rx, ry, rw, rh = refined
        # Analysis coords -> fractions of the real frame.
        region = ScoreRegion(
            x=(bx + rx) / aw,
            y=(y0 + (by + ry) * h / ah) / frame_h,
            w=rw / aw,
            h=rh * h / ah / frame_h,
        )
        score = rw * rh
        if best is None or score > best[0]:
            best = (score, region)
        # A confident hit in the first band means no need to decode the second.
        if rw >= 12 and rh >= 6:
            break
    return best[1] if best else None

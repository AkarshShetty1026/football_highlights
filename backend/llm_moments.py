"""Highlight-moment detection via an LLM (gpt-5.6-luna) reading the match
transcript, instead of keyword/audio-energy heuristics. The model understands
commentary semantically -- it can tell "he smashes it into the back of the
net" is a goal and "50 Premier League goals... quicker than Andy Cole" is a
stats aside, which substring keyword matching fundamentally cannot.

Each moment is also classified into an event category, which is what the
finished reel's chapter index is built from.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from .transcription import Segment

load_dotenv()

MODEL = "gpt-5.6-luna"
CHUNK_DURATION_SEC = 900.0  # 15 minutes of match time per LLM request
CHUNK_OVERLAP_SEC = 30.0    # shared context across chunk boundaries
MAX_WORKERS = 6
REASONING_EFFORT = "medium"          # per-chunk detection
RECONCILE_EFFORT = "high"            # the single whole-match consistency pass

# Event categories the reel is indexed by. "other" covers everything that is
# highlight-worthy but not one of the named events (saves, big chances, cards
# that aren't fouls, etc.) -- more named categories can be added here later.
CATEGORIES = ["goal", "foul", "half_time", "full_time", "other"]

CATEGORY_LABELS = {
    "goal": "Goal",
    "foul": "Foul",
    "half_time": "Half time",
    "full_time": "Full time",
    "other": "Highlight",
}

# When two moments get merged, the surviving label is the higher-priority one:
# a goal that happens to sit next to a foul is still, to a viewer, the goal.
CATEGORY_PRIORITY = {"goal": 0, "full_time": 1, "half_time": 2, "foul": 3, "other": 4}

# Structural markers -- they carry the reel's shape rather than action, so they
# are indexed and kept even though their raw excitement is low.
MARKER_CATEGORIES = ("half_time", "full_time")
MARKER_CLIP_PRE = 4.0
MARKER_CLIP_POST = 4.0

SYSTEM_PROMPT = (
    "You are analyzing football (soccer) match commentary to pick moments for a "
    "highlights reel. Timestamps in the transcript are in seconds from the start "
    "of the match. Identify moments a viewer would want to see: goals, big "
    "chances, great saves, key fouls/cards, penalties, controversial decisions, "
    "and other genuinely exciting passages of play. Also identify the two "
    "structural markers of the match if the commentary states them: the "
    "half-time whistle/break and the full-time whistle/final result.\n\n"
    "Classify every moment with a 'category':\n"
    "- 'goal': the ball goes in and the goal stands (not a disallowed goal, "
    "not a near miss, not a reference to a goal scored earlier or in another match).\n"
    "- 'foul': a foul, free kick award, penalty award, booking, or sending off.\n"
    "- 'half_time': the referee ends the first half / the half-time break begins.\n"
    "- 'full_time': the referee ends the match / the final whistle goes.\n"
    "- 'other': anything else worth watching (saves, big chances, near misses, "
    "controversial decisions).\n\n"
    "For each moment, give the start and end time in seconds spanning the "
    "relevant passage (include a little buildup context where natural), an "
    "importance score from 0 to 1 (1.0 = definite goal or match-defining "
    "moment, lower = still interesting but less critical), a brief reason, and "
    "a 'quote' field containing the EXACT verbatim transcript text (copy it, do "
    "not paraphrase) that the start/end times are based on. Do not flag routine "
    "buildup play, stats trivia, historical records, or analysis with no live "
    "action. Only report a moment if you can back it with a real, exact quote "
    "from the transcript."
)

MOMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "importance": {"type": "number"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "reason": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": [
                    "start_seconds", "end_seconds", "importance", "category", "reason", "quote",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["moments"],
    "additionalProperties": False,
}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@dataclass
class Moment:
    start: float
    end: float
    importance: float
    reason: str
    category: str = "other"


@dataclass
class HighlightWindow:
    """A span of the source match that made it into the reel, plus the label
    the chapter index will carry for it."""
    start: float
    end: float
    category: str
    reason: str
    importance: float

    @property
    def label(self) -> str:
        return CATEGORY_LABELS.get(self.category, "Highlight")


def _best_category(a: str, b: str) -> str:
    return a if CATEGORY_PRIORITY.get(a, 99) <= CATEGORY_PRIORITY.get(b, 99) else b


def _mergeable(a: str, b: str) -> bool:
    """Markers never absorb, or get absorbed by, ordinary moments.

    A stoppage-time winner sits seconds away from the final whistle, and
    proximity-merging the two would swallow the "Full time" entry into the goal
    -- the index would silently lose a chapter the user explicitly asked for."""
    return (a in MARKER_CATEGORIES) == (b in MARKER_CATEGORIES)


def _format_chunk(segments: list[Segment]) -> str:
    return "\n".join(f"[{seg.start:.1f} - {seg.end:.1f}] {seg.text}" for seg in segments)


def _chunk_segments(segments: list[Segment], total_duration: float) -> list[list[Segment]]:
    if not segments:
        return []
    chunks: list[list[Segment]] = []
    step = CHUNK_DURATION_SEC - CHUNK_OVERLAP_SEC
    window_start = 0.0
    while window_start < total_duration:
        window_end = window_start + CHUNK_DURATION_SEC
        chunk = [s for s in segments if s.start < window_end and s.end > window_start]
        if chunk:
            chunks.append(chunk)
        window_start += step
    return chunks


_TIMESTAMP_RE = re.compile(r"\[\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*\]")


def _normalize_quote(text: str) -> str:
    """Quotes come back spanning several transcript lines, so they arrive with
    newlines -- and sometimes with the '[123.0 - 130.0]' prefixes copied along
    with the words. Flatten both away before comparing, otherwise a perfectly
    real quote fails to match the transcript it was copied from."""
    return " ".join(_TIMESTAMP_RE.sub(" ", text.lower()).split())


def _is_grounded(quote: str, start: float, end: float, segments: list[Segment], tolerance: float = 8.0) -> bool:
    """Reject moments whose quote doesn't actually appear near the claimed
    timestamp -- catches both fabricated quotes and quotes that are real but
    misattributed to the wrong point in the match (both observed in testing)."""
    quote_norm = _normalize_quote(quote)
    if not quote_norm:
        return False
    nearby_text = _normalize_quote(
        " ".join(
            s.text for s in segments if s.end >= start - tolerance and s.start <= end + tolerance
        )
    )
    probe = quote_norm[:40]
    return probe in nearby_text


def _call_chunk(segments: list[Segment], retries: int = 1) -> list[Moment]:
    transcript_text = _format_chunk(segments)
    if not transcript_text.strip():
        return []

    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Transcript:\n{transcript_text}"},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "highlight_moments", "schema": MOMENT_SCHEMA, "strict": True},
                },
            )
            data = json.loads(response.choices[0].message.content)
            moments = []
            for m in data.get("moments", []):
                start, end = float(m["start_seconds"]), float(m["end_seconds"])
                if end <= start:
                    continue
                quote = m.get("quote", "")
                if not _is_grounded(quote, start, end, segments):
                    print(f"llm_moments: rejected ungrounded moment at {start:.0f}s: {quote[:60]!r}")
                    continue
                category = m.get("category", "other")
                if category not in CATEGORIES:
                    category = "other"
                moments.append(
                    Moment(start, end, float(m["importance"]), m.get("reason", ""), category)
                )
            return moments
        except Exception as exc:  # noqa: BLE001 -- one failed chunk shouldn't kill the whole job
            last_error = exc
    print(f"llm_moments: chunk failed after retries: {last_error}")
    return []


def _merge_overlapping(moments: list[Moment], gap_tolerance: float = 15.0) -> list[Moment]:
    """Merge moments that overlap OR sit close together in time. The model
    sometimes splits one continuous event (e.g. the strike, then a beat later
    the record-breaking reaction to it) into separate adjacent moments -- those
    need to end up as one clip, not compete against each other for a slot."""
    if not moments:
        return []
    moments = sorted(moments, key=lambda m: m.start)
    merged = [moments[0]]
    for m in moments[1:]:
        last = merged[-1]
        if m.start <= last.end + gap_tolerance and _mergeable(last.category, m.category):
            merged[-1] = Moment(
                start=last.start,
                end=max(last.end, m.end),
                importance=max(last.importance, m.importance),
                reason=last.reason if last.importance >= m.importance else m.reason,
                category=_best_category(last.category, m.category),
            )
        else:
            merged.append(m)
    return merged


RECONCILE_SYSTEM_PROMPT = (
    "You are auditing an automatic highlight detector's findings against the FULL "
    "transcript of one football match. The detector reads the match in 15-minute "
    "chunks in isolation, so it makes two predictable mistakes: it reports replays, "
    "recaps and statistical asides as if they were live goals, and it misses or "
    "misattributes goals because it never sees the rest of the match.\n\n"
    "Your job is to produce the definitive list of these events for the WHOLE "
    "match: every goal, the half-time whistle, and the full-time whistle.\n\n"
    "Use the running score as your check. Commentary states the score repeatedly "
    "and names the scorer, especially just after a goal and in the full-time "
    "summary. Work out the final score first, then make sure the goals you list "
    "add up to exactly that score. If a candidate goal would make the running "
    "score inconsistent with the final score, it is a replay or a recap, not a "
    "goal -- drop it. If the final score requires a goal the detector never "
    "reported, find it in the transcript and add it.\n\n"
    "Transcript lines marked (LONG Ns) are spans where crowd noise stopped "
    "Whisper segmenting cleanly. A goal very often falls inside one of these, "
    "because the loudest moment of the match is the goal itself. Within such a "
    "span the live call is usually LATE, after the buildup words -- so when your "
    "event lands in a long span, make start_seconds/end_seconds cover that whole "
    "span rather than guessing a point inside it.\n\n"
    "For each event give: 'start_seconds'/'end_seconds' spanning the LIVE call of "
    "the event (not the analysis afterwards, and never the replay), the scoring "
    "team and player where known, the score immediately after it, a one-line "
    "description, and a 'quote' holding the EXACT verbatim transcript text you "
    "based the timing on. Copy the quote, never paraphrase. Beware that the "
    "transcript garbles player names -- judge by context, and give the name you "
    "are confident the commentator meant.\n\n"
    "Report only goals that actually stood. Do not report disallowed goals, near "
    "misses, saves or fouls here; another pass handles those."
)

RECONCILE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_score": {"type": "string"},
        "reasoning_summary": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["goal", "half_time", "full_time"]},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "scorer": {"type": "string"},
                    "score_after": {"type": "string"},
                    "description": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": [
                    "category", "start_seconds", "end_seconds",
                    "scorer", "score_after", "description", "quote",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["final_score", "reasoning_summary", "events"],
    "additionalProperties": False,
}

KEY_CATEGORIES = ("goal", "half_time", "full_time")


LONG_SEGMENT_SEC = 20.0


def _format_full_transcript(segments: list[Segment]) -> str:
    """Rendering for the whole-match pass. Both timestamps are essential, not
    padding: crowd noise makes Whisper merge a goal into one huge segment (a
    47-second line held the entire Alexander-Arnold goal in testing, and the
    longest was 93 seconds). With only a start time the model cannot tell
    whether that content happened at the timestamp or a minute after it, and it
    will place the goal at the start of the span -- before the ball goes in.

    Long spans are flagged explicitly so the model treats them as imprecise."""
    lines = []
    for seg in segments:
        span = seg.end - seg.start
        marker = f" (LONG {span:.0f}s)" if span >= LONG_SEGMENT_SEC else ""
        lines.append(f"[{seg.start:.0f}-{seg.end:.0f}{marker}] {seg.text}")
    return "\n".join(lines)


def _snap_to_segment_bounds(
    moments: list[Moment], segments: list[Segment], max_extend: float = 60.0
) -> list[Moment]:
    """Widen each event to cover the transcript segments it lands in.

    When the live call is buried inside one long segment, no timestamp the model
    picks within that span is reliable -- so cover the whole span instead. The
    clip gets longer, but the goal is definitely inside it, which is the only
    thing that actually matters for a highlight."""
    snapped: list[Moment] = []
    for m in moments:
        overlapping = [s for s in segments if s.end > m.start and s.start < m.end]
        if not overlapping:
            snapped.append(m)
            continue
        start = min(m.start, min(s.start for s in overlapping))
        end = max(m.end, max(s.end for s in overlapping))
        # Never let one pathological segment turn a goal into a two-minute clip.
        start = max(start, m.start - max_extend)
        end = min(end, m.end + max_extend)
        snapped.append(Moment(start, end, m.importance, m.reason, m.category))
    return snapped


def reconcile_key_events(
    moments: list[Moment], segments: list[Segment]
) -> tuple[list[Moment], dict]:
    """Re-derive goals and the half/full-time whistles from the entire transcript
    at once, and swap them in for whatever the chunked pass produced.

    The chunked pass keeps ownership of fouls and general highlights -- those are
    local judgements it makes fine. Goals are the ones that need global context,
    because only the whole match tells you what the score ended up being.

    Returns (moments, info). On any failure the input is returned untouched: a
    reconciliation problem must never cost the user the reel entirely."""
    info: dict = {"ran": False}
    if not segments:
        return moments, info

    candidates = [m for m in moments if m.category in KEY_CATEGORIES]
    candidate_text = "\n".join(
        f"- {m.category} at {m.start:.0f}s-{m.end:.0f}s: {m.reason}" for m in candidates
    ) or "- (none found)"

    user_content = (
        f"FULL MATCH TRANSCRIPT (timestamps are seconds from the start of the recording):\n"
        f"{_format_full_transcript(segments)}\n\n"
        f"CANDIDATE EVENTS REPORTED BY THE CHUNKED DETECTOR (verify, correct, and complete "
        f"this list -- some may be replays or recaps, and some real events may be absent):\n"
        f"{candidate_text}"
    )

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            reasoning_effort=RECONCILE_EFFORT,
            messages=[
                {"role": "system", "content": RECONCILE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reconciled_events", "schema": RECONCILE_SCHEMA, "strict": True
                },
            },
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001 -- fall back to the unreconciled set
        print(f"llm_moments: reconciliation failed, keeping chunked results: {exc}")
        info["error"] = str(exc)
        return moments, info

    reconciled: list[Moment] = []
    for e in data.get("events", []):
        start, end = float(e["start_seconds"]), float(e["end_seconds"])
        if end <= start:
            end = start + 8.0
        category = e["category"]
        # Grounding still applies: the whole-transcript view makes the model more
        # accurate, not incapable of inventing a quote.
        if not _is_grounded(e.get("quote", ""), start, end, segments, tolerance=12.0):
            print(f"llm_moments: reconcile rejected ungrounded {category} at {start:.0f}s")
            continue
        scorer, score_after = e.get("scorer", "").strip(), e.get("score_after", "").strip()
        reason = e.get("description", "")
        if category == "goal" and scorer:
            reason = f"{scorer} ({score_after}) -- {reason}" if score_after else f"{scorer} -- {reason}"
        reconciled.append(Moment(start, end, 1.0, reason, category))

    if not reconciled:
        print("llm_moments: reconciliation returned nothing usable, keeping chunked results")
        info["error"] = "no grounded events"
        return moments, info

    reconciled = _snap_to_segment_bounds(reconciled, segments)
    kept = [m for m in moments if m.category not in KEY_CATEGORIES]
    goals = sum(1 for m in reconciled if m.category == "goal")
    info.update({
        "ran": True,
        "final_score": data.get("final_score", ""),
        "summary": data.get("reasoning_summary", ""),
        "candidates_in": len(candidates),
        "events_out": len(reconciled),
        "goals": goals,
    })
    print(
        f"llm_moments: reconciled {len(candidates)} candidate key events -> "
        f"{len(reconciled)} ({goals} goals), final score {data.get('final_score','?')!r}"
    )
    return _merge_overlapping(kept + reconciled), info


def detect_moments(segments: list[Segment], total_duration: float) -> list[Moment]:
    """Send the transcript to the LLM in overlapping time-chunks (in parallel),
    then merge duplicate/overlapping moments reported across chunk boundaries.

    This pass sees the match only 15 minutes at a time; run the results through
    reconcile_key_events() afterwards to catch what needs the whole picture."""
    chunks = _chunk_segments(segments, total_duration)
    if not chunks:
        return []

    all_moments: list[Moment] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as pool:
        futures = [pool.submit(_call_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            all_moments.extend(future.result())

    return _merge_overlapping(all_moments)


def select_highlight_windows(
    moments: list[Moment],
    target_total_duration_sec: float,
    sensitivity: float = 0.5,
    clip_pre: float = 8.0,
    clip_post: float = 6.0,
) -> list[HighlightWindow]:
    """Pick the highest-importance, non-overlapping moments (padded with a
    little pre/post-roll) until the highlight reel hits its target length.

    Half-time and full-time markers bypass the importance floor and the length
    budget: they are quiet moments that would never win a slot on excitement,
    but the reel is indexed by them, so one of each is always kept when the
    commentary contains it. They get tighter padding and are not charged
    against the budget, so indexing the reel never costs it a goal."""
    sensitivity = min(max(sensitivity, 0.0), 1.0)
    min_importance = 0.75 - sensitivity * 0.5  # higher sensitivity -> lower floor -> more clips

    def window_for(m: Moment) -> HighlightWindow:
        # A whistle needs no run-up; an attacking move does.
        pre, post = (
            (min(clip_pre, MARKER_CLIP_PRE), min(clip_post, MARKER_CLIP_POST))
            if m.category in MARKER_CATEGORIES
            else (clip_pre, clip_post)
        )
        return HighlightWindow(
            start=max(0.0, m.start - pre),
            end=m.end + post,
            category=m.category,
            reason=m.reason,
            importance=m.importance,
        )

    selected: list[HighlightWindow] = []
    total_duration = 0.0
    content_count = 0  # real football taken so far: goals + action, not whistles

    def take(m: Moment, enforce_budget: bool) -> None:
        nonlocal total_duration, content_count
        w = window_for(m)
        overlap_idx = next(
            (i for i, s in enumerate(selected) if not (w.end <= s.start or w.start >= s.end)), None
        )
        if overlap_idx is not None:
            # Same event the model split into two moments -- extend the
            # existing window rather than losing this one to a slot conflict.
            existing = selected[overlap_idx]
            new_start, new_end = min(existing.start, w.start), max(existing.end, w.end)
            total_duration += (new_end - new_start) - (existing.end - existing.start)
            keep_existing = existing.importance >= w.importance
            selected[overlap_idx] = HighlightWindow(
                start=new_start,
                end=new_end,
                category=_best_category(existing.category, w.category),
                reason=existing.reason if keep_existing else w.reason,
                importance=max(existing.importance, w.importance),
            )
            return

        clip_len = w.end - w.start
        # Something other than whistles always makes the cut, even if it alone
        # busts a very short target. Goals satisfy that on their own, so this
        # only forces a highlight in when the match had no goals at all.
        if enforce_budget and content_count and total_duration + clip_len > target_total_duration_sec:
            return
        selected.append(w)
        if m.category not in MARKER_CATEGORIES:
            content_count += 1
        if enforce_budget:
            total_duration += clip_len

    # Markers first, so they claim their slot before the budget is spent. Only
    # the best-supported occurrence of each is kept -- a match has one of each.
    for category in MARKER_CATEGORIES:
        best = max(
            (m for m in moments if m.category == category), key=lambda m: m.importance, default=None
        )
        if best is not None:
            take(best, enforce_budget=False)

    # Every goal, unconditionally. A highlight reel missing a goal is simply
    # wrong, so goals are not made to compete with saves and fouls for room --
    # the target length governs the discretionary highlights around them.
    for goal in sorted((m for m in moments if m.category == "goal"), key=lambda m: m.start):
        take(goal, enforce_budget=False)

    candidates = [
        m for m in moments
        if m.category not in MARKER_CATEGORIES and m.category != "goal"
        and m.importance >= min_importance
    ]
    candidates.sort(key=lambda m: -m.importance)
    for m in candidates:
        take(m, enforce_budget=True)

    selected.sort(key=lambda w: w.start)
    merged: list[HighlightWindow] = []
    for w in selected:
        if merged and w.start <= merged[-1].end + 1.0 and _mergeable(merged[-1].category, w.category):
            last = merged[-1]
            keep_last = last.importance >= w.importance
            merged[-1] = HighlightWindow(
                start=last.start,
                end=max(last.end, w.end),
                category=_best_category(last.category, w.category),
                reason=last.reason if keep_last else w.reason,
                importance=max(last.importance, w.importance),
            )
        else:
            merged.append(w)
    return merged


# The score graphic updates within a few seconds of the ball crossing the line
# (verified frame-by-frame: attack building at -24s, ball in and celebrating at
# the flip), so the clip needs a lead long enough to open on the buildup.
SCOREBOARD_LEAD_SEC = 26.0
SCOREBOARD_LAG_SEC = 16.0
SCOREBOARD_MATCH_TOLERANCE = 150.0


def expected_goal_count(final_score: str) -> int | None:
    """Total goals implied by a score the commentary stated, e.g. 'Man City 2-0
    Real Madrid' -> 2. Used to decide whether the scoreboard scan was complete
    enough to be allowed to overrule the transcript."""
    if not final_score:
        return None
    m = re.search(r"(\d+)\s*[-–:]\s*(\d+)", final_score)
    return int(m.group(1)) + int(m.group(2)) if m else None


def apply_scoreboard_goals(
    moments: list[Moment],
    changes: list,
    tolerance: float = SCOREBOARD_MATCH_TOLERANCE,
    final_score: str = "",
) -> tuple[list[Moment], dict]:
    """Let the scoreboard have the final say on goals.

    The commentary establishes WHO scored and at what score; the scoreboard
    establishes WHEN, and how many. Those are different strengths -- measured on
    a real match, the words "in the back of the net" trailed the ball crossing
    the line by 50 seconds, while the graphic flipped within a few. So each
    detected score change is retimed onto the video, keeping whatever the
    transcript knew about it, and a transcript goal with no corresponding score
    change is dropped as a replay the reader mistook for live action.

    If the scoreboard found nothing at all (no bug, unusual layout, wrong
    region) the transcript's goals are returned untouched -- a detector that
    sees nothing must not be allowed to erase a real goal."""
    if not changes:
        return moments, {"used": False, "reason": "no score changes detected"}

    transcript_goals = [m for m in moments if m.category == "goal"]
    others = [m for m in moments if m.category != "goal"]

    home = away = 0
    rebuilt: list[Moment] = []
    used: set[int] = set()
    for change in changes:
        if change.side == "home":
            home += 1
        else:
            away += 1
        score = f"{home}-{away}"

        best, best_gap = None, tolerance
        for i, g in enumerate(transcript_goals):
            if i in used:
                continue
            gap = abs((g.start + g.end) / 2 - change.time)
            if gap < best_gap:
                best, best_gap = i, gap

        if best is not None:
            used.add(best)
            reason = transcript_goals[best].reason
        else:
            side = "home" if change.side == "home" else "away"
            reason = f"Goal detected from the scoreboard ({side} team, {score})"

        rebuilt.append(Moment(
            start=max(0.0, change.time - SCOREBOARD_LEAD_SEC),
            end=change.time + SCOREBOARD_LAG_SEC,
            importance=1.0,
            reason=reason,
            category="goal",
        ))

    # Only let the scoreboard delete a goal when its own count is corroborated
    # by the score the commentary stated. A scan that found fewer goals than the
    # match actually had has missed one, and must not be trusted to decide that
    # a real goal was imaginary -- dropping a genuine goal is far worse than
    # keeping a doubtful one. (Observed: a 2-0 match where the scan found one
    # change, which deleted Bernardo Silva's second goal from the reel.)
    expected = expected_goal_count(final_score)
    # No stated score means no arbiter, so the scoreboard does not get to
    # overrule the transcript on its own word.
    complete = expected is not None and len(changes) >= expected
    unmatched = [g for i, g in enumerate(transcript_goals) if i not in used]
    if not complete and unmatched:
        print(
            f"llm_moments: scoreboard found {len(changes)} goals but the commentary "
            f"says {expected}; keeping {len(unmatched)} unconfirmed transcript goal(s)"
        )
        rebuilt.extend(unmatched)

    dropped = len(unmatched) if complete else 0
    info = {
        "used": True,
        "score_changes": len(changes),
        "final_score": f"{home}-{away}",
        "transcript_goals": len(transcript_goals),
        "retimed": len(used),
        "dropped_unconfirmed": max(dropped, 0),
        "added_missed": max(len(changes) - len(used), 0),
        "expected_goals": expected,
        "scan_complete": complete,
    }
    print(
        f"llm_moments: scoreboard gave {len(changes)} goals ({home}-{away}); "
        f"retimed {len(used)}, dropped {max(dropped,0)} unconfirmed, "
        f"added {max(len(changes)-len(used),0)} the transcript missed"
    )
    return _merge_overlapping(others + rebuilt), info

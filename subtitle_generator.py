"""
Subtitle Generator
==================
Converts speech in an audio or video file into timestamped subtitles (SRT and/or
WebVTT) using the faster-whisper implementation of OpenAI's Whisper model.
Everything runs locally: the audio is never uploaded anywhere.

Examples
--------
    python subtitle_generator.py --input test_media/clear_speech.m4a
    python subtitle_generator.py --input test_media/lecture.mp4 --model small --format both
    python subtitle_generator.py --input test_media --language en        (whole folder)
    python subtitle_generator.py --input test_media/nepali.m4a --translate
"""

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Hide harmless Hugging Face download warnings (Windows symlinks, no login token)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", module="huggingface_hub")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
SUPPORTED_EXTS = AUDIO_EXTS | VIDEO_EXTS
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "output"

# Readability rules (common subtitle guidelines, see README)
MAX_LINE_CHARS = 42        # characters per line
MAX_LINES = 2              # lines per cue
MAX_CUE_SECONDS = 6.0      # longest time one cue stays on screen
MIN_CUE_SECONDS = 1.0      # shortest time one cue stays on screen
MAX_CHARS_PER_SECOND = 20  # reading speed limit; faster cues are kept on screen longer
PAUSE_SPLIT_SECONDS = 0.8  # a silence this long starts a new cue
LOW_CONFIDENCE = 0.5       # cues below this average word probability are flagged


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------
def format_timestamp(seconds, separator):
    """Seconds -> 'HH:MM:SS,mmm' (SRT, separator ',') or 'HH:MM:SS.mmm' (VTT, '.')."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


# ---------------------------------------------------------------------------
# Building readable subtitle cues
# ---------------------------------------------------------------------------
class Cue:
    """One subtitle on screen: start time, end time, text and confidence."""

    def __init__(self, start, end, text, confidence=None):
        self.start = start
        self.end = end
        self.text = text.strip()
        self.confidence = confidence


def wrap_text(text, max_chars=MAX_LINE_CHARS):
    """Split text into at most two balanced lines, breaking at a space.

    A split near the middle looks better than a full first line and a short
    second line. A break right after a comma is preferred if one is close.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text

    middle = len(text) // 2
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return text

    def score(i):
        distance = abs(i - middle)
        after_comma = text[i - 1] in ",;:" if i > 0 else False
        return distance - (6 if after_comma else 0)

    # only split points where BOTH lines fit the limit; fall back to the middle
    fitting = [i for i in spaces if i <= max_chars and len(text) - i - 1 <= max_chars]
    best = min(fitting or spaces, key=score)
    return text[:best] + "\n" + text[best + 1:]


def cues_from_words(segments):
    """Group word-level timestamps into readable cues.

    A new cue starts when adding the next word would make the cue
      - longer than MAX_LINES x MAX_LINE_CHARS characters,
      - visible longer than MAX_CUE_SECONDS,
      - or when there is a pause of PAUSE_SPLIT_SECONDS before the word.
    Cues also end after sentence punctuation (unless the cue is tiny),
    so that sentences are not cut in awkward places.
    """
    max_chars = MAX_LINE_CHARS * MAX_LINES
    cues = []
    words, probs = [], []

    def close_cue():
        if words:
            text = "".join(w.word for w in words)
            conf = sum(probs) / len(probs) if probs else None
            cues.append(Cue(words[0].start, words[-1].end, text, conf))
        words.clear()
        probs.clear()

    for segment in segments:
        seg_words = segment.words or []
        for word in seg_words:
            if words:
                current_text = "".join(w.word for w in words)
                too_long = len((current_text + word.word).strip()) > max_chars
                too_slow = word.end - words[0].start > MAX_CUE_SECONDS
                paused = word.start - words[-1].end > PAUSE_SPLIT_SECONDS
                if too_long or too_slow or paused:
                    close_cue()
            words.append(word)
            probs.append(word.probability)

            current_text = "".join(w.word for w in words).strip()
            last = word.word.strip().rstrip(".?!")
            is_abbreviation = len(last) <= 1          # "B." in "B. U. S." is not a sentence end
            if (current_text.endswith((".", "?", "!")) and len(current_text) >= 15
                    and not is_abbreviation):
                close_cue()
    close_cue()
    return merge_orphans(cues)


def merge_orphans(cues):
    """Join very short cues ("files.", "University") with a neighbour.

    A cue of one or two words shown for under a second is hard to read.
    It is merged into the previous cue if the result still fits the
    readability limits, otherwise into the next cue.
    """
    max_chars = MAX_LINE_CHARS * MAX_LINES
    limit_s = MAX_CUE_SECONDS + 1.0

    def fits(a, b):
        return (not a.text.endswith((".", "?", "!"))       # keep sentences apart
                and len(a.text) + 1 + len(b.text) <= max_chars
                and b.end - a.start <= limit_s
                and b.start - a.end <= PAUSE_SPLIT_SECONDS)

    def join(a, b):
        confs = [c for c in (a.confidence, b.confidence) if c is not None]
        return Cue(a.start, b.end, a.text + " " + b.text,
                   sum(confs) / len(confs) if confs else None)

    result = []
    i = 0
    while i < len(cues):
        cue = cues[i]
        short = len(cue.text.split()) <= 2 and cue.end - cue.start < MIN_CUE_SECONDS
        if short and result and fits(result[-1], cue):
            result[-1] = join(result[-1], cue)
        elif short and i + 1 < len(cues) and fits(cue, cues[i + 1]):
            result.append(join(cue, cues[i + 1]))
            i += 1
        else:
            result.append(cue)
        i += 1
    return result


def cues_from_segments(segments):
    """Fallback when word timestamps are switched off: one cue per segment,
    long segments are split into pieces with proportionally divided time."""
    max_chars = MAX_LINE_CHARS * MAX_LINES
    cues = []
    for seg in segments:
        text = " ".join(seg.text.split())
        if not text:
            continue
        pieces, current = [], ""
        for token in text.split(" "):
            if current and len(current) + 1 + len(token) > max_chars:
                pieces.append(current)
                current = token
            else:
                current = f"{current} {token}".strip()
        pieces.append(current)

        duration = seg.end - seg.start
        total = sum(len(p) for p in pieces)
        t = seg.start
        for p in pieces:
            share = duration * len(p) / total
            cues.append(Cue(t, t + share, p))
            t += share
    return cues


def fix_timing(cues):
    """Make every cue readable: on screen for at least MIN_CUE_SECONDS and long
    enough to read at MAX_CHARS_PER_SECOND, but never overlapping the next cue."""
    for i, cue in enumerate(cues):
        next_start = cues[i + 1].start if i + 1 < len(cues) else None
        needed = max(MIN_CUE_SECONDS, len(cue.text) / MAX_CHARS_PER_SECOND)
        if cue.end - cue.start < needed:
            new_end = cue.start + needed
            if next_start is not None:
                new_end = min(new_end, next_start - 0.05)
            cue.end = max(cue.end, new_end)
        if next_start is not None and cue.end > next_start:
            cue.end = next_start
    return cues


# ---------------------------------------------------------------------------
# Writing subtitle files
# ---------------------------------------------------------------------------
def write_srt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(cue.start, ',')} --> {format_timestamp(cue.end, ',')}\n")
            f.write(f"{wrap_text(cue.text)}\n\n")


def write_vtt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for cue in cues:
            f.write(f"{format_timestamp(cue.start, '.')} --> {format_timestamp(cue.end, '.')}\n")
            f.write(f"{wrap_text(cue.text)}\n\n")


def write_transcript(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(c.text for c in cues) + "\n")


def write_report(cues, path):
    """CSV with every cue and its confidence, for testing and manual review."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cue", "start", "end", "duration_s", "chars", "chars_per_s",
                    "confidence", "flag", "text"])
        for i, c in enumerate(cues, start=1):
            conf = "" if c.confidence is None else f"{c.confidence:.2f}"
            flag = "LOW" if c.confidence is not None and c.confidence < LOW_CONFIDENCE else ""
            duration = max(c.end - c.start, 0.001)
            w.writerow([i, format_timestamp(c.start, "."), format_timestamp(c.end, "."),
                        f"{duration:.2f}", len(c.text), f"{len(c.text) / duration:.1f}",
                        conf, flag, c.text])


# ---------------------------------------------------------------------------
# Model and transcription
# ---------------------------------------------------------------------------
def load_model(args):
    """Load the Whisper model. The first run downloads it once (needs internet);
    after that it runs fully offline. --offline refuses to download at all."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("[ERROR] faster-whisper is not installed. Run: pip install -r requirements.txt")

    compute_type = "int8" if args.device == "cpu" else "float16"
    print(f"[MODEL] loading '{args.model}' on {args.device} ({compute_type}) ...")
    start = time.time()
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=compute_type,
                             local_files_only=args.offline)
    except Exception as e:
        hint = ("the model is not downloaded yet - run once without --offline"
                if args.offline else
                "check the model name/path and your internet connection "
                "(the model is downloaded on first use)")
        sys.exit(f"[ERROR] Could not load model '{args.model}': {e}\n        Hint: {hint}")
    print(f"[MODEL] ready in {time.time() - start:.1f} s")
    return model


def transcribe_file(model, path, args):
    print(f"\n[FILE] {path.name}")
    start = time.time()
    try:
        segments, info = model.transcribe(
            str(path),
            language=args.language,
            task="translate" if args.translate else "transcribe",
            beam_size=5,
            word_timestamps=not args.no_word_timestamps,
            vad_filter=not args.no_vad,              # skip silence for better timing
            vad_parameters={"min_silence_duration_ms": 500,
                            "threshold": args.vad_threshold},
            initial_prompt=args.prompt,              # optional vocabulary hints
        )
        print(f"       language: {info.language} (probability {info.language_probability:.2f}), "
              f"duration {info.duration:.1f} s")

        collected = []
        for seg in segments:          # transcription happens while iterating
            collected.append(seg)
            if args.verbose:
                print(f"       [{format_timestamp(seg.start, '.')} -> "
                      f"{format_timestamp(seg.end, '.')}] {seg.text.strip()}")
    except KeyboardInterrupt:
        sys.exit("\n[STOPPED] transcription cancelled by user")
    except Exception as e:
        print(f"[ERROR] Transcription failed for {path.name}: {e}")
        print("        The file may be corrupt, have no audio track, or use an unsupported codec.")
        return

    if not collected:
        print("[WARN] No speech detected - no subtitle file written.")
        return

    if args.no_word_timestamps:
        cues = cues_from_segments(collected)
    else:
        cues = cues_from_words(collected)
    cues = fix_timing(cues)

    # outputs
    stem = path.stem + ("_en" if args.translate else "") + args.suffix
    written = []
    if args.format in ("srt", "both"):
        p = args.output / f"{stem}.srt"
        write_srt(cues, p)
        written.append(p)
    if args.format in ("vtt", "both"):
        p = args.output / f"{stem}.vtt"
        write_vtt(cues, p)
        written.append(p)
    write_transcript(cues, args.output / f"{stem}.txt")
    write_report(cues, args.output / f"{stem}_report.csv")

    # summary
    elapsed = time.time() - start
    confs = [c.confidence for c in cues if c.confidence is not None]
    low = [i for i, c in enumerate(cues, 1)
           if c.confidence is not None and c.confidence < LOW_CONFIDENCE]
    longest_line = max(len(line) for c in cues for line in wrap_text(c.text).split("\n"))
    speed = info.duration / elapsed if elapsed > 0 else 0
    print(f"       cues: {len(cues)} | processing time: {elapsed:.1f} s "
          f"({speed:.1f}x real time)")
    if confs:
        print(f"       average word confidence: {sum(confs) / len(confs):.2f} | "
              f"low-confidence cues (<{LOW_CONFIDENCE}): {low if low else 'none'}")
    fastest = max(len(c.text) / max(c.end - c.start, 0.001) for c in cues)
    shortest = min(c.end - c.start for c in cues)
    print(f"       longest line: {longest_line} characters | shortest cue: {shortest:.1f} s | "
          f"fastest cue: {fastest:.0f} characters/s")
    for p in written:
        print(f"       saved {p}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate SRT/WebVTT subtitles from audio or video with faster-whisper.")
    p.add_argument("--input", type=Path, required=True, help="audio/video file or a folder of files")
    p.add_argument("--model", default="small",
                   help=f"model size ({', '.join(MODEL_SIZES)}) or a path to a local model folder")
    p.add_argument("--language", default=None,
                   help="language code, e.g. en, fi, ne (default: detect automatically)")
    p.add_argument("--prompt", default=None,
                   help='vocabulary hints for names/terms, e.g. "Bibas Dhital, Metropolia, WebVTT"')
    p.add_argument("--suffix", default="",
                   help="text added to output file names, e.g. _prompt (to compare settings)")
    p.add_argument("--translate", action="store_true", help="translate the speech into English subtitles")
    p.add_argument("--format", choices=["srt", "vtt", "both"], default="both", help="subtitle format")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output folder")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="run on CPU or NVIDIA GPU")
    p.add_argument("--no-vad", action="store_true", help="do not skip silent parts")
    p.add_argument("--vad-threshold", type=float, default=0.5,
                   help="speech detection sensitivity 0-1 (lower keeps quieter speech, default 0.5)")
    p.add_argument("--no-word-timestamps", action="store_true",
                   help="use segment timestamps only (faster, rougher line splitting)")
    p.add_argument("--offline", action="store_true", help="never download: use an already downloaded model")
    p.add_argument("--verbose", action="store_true", help="print each segment while transcribing")
    return p.parse_args()


def main():
    args = parse_args()

    # check the input before loading the (slow) model
    if not args.input.exists():
        sys.exit(f"[ERROR] Input not found: {args.input}")
    if args.input.is_dir():
        files = sorted(f for f in args.input.iterdir() if f.is_file())
        skipped = [f for f in files if f.suffix.lower() not in SUPPORTED_EXTS]
        files = [f for f in files if f.suffix.lower() in SUPPORTED_EXTS]
        for f in skipped:
            print(f"[SKIP] Unsupported file type: {f.name}")
        if not files:
            sys.exit(f"[ERROR] No supported audio/video files in {args.input}\n"
                     f"        Supported: {', '.join(sorted(SUPPORTED_EXTS))}")
    else:
        if args.input.suffix.lower() not in SUPPORTED_EXTS:
            sys.exit(f"[ERROR] Unsupported file type: {args.input.suffix or '(none)'}\n"
                     f"        Supported: {', '.join(sorted(SUPPORTED_EXTS))}")
        files = [args.input]

    if not 0.0 < args.vad_threshold < 1.0:
        sys.exit("[ERROR] --vad-threshold must be between 0 and 1")

    if args.model not in MODEL_SIZES and not Path(args.model).exists():
        sys.exit(f"[ERROR] Unknown model '{args.model}'. Use one of {', '.join(MODEL_SIZES)} "
                 "or a path to a downloaded model folder.")

    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    for f in files:
        transcribe_file(model, f, args)


if __name__ == "__main__":
    main()

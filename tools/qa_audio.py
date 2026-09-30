"""Audio QA: regenerate every clip and check it.

    python3 tools/qa_audio.py [--asr path/to/sherpa-onnx-whisper-dir]

Words, sentences and prompts are transcribed with Whisper (sherpa-onnx) and compared
with the text they should say. Letter sounds can't be checked by a recognizer, so they
are measured instead: length, loudness, clipping, and a fading tail.
"""
import argparse, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build  # noqa: E402

captured = {}


class Capture:
    """Stands in for the MP3 encoder so we get the exact samples the game would ship."""
    def set_bit_rate(self, *_): pass
    def set_in_sample_rate(self, r): captured["rate"] = r
    def set_channels(self, *_): pass
    def set_quality(self, *_): pass
    def encode(self, b): captured["x"] = np.frombuffer(b, np.int16).astype(float); return b""
    def flush(self): return b""


build.lameenc.Encoder = Capture

SAME = {"sun": {"son"}, "bee": {"be", "b"}, "tee": {"t"}, "see": {"sea", "c"}, "i": {"eye", "aye"},
        "nine": {"9"}, "five": {"5"}, "six": {"6"}, "ten": {"10"}, "red": {"read"}, "ant": {"aunt"},
        "hi": {"high"}, "sat": {"set"}, "moose": {"mousse"}, "fern": {"firn"}}


def norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="/tmp/claude-0/sherpa-onnx-whisper-small.en")
    ap.add_argument("--only", default="", help="regex of clip names to check")
    args = ap.parse_args()

    import sherpa_onnx
    d = Path(args.asr)
    stem = d.name.replace("sherpa-onnx-whisper-", "")
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(d / f"{stem}-encoder.int8.onnx"), decoder=str(d / f"{stem}-decoder.int8.onnx"),
        tokens=str(d / f"{stem}-tokens.txt"), language="en", task="transcribe", num_threads=4)

    voice = build.PiperVoice.load(str(build.ROOT / "voice" / "en-us-lessac-medium.onnx"))
    data = build.build_data()
    # single words are hard for a recognizer out of context, so each word clip is heard after
    # "The word is", spoken by the same voice
    build.synth(voice, "The word is", "word")
    carrier = captured["x"]
    problems, checked = [], 0
    for name, (text, kind) in sorted(build.clips(data).items()):
        if args.only and not re.search(args.only, name):
            continue
        build.synth(voice, text, kind)
        x, rate = captured["x"], captured["rate"]
        checked += 1
        dur = len(x) / rate
        peak = np.abs(x).max()
        rms = np.sqrt((x ** 2).mean())
        if peak >= 32500:
            problems.append((name, "clipping"))
        if name.startswith("p_"):
            lo, hi = (0.4, 0.9) if kind == "hold" else (0.06, 0.4)
            if not lo <= dur <= hi:
                problems.append((name, f"length {dur:.2f}s outside {lo}-{hi}s"))
            if kind == "hold":
                q = len(x) // 4
                a, b = np.sqrt((x[q:2 * q] ** 2).mean()), np.sqrt((x[3 * q:] ** 2).mean())
                if b < a * 0.35:
                    problems.append((name, f"fades out (tail {b / a:.0%} of body)"))
            continue
        # recognizer check for anything with words in it
        pad = np.zeros(int(rate * 0.3))
        if name.startswith("w_"):
            y = np.concatenate([pad, carrier, np.zeros(int(rate * 0.12)), x, pad]) / 32768
        else:
            y = np.concatenate([pad, x, pad]) / 32768
        s = rec.create_stream()
        s.accept_waveform(rate, y.astype(np.float32))
        rec.decode_stream(s)
        heard = norm(s.result.text)
        if name.startswith("w_"):
            heard = [w for w in heard if w not in ("the", "word", "is")] or heard[-1:]
        want = norm(text)
        miss = [w for w in want if w not in heard and not (SAME.get(w, set()) & set(heard))]
        if name.startswith("w_"):
            ok = not miss
        else:  # sentences and prompts: allow one slip from the recognizer
            ok = len(miss) <= max(0, len(want) // 6)
        if not ok:
            problems.append((name, f'said "{text}" but heard "{s.result.text.strip()}"'))
        if rms < 1000:
            problems.append((name, f"quiet (rms {rms:.0f})"))
    print(f"checked {checked} clips, {len(problems)} problems")
    for n, p in problems:
        print(f"  {n}: {p}")


if __name__ == "__main__":
    main()

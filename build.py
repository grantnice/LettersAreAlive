"""Build Blend Farm: generate voice clips with Piper and bake everything into one HTML file.

    pip install piper-tts lameenc numpy
    python3 build.py            # writes index.html (artifact) and play.html (open locally / iPad)

The curriculum lives in CURRICULUM below. Rules it follows:
  * Letters make only their most common sound until level 14 (no s=/z/, no schwa "a"/"the").
  * Each level adds a few sounds; every word in a level is decodable with sounds taught so far.
  * Words are written as tiles: "sh-i-p" or "c-a:A-k-e:_" (grapheme:phonemeKey, "_" = silent).
    Words without dashes are one tile per letter.
"""
import base64, io, json, re, wave
from pathlib import Path

import lameenc
import numpy as np
from piper import PiperVoice, SynthesisConfig

ROOT = Path(__file__).parent

# phoneme key -> IPA fed to Piper as raw phonemes. Continuous sounds are stretched so they can be
# held while blending; stop sounds get the smallest possible vowel so "b" is /b/, not "buh".
PHONEMES = {
    "m": "mː", "s": "sː", "f": "fː", "n": "nː", "l": "lː", "r": "ɹː",
    "v": "vː", "z": "zː", "sh": "ʃː", "th": "θː",
    "a": "æː", "e": "ɛː", "i": "ɪː", "o": "ɑː", "u": "ʌː",
    "b": "bə", "k": "kə", "d": "də", "g": "ɡə", "h": "hə", "j": "dʒə", "p": "pə", "t": "tə",
    "w": "wə", "y": "jə", "ch": "tʃə", "x": "ks", "qu": "kwə",
    "A": "eɪ", "E": "iː", "I": "aɪ", "O": "oʊ", "U": "juː",
    "oo": "uː", "OO": "ʊ", "ow": "aʊ", "ar": "ɑːɹ", "or": "ɔːɹ", "er": "ɜːɹ",
}
CONTINUOUS = {"m", "s", "f", "n", "l", "r", "v", "z", "sh", "th", "a", "e", "i", "o", "u"}
HOLD_SECONDS = 0.65

# graphemes whose default phoneme key differs from their spelling
DEFAULT_KEY = {"c": "k", "ck": "k", "ss": "s", "ll": "l", "ff": "f", "gg": "g",
               "ee": "E", "ea": "E", "oa": "O", "ai": "A", "ir": "er"}

LEVELS = [
    dict(title="M, A, S, T", sounds=["m", "a", "s", "t"],
         words=["am", "at", "sat", "mat"], note="Continuous sounds first; blend by holding each sound."),
    dict(title="P and I", sounds=["p", "i"],
         words=["map", "sip", "sit", "pit", "pat", "sap", "it", "mat", "sat"]),
    dict(title="N and O", sounds=["n", "o"],
         words=["nap", "pan", "pin", "man", "mop", "pot", "top", "on", "not", "pop", "mom", "tin", "nip"]),
    dict(title="C and D", sounds=["c", "d"],
         words=["cat", "cap", "can", "cot", "dad", "dot", "sad", "mad", "pad", "cod", "dip", "pod", "cop", "did"]),
    dict(title="G and H", sounds=["g", "h"],
         words=["pig", "dig", "hat", "hot", "hog", "tag", "gas", "dog", "hop", "him", "hip", "hid", "got"]),
    dict(title="U and B", sounds=["u", "b"],
         words=["bus", "bug", "sun", "nut", "cup", "hug", "hut", "bat", "tub", "cut", "bun", "gum", "mud", "cub", "bib"]),
    dict(title="F and E", sounds=["f", "e"],
         words=["fan", "fin", "fun", "fig", "bed", "pen", "ten", "net", "hen", "pet", "fed", "beg"]),
    dict(title="R and L", sounds=["r", "l"],
         words=["rat", "run", "red", "rug", "leg", "log", "lip", "lid", "rag", "rib", "lap", "lot"]),
    dict(title="K, J, W, V", sounds=["k", "j", "w", "v"],
         words=["kid", "kit", "jam", "jet", "jug", "web", "wig", "wet", "van", "vet", "win", "jog"]),
    dict(title="X, Y, Z, Qu", sounds=["x", "y", "z", "qu"],
         words=["fox", "box", "six", "ox", "yak", "yum", "zip", "zap", "qu-i-z", "yes", "wax", "mix", "zig"]),
    dict(title="Sh, Ch, Th, Ck", sounds=["sh", "ch", "th", "ck"],
         words=["f-i-sh", "sh-i-p", "sh-o-p", "d-i-sh", "d-u-ck", "s-o-ck", "r-o-ck", "l-o-ck",
                "ch-i-ck", "ch-i-p", "ch-i-n", "b-a-th", "m-o-th", "b-a-ck"]),
    dict(title="Ending blends", sounds=[],
         words=["ant", "hand", "tent", "nest", "milk", "lamp", "belt", "gift", "pond", "sand", "mask",
                "e-gg", "b-e-ll", "d-o-ll", "h-i-ll"], note="Four-sound words; double letters make one sound."),
    dict(title="Starting blends", sounds=[],
         words=["frog", "crab", "drum", "flag", "sled", "clap", "swim", "plug", "stop", "grin", "crib",
                "s-n-a-ck", "trip", "s-k-u-n-k"]),
    dict(title="Magic E", sounds=["a_e:A", "i_e:I", "o_e:O", "u_e:U"],
         words=["c-a:A-k-e:_", "b-i:I-k-e:_", "k-i:I-t-e:_", "b-o:O-n-e:_", "c-u:U-b-e:_", "h-o:O-m-e:_",
                "c-o:O-n-e:_", "l-a:A-k-e:_", "n-i:I-n-e:_", "f-i:I-v-e:_", "g-a:A-t-e:_", "r-o:O-p-e:_",
                "t-a:A-p-e:_"], note="First secondary sounds: the silent e makes the vowel say its name."),
    dict(title="Vowel teams", sounds=["ee", "ea", "oa", "ai", "oo"],
         words=["sh-ee-p", "b-ee", "t-r-ee", "f-ee-t", "s-ea-l", "g-oa-t", "b-oa-t", "r-ai-n", "s-n-ai-l",
                "t-r-ai-n", "m-oo-n", "r-oa-d", "qu-ee-n"]),
    dict(title="Bossy R and Ow", sounds=["ar", "or", "er", "ir", "ow"],
         words=["s-t-ar", "c-ar", "b-ar-n", "sh-ar-k", "f-ar-m", "c-or-n", "f-or-k", "h-or-n", "c-ow",
                "ow-l", "b-ir-d", "f-er-n", "s-t-or-m", "g-ir-l"]),
    dict(title="Reading sentences", sounds=[], words=[], hearts=["the", "a", "is", "I", "see", "has"],
         note="Heart words are learned by heart; everything else is decodable."),
]

PICTURES = {
    "map": "🗺️", "tap": "🚰", "sip": "🥤", "sit": "🪑", "nap": "😴", "pan": "🍳", "pin": "📌", "man": "👨",
    "mop": "🧹", "pot": "🍲", "mom": "👩", "cat": "🐱", "cap": "🧢", "can": "🥫", "dot": "⚫", "sad": "😢",
    "mad": "😠", "cod": "🐟", "pod": "🫛", "cop": "👮", "dad": "👨‍🍼", "pig": "🐷", "dig": "⛏️", "hat": "👒",
    "hot": "🔥", "hog": "🐗", "tag": "🏷️", "gas": "⛽", "dog": "🐶", "bus": "🚌", "bug": "🐛", "sun": "☀️",
    "nut": "🥜", "cup": "☕", "hug": "🤗", "hut": "🛖", "bat": "🦇", "tub": "🛁", "cut": "✂️", "bun": "🍞",
    "gum": "🍬", "fan": "🪭", "fun": "🎉", "bed": "🛏️", "pen": "🖊️", "ten": "🔟", "net": "🥅", "hen": "🐔",
    "fox": "🦊", "rat": "🐀", "run": "🏃", "red": "🟥", "leg": "🦵", "log": "🪵", "lip": "👄", "kid": "🧒",
    "jam": "🍓", "jet": "✈️", "web": "🕸️", "wig": "💇", "wet": "💦", "van": "🚐", "win": "🏆", "box": "📦",
    "six": "6️⃣", "ox": "🐂", "yak": "🐃", "yum": "😋", "zip": "🤐", "zap": "⚡", "yes": "✅", "fish": "🐟",
    "ship": "🚢", "dish": "🍽️", "duck": "🦆", "sock": "🧦", "rock": "🪨", "lock": "🔒", "chick": "🐤",
    "chip": "🍟", "bath": "🛀", "moth": "🦋", "ant": "🐜", "hand": "✋", "tent": "⛺", "nest": "🪺",
    "milk": "🥛", "lamp": "💡", "gift": "🎁", "sand": "🏖️", "mask": "🎭", "egg": "🥚", "bell": "🔔",
    "doll": "🪆", "hill": "⛰️", "frog": "🐸", "crab": "🦀", "drum": "🥁", "flag": "🚩", "sled": "🛷",
    "clap": "👏", "swim": "🏊", "plug": "🔌", "stop": "🛑", "grin": "😁", "snack": "🍪", "skunk": "🦨",
    "cake": "🎂", "bike": "🚲", "kite": "🪁", "bone": "🦴", "cube": "🧊", "home": "🏠", "cone": "🍦",
    "lake": "🏞️", "nine": "9️⃣", "five": "5️⃣", "gate": "🚧", "tape": "📼", "sheep": "🐑", "bee": "🐝",
    "tree": "🌳", "feet": "🦶", "seal": "🦭", "goat": "🐐", "boat": "⛵", "rain": "🌧️", "snail": "🐌",
    "train": "🚆", "moon": "🌙", "road": "🛣️", "queen": "👸", "book": "📖", "star": "⭐", "car": "🚗",
    "barn": "🛖", "shark": "🦈", "farm": "🚜", "corn": "🌽", "fork": "🍴", "horn": "📯", "cow": "🐄",
    "owl": "🦉", "bird": "🐦", "fern": "🌿", "storm": "⛈️", "girl": "👧",
}
# (picture word for barn collides with hut; keep only one of them in any choice set)
SAME_PICTURE = [["hut", "barn"], ["cod", "fish"], ["man", "dad"]]

SENTENCES = [
    ("the cat is on the bed", "🐱🛏️", "🐶🛏️"),
    ("the dog can dig", "🐶⛏️", "🐱⛏️"),
    ("I see a pig in the sun", "🐷☀️", "🐷🌧️"),
    ("the fox is in a box", "🦊📦", "🐶📦"),
    ("I see a bug on a log", "🐛🪵", "🐛🛏️"),
    ("the frog is on a rock", "🐸🪨", "🐸🛏️"),
    ("a duck is in the pond", "🦆💧", "🐔💧"),
    ("the sheep can see the moon", "🐑🌙", "🐑☀️"),
    ("a goat is on the hill", "🐐⛰️", "🐄⛰️"),
    ("the cow is in the barn", "🐄🛖", "🐖🛖"),
    ("I see a kite and a tree", "🪁🌳", "🚲🌳"),
    ("the cat has a hat", "🐱👒", "🐱🧦"),
    ("a bee is on the corn", "🐝🌽", "🐛🌽"),
    ("the hen sat on an egg", "🐔🥚", "🦆🥚"),
    ("the queen has a crown", "👸👑", "👸🎩"),
    ("a seal can swim", "🦭🏊", "🐄🏊"),
    ("the girl is on a bike", "👧🚲", "👧🛷"),
    ("I see a star and the moon", "⭐🌙", "⭐☀️"),
]

# decodable words that appear only in sentences (anything else in a sentence is a heart word)
SENTENCE_WORDS = ["in", "and", "an", "c-r-ow-n"]

# one friend joins the farm for each mastered level
ANIMALS = [
    ("🐑", "Gotland sheep", "baa baa"), ("🐔", "Swedish flower hen", "cluck cluck"),
    ("🐄", "Fjällko mountain cow", "moo"), ("🐖", "Linderöd pig", "oink oink"),
    ("🐐", "Göinge goat", "maa maa"), ("🐴", "Gotland pony", "neigh"),
    ("🦆", "Swedish blue duck", "quack quack"), ("🐈", "Barn cat", "meow"),
    ("🐕", "Swedish Vallhund", "woof woof"), ("🐇", "Gotland rabbit", "thump thump"),
    ("🪿", "Skåne goose", "honk honk"), ("🐸", "Pond frog", "ribbit ribbit"),
    ("🐝", "Honey bees", "buzz buzz"), ("🦔", "Hedgehog", "snuffle snuffle"),
    ("🐎", "Dala horse", "clip clop"), ("🦌", "Reindeer", "snort snort"),
    ("🫎", "Moose", "hrrumph"),
]

# listening warm-up (no letters): hear two words, pick the pictures in that order ("dog fish" vs "fish dog")
PAIRS = [("dog", "fish"), ("cow", "bee"), ("frog", "tree"), ("sun", "hat"), ("cat", "boat"), ("pig", "cake"),
         ("duck", "egg"), ("sheep", "moon"), ("fox", "box"), ("goat", "star"), ("bus", "sock"), ("hen", "nest")]

PROMPTS = {
    "listen2": "Listen. Which one did you hear?",
    "find": "Find the egg that says",
    "hop": "Slide the frog across. Say each sound.",
    "fast": "Now say it fast! Which one is it?",
    "say": "Say it out loud. Then ring the bell.",
    "spell": "Spell it!",
    "match": "Read it. Does the picture match?",
    "again": "Oops! Let's sound it out again.",
    "tryagain": "Try again!",
    "listen": "Listen.",
    "yes1": "Yes!", "yes2": "Great reading!", "yes3": "You got it!", "yes4": "Super!", "yes5": "Wonderful!",
    "passed": "You mastered it! Great work!",
    "almost": "So close! Let's practice a little more.",
    "levelup": "Level up! A new friend is coming to the farm!",
    "heart": "This is a heart word. Learn it by heart.",
}


def tiles(spec):
    parts = spec.split("-") if "-" in spec else list(spec)
    out = []
    for part in parts:
        g, _, key = part.partition(":")
        key = key or DEFAULT_KEY.get(g, g)
        out.append({"g": g, "p": None if key == "_" else key})
    return out


def sound(spec):
    g, _, key = spec.partition(":")
    return {"g": g, "p": key or DEFAULT_KEY.get(g, g)}


def word_of(spec):
    return re.sub(r":[^-]*", "", spec).replace("-", "")


def validate(levels):
    """Every word must use only sounds taught at or before its level."""
    known = set()
    for n, lv in enumerate(levels, 1):
        known |= {sound(s)["p"] for s in lv["sounds"]}
        for spec in lv["words"]:
            for t in tiles(spec):
                if t["p"] and t["p"] not in known:
                    raise SystemExit(f"level {n}: '{word_of(spec)}' uses untaught sound {t['p']}")
                if t["p"] and t["p"] not in PHONEMES:
                    raise SystemExit(f"no phoneme audio for {t['p']}")


def build_data():
    validate(LEVELS)
    levels = []
    for n, lv in enumerate(LEVELS, 1):
        emoji, breed, noise = ANIMALS[n - 1]
        levels.append({
            "n": n, "title": lv["title"], "note": lv.get("note", ""),
            "sounds": [sound(s) for s in lv["sounds"]],
            "words": [{"w": word_of(s), "t": tiles(s)} for s in lv["words"]],
            "hearts": lv.get("hearts", []),
            "animal": {"e": emoji, "name": breed, "noise": noise},
        })
    sentences = [{"s": s, "yes": a, "no": b} for s, a, b in SENTENCES]
    extra = [{"w": word_of(s), "t": tiles(s)} for s in SENTENCE_WORDS]
    all_words = {w["w"] for lv in levels for w in lv["words"]} | {w["w"] for w in extra}
    hearts = {h.lower() for h in LEVELS[-1]["hearts"]}
    for s, _, _ in SENTENCES:
        for w in s.lower().split():
            if w not in all_words and w not in hearts:
                raise SystemExit(f"sentence word '{w}' is neither decodable nor a heart word")
    return {"levels": levels, "pictures": PICTURES, "extraWords": extra, "pairs": PAIRS, "continuous": sorted(CONTINUOUS), "samePicture": SAME_PICTURE,
            "sentences": sentences, "prompts": list(PROMPTS)}


def clips(data):
    """name -> (text, kind) where kind is hold, stop or word"""
    out = {f"p_{k}": (f"[[{ipa}]]", "hold" if k in CONTINUOUS else "stop") for k, ipa in PHONEMES.items()}
    words = {w["w"] for lv in data["levels"] for w in lv["words"]}
    for s in SENTENCES:
        words |= set(s[0].lower().split())
    words |= {"the", "a", "is", "i", "see"} | {w for pair in PAIRS for w in pair}
    for w in words:
        out[f"w_{w.lower()}"] = (w, 'word')
    for i, s in enumerate(SENTENCES):
        out[f"s_{i}"] = (s[0].capitalize() + ".", "word")
    for k, text in PROMPTS.items():
        out[f"q_{k}"] = (text, "word")
    for i, (_, breed, noise) in enumerate(ANIMALS):
        out[f"a_{i}"] = (f"{noise.capitalize()}! I'm a {breed}.", "word")
    return out


def stretch(x, rate, seconds):
    """Lengthen a steady sound (mmm, sss, aaa) by looping its middle with crossfades."""
    xf = int(rate * 0.03)
    mid = x[len(x) * 3 // 10: len(x) * 7 // 10]
    if len(mid) < xf * 3:
        return x
    head, tail = x[: len(x) * 3 // 10], x[len(x) * 7 // 10:]
    out = np.concatenate([head, mid])
    ramp = np.linspace(0, 1, xf)
    while len(out) + len(tail) < seconds * rate:
        out[-xf:] = out[-xf:] * (1 - ramp) + mid[:xf] * ramp
        out = np.concatenate([out, mid[xf:]])
    out[-xf:] = out[-xf:] * (1 - ramp) + tail[:xf] * ramp if len(tail) >= xf else out[-xf:]
    return np.concatenate([out, tail[xf:]])


def synth(voice, text, kind):
    is_phoneme = kind in ("hold", "stop")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=2.0 if kind == "hold" else 1.1))
    buf.seek(0)
    with wave.open(buf) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    # trim silence so sounds line up tightly when hopping across the pads
    env = np.convolve(np.abs(x.astype(float)), np.ones(220) / 220, "same")
    loud = np.where(env > (600 if is_phoneme else 350))[0]
    if len(loud):
        x = x[max(0, loud[0] - 330): loud[-1] + 660]
    x = x.astype(float)
    if kind == "hold":
        x = stretch(x, rate, HOLD_SECONDS)
    fade = min(330, len(x) // 4)
    x[-fade:] *= np.linspace(1, 0, fade)
    enc = lameenc.Encoder()
    enc.set_bit_rate(48)
    enc.set_in_sample_rate(rate)
    enc.set_channels(1)
    enc.set_quality(2)
    mp3 = enc.encode(x.astype(np.int16).tobytes()) + enc.flush()
    return base64.b64encode(mp3).decode()


def main():
    data = build_data()
    cache_file = ROOT / ".audio-cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    voice = None
    audio = {}
    for name, (text, kind) in sorted(clips(data).items()):
        key = f"{text}|{kind}"
        if key not in cache:
            voice = voice or PiperVoice.load(str(ROOT / "voice" / "en-us-lessac-medium.onnx"))
            cache[key] = synth(voice, text, kind)
        audio[name] = cache[key]
    used = {f"{t}|{k}" for t, k in clips(data).values()}
    cache_file.write_text(json.dumps({k: v for k, v in cache.items() if k in used}))

    page = (ROOT / "game.template.html").read_text()
    page = page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    page = page.replace("/*__AUDIO__*/null", json.dumps(audio))
    (ROOT / "index.html").write_text(page)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,user-scalable=no">'
            '<meta name="apple-mobile-web-app-capable" content="yes"></head><body>')
    (ROOT / "play.html").write_text(head + page + "</body></html>")
    words = sum(len(l["words"]) for l in data["levels"])
    print(f"{len(data['levels'])} levels, {words} words, {len(audio)} clips, "
          f"{len(page) / 1e6:.1f} MB page")


if __name__ == "__main__":
    main()

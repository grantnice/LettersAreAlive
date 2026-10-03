# CLAUDE.md

Letters Are Alive is a phonics blending game for a 4-year-old, played on an iPad. It is set in an Alpine valley (Sound of Music) with Swedish farm touches. The game is published on GitHub Pages at https://grantnice.github.io/LettersAreAlive/ (repo root, `main` branch, `.nojekyll`).

## Files

- `build.py` holds the curriculum (17 levels, words, pictures, sentences, animal roster, prompts). It generates the audio and bakes everything into `index.html` (the standalone game) and `artifact.html` (the claude.ai preview, which has no 3D).
- `game.template.html` holds all the game code and CSS: one file, vanilla JS, no build step.
- `world3d.js` is the 3D world (three.js r169 via a jsdelivr importmap). It provides the backdrops behind every screen and the live 3D farm. Assets are in `assets/`: Quaternius CC0 animals converted to meshopt GLBs, plus `nature/nature.glb`. If WebGL or the module is unavailable, the game falls back to 2D scenery.
- `voice/recordings.json` holds the parent's own letter-sound recordings, exported from the in-game studio. They override the TTS for those sounds.
- `docs/TENETS.md` contains the teaching principles (Mentava, Engelmann, mastery learning). It is rendered in-app under the grown-ups corner.
- `tools/playtest.py` is a headless playthrough of every activity. `tools/qa_audio.py` is a Whisper check of the clips. `tools/world3d-demo.html` is the 3D demo.

## Build and test

```bash
pip install piper-tts lameenc numpy markdown sherpa-onnx playwright
# voice model (not committed): voice/en-us-lessac-medium.onnx(.json) from
#   https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz
# optional, picks clear word takes: voice/sherpa-onnx-whisper-small.en from the sherpa-onnx asr-models release
python3 build.py
python3 tools/playtest.py --levels 1,4,11,14,17   # must report 0 issues
```

Rebuilding audio from scratch takes about 11 minutes with the Whisper take-picker. `.audio-cache.json` (gitignored) makes rebuilds fast. In Playwright, call `void runStage(...)`: `runStage` returns a promise that resolves only when the stage ends. 3D tests need http (`python3 -m http.server`) plus Chromium flags `--use-angle=swiftshader --enable-unsafe-swiftshader` and the agent proxy.

## Teaching rules (don't break these)

- Use only the most common sound of each letter until Level 14 (Magic E). `build.py` refuses to build if a word uses an untaught sound.
- Introduce continuous sounds first. Stop sounds come only at the ends of words early on.
- Mastery-gate every step at about 90% first-try correct, in order: listen → sounds → blend → spell → check. Mix in spaced, interleaved review. A missed item returns 3 turns later (model → lead → test).
- Read left to right with one finger: slide the frog or ladybug under the letters. Ignore stray touches and second fingers. Show picture choices only after decoding, with foils that differ by one sound.
- Rewards follow work, never interrupt it: stars, animals per level, the farm shop, and the rewards jar (video time, plus real-world treats a parent marks as given).

## Conventions

- The parent prefers their own recorded letter sounds. Keep the recording studio and its export/import working.
- The grown-ups corner opens with an addition-question gate. Developer mode (on by default) opens every level.
- Stay theme-consistent: Falu-red barn, Swedish blue and yellow accents, Andika font for anything a child reads (single-story a and g).

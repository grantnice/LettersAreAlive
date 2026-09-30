# Letters Are Alive

A phonics game for learning to blend letter sounds. It's set in an Alpine valley with a Falu-red Swedish barn, and each level he masters brings a new Swedish farm animal to his farm. There are no videos. Every minute is spent practicing: hearing sounds, sliding a finger under letters from left to right to blend them, spelling, and reading.

## Play it

- **Hosted (needed for the microphone):** serve `index.html` over https, e.g. GitHub Pages (Settings → Pages → deploy from `main`, root) or Netlify, then on the iPad open it in Safari and Share → *Add to Home Screen*.
- **On a computer:** open `index.html` in any browser. The whole game is one offline file with all the audio built in. Progress is saved in that browser.
- **Tip:** turn on iPad *Guided Access* (Settings → Accessibility) so he can't leave the app.

## How it teaches

The design principles are in [`docs/TENETS.md`](docs/TENETS.md). In short:

1. **Only the most common sound for each letter** until Level 14. There is no s-as-/z/, and no "the" or "a" until the sentence level, where they are taught as heart words.
2. **Continuous sounds come first** (m, a, s), and stop sounds appear only at the end of words at first.
3. **A few new sounds per level.** Every word in a level can be decoded with sounds already taught. `build.py` checks this and refuses to build if a word breaks the rule.
4. **Each level is a ladder of mastery steps:** Listening Game (Level 1 only) → Egg Sounds → Lily Pad Hop → Meadow Spelling → Mastery Climb. Each step needs about 90% right on the first try, and the next step stays locked until then.
5. **Review is spaced and interleaved.** Earlier sounds and words are mixed into every session, and missed items come back more often.
6. **Missed items come back later.** A missed item is re-asked three turns later, following Engelmann's error-correction pattern.
7. **Reading is left to right, one finger.** In Lily Pad Hop he drags the frog under the letters and each letter sounds as the frog reaches it. It can't skip ahead. In Story Barn a ladybug slides under the words. Only the first finger counts, so palms and stray taps elsewhere are ignored. Leaving a lesson requires pressing and holding the mountain button.
8. **No guessing from pictures.** Picture choices appear only after the word has been sounded out. The wrong-answer pictures differ from the right one by one sound (cat, cap, can).

## Rewards

- **Stars:** 2 for a first-try correct answer, 1 after a retry, and bonuses for mastering a step or a level.
- **Animals:** 17 animated 3D animals (Quaternius, CC0), one per level: Gotland sheep, Fjällko cow, Linderöd pig, Gotland pony, donkey, farm dog, llama, alpaca, roe deer, red fox, pug, bull, husky, white horse, red deer stag, grey wolf and a zebra. They wander the farm and talk when tapped.
- **Farm shop:** spend stars on flowers, fir trees, hay, a tractor and more.
- **Sing Along:**
  - "Little Frog on a Log" (to the tune of Twinkle Twinkle)
  - "Old MacDonald's Swedish Farm", which sings about the animals he has earned
  - "Mountain Word Ladder": he reads a word on each note of the do-re-mi scale, up and down the mountain
- **Five-in-a-row combos** play a yodel and confetti.

## Microphone practice (optional)

Turn it on in the grown-ups corner. It needs the game opened from an https address, not the claude.ai preview.

- **Voice line:** in Lily Pad Hop the pads stay quiet and he says the sounds himself while sliding the frog. A live voice line shows his voice. If it drops out between sounds for too long, the frog splashes back and he tries again ("keep your voice on": *mmmaaat*, not *m… a… t*). The game gets a little more patient after each splash, and a splash never counts as a miss.
- **Word check:** after the slide he taps 🎤 and says the word fast. The browser's speech recognizer shows what it heard. It is encouragement only and never marks him wrong.

## Rewards

Every star he earns also goes into a savings jar. He picks what he's saving for on the map (🎬 video time, 🍪 a cookie from the cookie store, 🤸 the trampoline park, or anything you add). Video rewards play right away: 60 seconds of a SpaceX or Blippi video (or your own links), continuing where the last turn stopped. Real-world rewards show a "Show a grown-up!" screen and wait in the grown-ups corner until you mark them given. Names, emoji and star costs are all editable.

## Your own voice for the letter sounds

Grown-ups corner → 🎙️ Record my letter sounds walks through every letter sound. Recordings are trimmed and leveled automatically and replace the computer voice everywhere. They're kept in the browser; **Save recordings file** exports them. Put that file at `voice/recordings.json` and run `python3 build.py` to build them into the game for every device.

## Grown-ups corner

Tap ⚙️ on the map and answer a quick addition question (a grown-ups check). It shows accuracy for each sound, the words to watch, recent sessions, level placement, and a reset.

## Building

```bash
pip install piper-tts lameenc numpy
# voice model (not committed; ~60 MB):
curl -L https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz | tar xz -C voice
python3 build.py   # writes index.html (the game) and artifact.html (for a claude.ai preview)
```

Optional but recommended: download `sherpa-onnx-whisper-small.en` from the [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) into `voice/` (or point `LAA_ASR` at it). The build then records up to 9 takes of each word and keeps the first one Whisper hears correctly.

### Quality checks

- `python3 tools/playtest.py` plays every step of several levels in a headless browser, answering correctly, and reports stuck activities, bad answer choices, and page errors.
- `python3 tools/qa_audio.py` regenerates clips and checks them: Whisper transcription for words, sentences and prompts, and length, loudness and clipping checks for letter sounds.

The curriculum (levels, words, pictures, sentences, animals) lives at the top of `build.py`. The game code and art live in `game.template.html`. Words are spoken inside a short sentence ("Say, sheep.") and cut out using the voice's phoneme timings, because this voice says lone words poorly. Letter sounds are made with the [Piper](https://github.com/rhasspy/piper) neural voice from IPA phonemes, so a sound like /b/ is a clipped /b/ and not "bee" or "buh". Continuous sounds are stretched so they can be held while blending. The Lessac voice's dataset is licensed for non-commercial use, which covers family use.

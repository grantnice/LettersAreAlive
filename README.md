# Letters Are Alive

A phonics game for learning to blend letter sounds. It's set in an Alpine valley with a Falu-red Swedish barn, and each level he masters brings a new Swedish farm animal to his farm. There are no videos. Every minute is spent practicing: hearing sounds, sliding a finger under letters from left to right to blend them, spelling, and reading.

## Play it

- **On the iPad:** open `play.html` in Safari, then Share → *Add to Home Screen*. The whole game is one offline file with all the audio built in. Progress is saved in that browser.
- **On a computer:** open `play.html` in any browser.
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
- **Animals:** 17 Swedish farm animals, one per level: Gotland sheep, flower hen, Fjällko mountain cow, Linderöd pig, and more. They wander the farm and talk when tapped.
- **Farm shop:** spend stars on flowers, fir trees, hay, a tractor and more.
- **Sing Along:**
  - "Little Frog on a Log" (to the tune of Twinkle Twinkle)
  - "Old MacDonald's Swedish Farm", which sings about the animals he has earned
  - "Mountain Word Ladder": he reads a word on each note of the do-re-mi scale, up and down the mountain
- **Five-in-a-row combos** play a yodel and confetti.

## Grown-ups corner

Press and hold ⚙️ on the map for 1.5 seconds. It shows accuracy for each sound, the words to watch, recent sessions, level placement, and a reset.

## Building

```bash
pip install piper-tts lameenc numpy
# voice model (not committed; ~60 MB):
curl -L https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz | tar xz -C voice
python3 build.py   # writes index.html (for hosting as an artifact) and play.html (standalone)
```

The curriculum (levels, words, pictures, sentences, animals) lives at the top of `build.py`. The game code and art live in `game.template.html`. Letter sounds are made with the [Piper](https://github.com/rhasspy/piper) neural voice from IPA phonemes, so a sound like /b/ is a clipped /b/ and not "bee" or "buh". Continuous sounds are stretched so they can be held while blending. The Lessac voice's dataset is licensed for non-commercial use, which covers family use.

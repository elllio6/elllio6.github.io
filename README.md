```markdown
# Image Guess & Train — Play-to-Teach AI (Streamlit) — Voice + "Bad at Start"

This version adds:
- Text-to-speech: the app will speak the AI's top guess (best-effort) using gTTS.
- Intentional poor initial performance: while the player is at level 1 the AI will often make wrong or low-confidence guesses to encourage playing puzzles and teaching it.
- When you solve puzzles and fine-tune the model, the AI becomes better (because it uses your data).

Requirements
- Python 3.8+
- Install requirements:
```
pip install -r requirements.txt
```

Notes about TTS
- This app uses gTTS (Google Text-to-Speech) to create short mp3 audio for the AI's spoken guess.
- gTTS needs internet access to work. If gTTS isn't available the app will still function but without voice.

Run
```
streamlit run streamlit_app.py
```

How the "bad at start" behavior works
- While you are at level 1 the predictor intentionally adds noise:
  - Replaces the top guess with a random common object with high probability.
  - Lowers confidences to make the AI sound uncertain.
- When you collect labeled images and fine-tune (by solving puzzles and saving the labeled images), the AI will use your custom model and stop being intentionally bad.

Files
- `streamlit_app.py` — main Streamlit UI (now with voice).
- `model_utils.py` — prediction, saving images, and fine-tuning logic (predictor accepts player_level).
- `user_data/` — where labeled images are stored.
- `user_model.h5` and `classes.json` — saved model & mapping after fine-tuning.

Next ideas
- Add an option to toggle "make AI intentionally bad" (for practice).
- Use a local TTS fallback (pyttsx3) for offline speech.
- Make the "badness" scale gradually with levels rather than just level 1.
```

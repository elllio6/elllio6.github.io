```markdown
# FunnyImageAI — Always a little bad (Streamlit)

Overview
- Upload images and the AI will try to guess them.
- The AI is intentionally always "a little bad" (funny behavior) — no puzzles or leveling.
- The app has three voice scripts (Shy, Sarcastic, Confident) that speak the top guess.
- You can mark "AI correct" to get points, or "AI incorrect" and save a labeled image to your dataset.
- Optionally fine-tune a small model on your collected images; note the AI will still keep a small playful "badness" after training (per your request).

Files
- `streamlit_app.py` — main Streamlit UI (no puzzles, always slightly bad).
- `model_utils.py` — predictor, saving images, fine-tuning logic.
- `user_data/` — where labeled images are stored (created automatically).
- `user_model.h5` and `classes.json` — saved model & mapping after fine-tuning.
- `requirements.txt` — Python packages.

Requirements
- Python 3.8+
- Install requirements:
```
pip install -r requirements.txt
```

Run
```
streamlit run streamlit_app.py
```

Audio notes
- The app supports `pyttsx3` (offline) and `gTTS` (online). Pick a backend in the sidebar.
- If no TTS backend is available the UI still works (text-only).

Important behavior choices you requested
- Puzzles removed entirely.
- The AI is always slightly "bad" for a fun / humorous experience (applies to both fallback ImageNet and a fine-tuned user model).

Next ideas (optional)
- Add an intensity slider to control how "bad" the AI remains.
- Add more voice scripts or allow custom phrases.
- Add export/import for dataset and model.
```

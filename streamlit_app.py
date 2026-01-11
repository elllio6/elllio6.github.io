import streamlit as st
from PIL import Image
import io
import os
import json
import numpy as np
from model_utils import Predictor, save_user_image, fine_tune_on_user_data, ensure_user_dirs

# Optional TTS
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# Constants
STATE_FILE = "state.json"
USER_DATA_DIR = "user_data"
MODEL_FILE = "user_model.h5"
CLASSES_FILE = "classes.json"

# Ensure directories exist
ensure_user_dirs(USER_DATA_DIR)

# Load predictor (will use ImageNet decode if no custom model)
predictor = Predictor(user_model_path=MODEL_FILE, classes_path=CLASSES_FILE)

# Persistent state: points and unlocked level
if "state" not in st.session_state:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            st.session_state.state = json.load(f)
    else:
        st.session_state.state = {"points": 0, "level": 1}
def save_state():
    with open(STATE_FILE, "w") as f:
        json.dump(st.session_state.state, f)

st.title("Image Guess & Train — Play to teach the AI (with voice)")

# Sidebar: score and actions
st.sidebar.header("Player")
st.sidebar.write(f"Points: {st.session_state.state['points']}")
st.sidebar.write(f"Level: {st.session_state.state['level']}")
if st.sidebar.button("Reset score"):
    st.session_state.state = {"points": 0, "level": 1}
    save_state()
    st.sidebar.success("Score reset")

st.header("Upload an image for the AI to guess (audio will speak the top guess)")
uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    # Predict (pass player level so predictor can intentionally be 'bad' at the start)
    with st.spinner("AI is guessing..."):
        preds = predictor.predict_pil(image, top=3, player_level=st.session_state.state['level'])
    st.write("AI guesses (top 3):")
    for i,(label,score) in enumerate(preds):
        st.write(f"{i+1}. **{label}** — {score:.2%}")

    # Text-to-speech for the top guess (best-effort)
    top_label = preds[0][0]
    tts_status = None
    if TTS_AVAILABLE:
        try:
            tbuf = io.BytesIO()
            # short spoken phrase
            phrase = f"Hmm, I think this is {top_label}."
            gTTS(text=phrase, lang="en").write_to_fp(tbuf)
            tbuf.seek(0)
            st.audio(tbuf.read(), format="audio/mp3")
            tts_status = "ok"
        except Exception as e:
            tts_status = f"failed: {e}"
    else:
        st.info("Text-to-speech not available (install gTTS to enable).")

    cols = st.columns(3)
    if cols[0].button("AI correct"):
        st.session_state.state['points'] += 1
        st.success("Point awarded!")
        # Level formula: every 5 points -> new level
        st.session_state.state['level'] = 1 + st.session_state.state['points'] // 5
        save_state()
    if cols[1].button("AI incorrect"):
        st.info("Provide the correct label to teach the AI")
        correct_label = st.text_input("Correct label (exact):")
        if st.button("Save labeled image"):
            # save into user dataset and optionally fine-tune
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)
            save_user_image(buf, correct_label, USER_DATA_DIR)
            st.success(f"Saved image as label: {correct_label}")
    if cols[2].button("Skip"):
        st.write("Skipped")

st.markdown("---")
st.header("Training / Puzzle (unlock by leveling up)")
unlock_level = 2
if st.session_state.state['level'] < unlock_level:
    st.info(f"Reach level {unlock_level} to unlock training games (current level: {st.session_state.state['level']}).")
else:
    st.write("Puzzle Trainer — complete the puzzle to add labeled images and improve the AI.")
    # Choose an image from your uploaded/user dataset or upload one to make a puzzle
    puzzle_choice = st.radio("Use image from", ("Upload new", "From user dataset"))
    puzzle_image = None
    puzzle_label = None
    if puzzle_choice == "Upload new":
        p_upload = st.file_uploader("Upload an image for puzzle (only here)", key="puzzle_upload", type=["png","jpg","jpeg"])
        if p_upload:
            puzzle_image = Image.open(p_upload).convert("RGB")
    else:
        # list images in user_data
        labels = os.listdir(USER_DATA_DIR)
        opts = []
        for lab in labels:
            labdir = os.path.join(USER_DATA_DIR, lab)
            if os.path.isdir(labdir):
                files = os.listdir(labdir)
                for f in files:
                    opts.append(os.path.join(labdir, f))
        selected = st.selectbox("Select user image", options=[""] + opts)
        if selected:
            puzzle_image = Image.open(selected).convert("RGB")
            puzzle_label = os.path.basename(os.path.dirname(selected))

    if puzzle_image is not None:
        grid_size = st.slider("Grid size (NxN)", 2, 4, 3)
        # Initialize puzzle in session_state
        if "puzzle" not in st.session_state:
            st.session_state.puzzle = {}
        # Create puzzle once
        if st.session_state.puzzle.get("src_bytes") != puzzle_image.tobytes() or st.session_state.puzzle.get("N") != grid_size:
            # new puzzle
            w,h = puzzle_image.size
            # Resize square for neat tiles
            side = 300
            puzzle_image = puzzle_image.resize((side, side))
            tiles = []
            N = grid_size
            tile_w = side // N
            for r in range(N):
                for c in range(N):
                    box = (c*tile_w, r*tile_w, (c+1)*tile_w, (r+1)*tile_w)
                    tiles.append(puzzle_image.crop(box))
            order = list(range(len(tiles)))
            np.random.shuffle(order)
            st.session_state.puzzle = {
                "N": N,
                "tiles": tiles,
                "order": order,
                "first_sel": None,
                "src_bytes": puzzle_image.tobytes(),
                "label": puzzle_label or ""
            }

        # Render grid with swap buttons
        N = st.session_state.puzzle["N"]
        order = st.session_state.puzzle["order"]
        tiles = st.session_state.puzzle["tiles"]
        st.write("Click two tiles to swap them.")
        # display grid
        for r in range(N):
            cols = st.columns(N)
            for c in range(N):
                idx = r*N + c
                tile_idx = order[idx]
                with cols[c]:
                    st.image(tiles[tile_idx], use_column_width=True)
                    if st.button(f"Sel {idx}", key=f"sel_{idx}"):
                        if st.session_state.puzzle["first_sel"] is None:
                            st.session_state.puzzle["first_sel"] = idx
                        else:
                            a = st.session_state.puzzle["first_sel"]
                            b = idx
                            # swap
                            order[a], order[b] = order[b], order[a]
                            st.session_state.puzzle["first_sel"] = None
                            st.session_state.puzzle["order"] = order
        # Check solved
        if order == list(range(len(tiles))):
            st.balloons()
            st.success("Puzzle solved! You can save this image to the training set.")
            label_input = st.text_input("Label for this image (will be added to training set):", st.session_state.puzzle.get("label",""))
            if st.button("Save image to dataset and fine-tune model"):
                # reconstruct full image from tiles (original correct order)
                N = st.session_state.puzzle["N"]
                tile_w, tile_h = tiles[0].size
                new_im = Image.new("RGB", (tile_w*N, tile_h*N))
                for r in range(N):
                    for c in range(N):
                        idx = r*N + c
                        new_im.paste(tiles[idx], (c*tile_w, r*tile_h))
                buf = io.BytesIO()
                new_im.save(buf, format="JPEG")
                buf.seek(0)
                save_user_image(buf, label_input, USER_DATA_DIR)
                st.success(f"Saved to dataset as '{label_input}'. Starting fine-tune (short)...")
                # fine tune
                fine_tune_on_user_data(USER_DATA_DIR, MODEL_FILE, CLASSES_FILE, epochs=3)
                # reload predictor
                predictor.reload_if_updated()
        else:
            st.write("Puzzle not yet solved. Keep swapping tiles.")

st.markdown("---")
st.header("Developer / Advanced")
st.write("You can force a short fine-tune on your collected images.")
if st.button("Run short fine-tune now"):
    fine_tune_on_user_data(USER_DATA_DIR, MODEL_FILE, CLASSES_FILE, epochs=3)
    predictor.reload_if_updated()
    st.success("Fine-tune finished.")

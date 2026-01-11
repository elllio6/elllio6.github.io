import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import io
import random

IMG_SIZE = (224,224)

def ensure_user_dirs(base_dir="user_data"):
    os.makedirs(base_dir, exist_ok=True)

def save_user_image(file_like, label, base_dir="user_data"):
    """
    file_like: file-like object (BytesIO) or path
    label: string
    """
    labdir = os.path.join(base_dir, label)
    os.makedirs(labdir, exist_ok=True)
    # generate filename
    idx = len([n for n in os.listdir(labdir) if os.path.isfile(os.path.join(labdir, n))])
    path = os.path.join(labdir, f"{idx+1}.jpg")
    if isinstance(file_like, str):
        # path
        img = Image.open(file_like).convert("RGB")
        img.save(path, format="JPEG")
    else:
        img = Image.open(file_like).convert("RGB")
        img.save(path, format="JPEG")
    return path

class Predictor:
    """
    Predictor supports:
     - a user fine-tuned model (if present)
     - fallback to ImageNet MobileNetV2 decode
    Additionally, to make the AI "bad at the start", the predictor accepts
    `player_level` and will intentionally degrade or randomize guesses when
    the player is low level (encouraging gameplay and training).
    """
    def __init__(self, user_model_path="user_model.h5", classes_path="classes.json"):
        self.user_model_path = user_model_path
        self.classes_path = classes_path
        self.user_model = None
        self.class_names = None
        self._load_user_model()

    def _load_user_model(self):
        if os.path.exists(self.user_model_path) and os.path.exists(self.classes_path):
            try:
                self.user_model = tf.keras.models.load_model(self.user_model_path)
                with open(self.classes_path, "r") as f:
                    self.class_names = json.load(f)  # expected to be list: index -> label
                print("Loaded user model with classes:", self.class_names)
            except Exception as e:
                print("Failed to load user model:", e)
                self.user_model = None
                self.class_names = None
        else:
            self.user_model = None
            self.class_names = None

    def reload_if_updated(self):
        self._load_user_model()

    def _preprocess_pil(self, pil_img):
        img = pil_img.resize(IMG_SIZE)
        arr = np.array(img).astype(np.float32)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def predict_pil(self, pil_img, top=3, player_level=1):
        """
        Returns list of (label, score)

        If a user model exists, returns predictions from it.
        Otherwise falls back to ImageNet MobileNetV2 decoded labels.

        To make the AI 'bad at the start', when player_level is low (e.g. 1),
        this method will intentionally lower confidence and sometimes replace
        the top guess with a random common object.
        """
        # If we have a custom user model, use it (it's trained on the player's dataset)
        if self.user_model is not None and self.class_names:
            x = self._preprocess_pil(pil_img)
            preds = self.user_model.predict(x)[0]
            idxs = preds.argsort()[::-1][:top]
            return [(self.class_names[i], float(preds[i])) for i in idxs]

        # Fallback to ImageNet MobileNetV2
        x = self._preprocess_pil(pil_img)
        base = MobileNetV2(weights="imagenet")
        preds = base.predict(x)
        decoded = decode_predictions(preds, top=top)[0]
        labels = [(d[1].replace("_"," "), float(d[2])) for d in decoded]

        # Intentional poor performance when player level is low
        # The behavior: at level 1 most top guesses are noisy/low-confidence.
        # At higher levels, return the decoded labels unchanged.
        if player_level <= 1:
            commons = ["banana", "chair", "rock", "shoe", "cat", "dog", "tree", "car", "clock", "bottle", "box", "cup"]
            # Replace top label with a random common noun 70% of the time
            if random.random() < 0.7:
                fake = random.choice(commons)
                labels[0] = (fake, 0.12 + random.random() * 0.08)  # low confidence
                # slightly lower the confidence of other guesses to make it sound uncertain
                for i in range(1, len(labels)):
                    labels[i] = (labels[i][0], max(0.01, labels[i][1]*0.5))
            else:
                # allow real top label but lower confidence to make it "hesitant"
                labels = [(lbl, max(0.05, score * 0.3)) for lbl,score in labels]
            # Shuffle second and third occasionally
            if random.random() < 0.4 and len(labels) > 2:
                labels[1], labels[2] = labels[2], labels[1]

        return labels

def fine_tune_on_user_data(user_data_dir, save_model_path="user_model.h5", classes_path="classes.json", epochs=5):
    """
    Build a classifier from user-labeled images in user_data_dir.
    Directory structure: user_data_dir/<label>/*.jpg
    Saves:
      - model to save_model_path
      - classes (list index->label) to classes_path
    """
    labels = [d for d in os.listdir(user_data_dir) if os.path.isdir(os.path.join(user_data_dir,d))]
    if not labels:
        print("No user data found for fine-tuning.")
        return
    labels.sort()
    num_classes = len(labels)
    # ImageDataGenerator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2,
                                 rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_gen = datagen.flow_from_directory(user_data_dir, target_size=IMG_SIZE, batch_size=8, subset="training")
    val_gen = datagen.flow_from_directory(user_data_dir, target_size=IMG_SIZE, batch_size=8, subset="validation")
    # build model
    base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))
    base.trainable = False
    x = base.output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    # train
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)
    # save
    model.save(save_model_path)
    # Convert class_indices (dict) to list index->label for predictable loading
    class_indices = train_gen.class_indices  # e.g. {'cat': 0, 'dog': 1}
    idx_to_label = {v:k for k,v in class_indices.items()}
    ordered = [idx_to_label[i] for i in range(len(idx_to_label))]
    with open(classes_path, "w") as f:
        json.dump(ordered, f)
    print("Saved user model and classes.")

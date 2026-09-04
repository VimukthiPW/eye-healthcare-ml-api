from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


# =================================================
# Configuration
# =================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "eye_validation_split"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 15

MODEL_OUTPUT = BASE_DIR / "Eye_NonEye_Validator.keras"


# =================================================
# Check TensorFlow
# =================================================

print("TensorFlow version:", tf.__version__)

print("\nLoading datasets...")


# =================================================
# Load Training Dataset
# =================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# =================================================
# Load Validation Dataset
# =================================================

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =================================================
# Load Test Dataset
# =================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =================================================
# Show Class Names
# =================================================

print("\nClass names:")
print(train_dataset.class_names)

# Expected:
# ['eye', 'non_eye']


# =================================================
# Performance Optimization
# =================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# =================================================
# Data Augmentation
# =================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)


# =================================================
# Load Pretrained MobileNetV3Small
# =================================================

print("\nLoading MobileNetV3Small...")

base_model = MobileNetV3Small(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Freeze pretrained layers initially

base_model.trainable = False


# =================================================
# Build Model
# =================================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# =================================================
# Compile Model
# =================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# =================================================
# Model Summary
# =================================================

print("\nModel Summary:")
model.summary()


# =================================================
# Callbacks
# =================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_OUTPUT),
        monitor="val_accuracy",
        save_best_only=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
]


# =================================================
# Train Model
# =================================================

print("\n========================================")
print("Starting Eye / Non-Eye Training")
print("========================================\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =================================================
# Evaluate Test Dataset
# =================================================

print("\n========================================")
print("Evaluating Test Dataset")
print("========================================\n")

test_results = model.evaluate(
    test_dataset,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    test_results
):
    print(
        f"{name}: {value:.4f}"
    )


# =================================================
# Save Final Model
# =================================================

model.save(
    str(MODEL_OUTPUT)
)

print("\n========================================")
print("Training Completed Successfully!")
print("========================================")

print(
    f"\nModel saved to:\n{MODEL_OUTPUT}"
)
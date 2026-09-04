import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


app = Flask(__name__)


# =================================================
# Model Configuration
# =================================================

BASE_DIR = Path(__file__).resolve().parent

# Disease classification model
DISEASE_MODEL_PATH = BASE_DIR / "Trained_Eye_disease_model.keras"

# Eye / Non-Eye validation model
VALIDATOR_MODEL_PATH = BASE_DIR / "Eye_NonEye_Validator.keras"


# Disease model classes
CLASS_NAMES = [
    "Cataract",
    "Conjunctivitis",
    "Healthy"
]

IMAGE_SIZE = (224, 224)


# =================================================
# Load Models
# =================================================

print("Loading Eye / Non-Eye validation model...")

validator_model = tf.keras.models.load_model(
    str(VALIDATOR_MODEL_PATH),
    compile=False
)

print("Eye / Non-Eye validation model loaded successfully!")


print("Loading Eye Disease model...")

disease_model = tf.keras.models.load_model(
    str(DISEASE_MODEL_PATH),
    compile=False
)

print("Eye Disease model loaded successfully!")


# =================================================
# Prepare Image
# =================================================

def prepare_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =================================================
# Eye / Non-Eye Validation
# =================================================

def validate_eye_image(image):

    image_array = prepare_image(image)

    # MobileNetV3 preprocessing
    image_array = preprocess_input(
        image_array
    )

    # Validator prediction
    prediction = validator_model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Dataset class order:
    # 0 = eye
    # 1 = non_eye
    #
    # Sigmoid output:
    # close to 0 = eye
    # close to 1 = non_eye

    non_eye_probability = float(
        prediction
    )

    eye_probability = 1.0 - non_eye_probability

    # Decide whether image is an eye image
    is_eye = eye_probability >= non_eye_probability

    return {
        "is_eye": is_eye,
        "eye_confidence": round(
            eye_probability * 100,
            2
        ),
        "non_eye_confidence": round(
            non_eye_probability * 100,
            2
        )
    }


# =================================================
# Eye Disease Prediction
# =================================================

def predict_eye_disease(image):

    image_array = prepare_image(image)

    # MobileNetV3 preprocessing
    image_array = preprocess_input(
        image_array
    )

    # Disease model prediction
    predictions = disease_model.predict(
        image_array,
        verbose=0
    )[0]

    # Convert output to probabilities
    if (
        np.all(predictions >= 0)
        and np.all(predictions <= 1)
        and np.isclose(
            np.sum(predictions),
            1.0,
            atol=1e-3
        )
    ):

        probabilities = predictions

    else:

        probabilities = tf.nn.softmax(
            predictions
        ).numpy()

    # Find highest probability class
    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    # All probabilities
    probability_details = {
        CLASS_NAMES[i]: round(
            float(probabilities[i]) * 100,
            2
        )
        for i in range(len(CLASS_NAMES))
    }

    return {
        "condition": predicted_class,
        "confidence": round(
            confidence * 100,
            2
        ),
        "message": (
            "Prediction completed successfully."
        ),
        "probabilities": probability_details
    }


# =================================================
# Health Check
# =================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "message": "ML API is running"
    })


# =================================================
# Prediction API
# =================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "success": False,
                "message": "No image uploaded"
            }), 400

        image_file = request.files["image"]

        # Check filename
        if image_file.filename == "":

            return jsonify({
                "success": False,
                "message": "No image selected"
            }), 400

        # Open image
        image = Image.open(
            image_file.stream
        )

        # =================================================
        # STEP 1: Eye / Non-Eye Validation
        # =================================================

        validation = validate_eye_image(
            image
        )

        print(
            "Eye validation:",
            validation
        )

        # =================================================
        # STEP 2: Reject Non-Eye Image
        # =================================================

        if not validation["is_eye"]:

            return jsonify({
                "success": True,
                "result": {
                    "condition": "Invalid Image",
                    "confidence": validation[
                        "non_eye_confidence"
                    ],
                    "message": (
                        "Please upload a clear eye image."
                    ),
                    "validation": {
                        "eye_confidence": validation[
                            "eye_confidence"
                        ],
                        "non_eye_confidence": validation[
                            "non_eye_confidence"
                        ]
                    }
                }
            })

        # =================================================
        # STEP 3: Run Disease Model
        # =================================================

        result = predict_eye_disease(
            image
        )

        # Add validation information
        result["validation"] = {
            "eye_confidence": validation[
                "eye_confidence"
            ],
            "non_eye_confidence": validation[
                "non_eye_confidence"
            ]
        }

        # =================================================
        # STEP 4: Return Final Result
        # =================================================

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# =================================================
# Start Flask Server
# =================================================

if __name__ == "__main__":

    # Railway provides the PORT environment variable.
    # Use 8000 when running locally.
    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    print(
        f"Starting ML API on port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
"""
Test / Inference script for Medicinal Plant Recognition Model
Upload an image of a flower or leaf → get the plant name + medicinal properties
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# ─── Configuration ─────────────────────────────────────────────────
MODEL_PATH = "medicinal_model.keras"
CLASS_INDICES_PATH = "medicinal_class_indices.json"
PROPERTIES_PATH = "medicinal_properties.json"
IMG_SIZE = (224, 224)
# ───────────────────────────────────────────────────────────────────


def load_resources():
    """Load the trained model, class indices, and medicinal properties."""
    # Check files exist
    for fpath, label in [(MODEL_PATH, "Model"), (CLASS_INDICES_PATH, "Class indices"), (PROPERTIES_PATH, "Medicinal properties")]:
        if not os.path.exists(fpath):
            print(f"ERROR: {label} file not found: {fpath}")
            print("Please run 'python train_medicinal.py' first to train the model.")
            exit(1)

    print("Loading model...")
    model = load_model(MODEL_PATH)

    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Invert: index -> class name
    labels = {v: k for k, v in class_indices.items()}

    with open(PROPERTIES_PATH, 'r') as f:
        properties = json.load(f)

    print("Model loaded successfully!\n")
    return model, labels, properties


def predict(image_path, model, labels, properties, top_k=3):
    """
    Predict the plant from an image and return medicinal properties.

    Args:
        image_path: Path to the image file
        model: Loaded Keras model
        labels: Dict mapping index -> class name
        properties: Dict mapping class name -> medicinal properties
        top_k: Number of top predictions to show
    """
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at '{image_path}'")
        return None

    # Preprocess image
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    top_indices = np.argsort(predictions[0])[::-1][:top_k]

    # Top prediction
    predicted_index = top_indices[0]
    confidence = predictions[0][predicted_index]
    predicted_name = labels.get(predicted_index, labels.get(str(predicted_index), "Unknown"))

    # ─── Display Results ──────────────────────────────────────────
    print("=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"\n  Identified Plant: {predicted_name}")
    print(f"  Confidence:       {confidence:.2%}")

    # Top-K predictions
    print(f"\n  Top-{top_k} Predictions:")
    for rank, idx in enumerate(top_indices, 1):
        name = labels.get(idx, labels.get(str(idx), "Unknown"))
        conf = predictions[0][idx]
        bar = "█" * int(conf * 20)
        print(f"    {rank}. {name:30s} {conf:6.2%}  {bar}")

    # Medicinal properties
    plant_info = properties.get(predicted_name)
    if plant_info:
        print("\n" + "=" * 60)
        print("  MEDICINAL PROPERTIES")
        print("=" * 60)
        print(f"\n  Scientific Name : {plant_info['scientific_name']}")
        print(f"  Family          : {plant_info['family']}")
        print(f"  Common Names    : {', '.join(plant_info['common_names'])}")
        print(f"  Parts Used      : {', '.join(plant_info['parts_used'])}")

        print("\n  Medicinal Uses:")
        for use in plant_info['medicinal_uses']:
            print(f"    • {use}")

        print(f"\n  Key Compounds   : {', '.join(plant_info['key_compounds'])}")
        print(f"  Traditional Systems: {', '.join(plant_info['traditional_systems'])}")

        print(f"\n  ⚠ Precautions:")
        print(f"    {plant_info['precautions']}")
    else:
        print(f"\n  (No medicinal property data found for '{predicted_name}')")

    print("\n" + "=" * 60)
    print("  DISCLAIMER: This information is for educational purposes")
    print("  only and should NOT be considered medical advice.")
    print("=" * 60)

    return {
        "name": predicted_name,
        "confidence": float(confidence),
        "properties": plant_info
    }


# ─── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, labels, properties = load_resources()

    print("=" * 60)
    print("  MEDICINAL PLANT RECOGNITION SYSTEM")
    print("  Enter an image path to identify a plant and its")
    print("  medicinal properties. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        print()
        img_path = input("Enter image path (or 'quit'): ").strip().strip("\"'")

        if img_path.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        if not img_path:
            print("Please enter a valid path.")
            continue

        predict(img_path, model, labels, properties)

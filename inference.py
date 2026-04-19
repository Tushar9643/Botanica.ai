import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_flower(image_path, model_path='efficientnetb3_flower_model.keras', class_indices_path='class_indices.json'):
    """
    Predicts the flower class for a given image path.
    """
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}. Please train the model in the notebook first.")
            return
            
        if not os.path.exists(class_indices_path):
            print(f"Class indices file not found: {class_indices_path}. Please generate it in the notebook.")
            return

        print("Loading model...")
        model = load_model(model_path)
        
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        # JSON keys are strings, values are ints. Invert: val (int) -> key (str)
        labels = {v: k for k, v in class_indices.items()}

        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return

        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        predicted_label = labels.get(predicted_index) or labels.get(str(predicted_index))

        print(f"Prediction: {predicted_label} ({confidence:.2%})")
        
        plt.imshow(img)
        plt.title(f"{predicted_label} ({confidence:.2%})")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    img_path = input("Enter the path to the flower image: ")
    # Remove quotes if user added them
    img_path = img_path.strip('"\'')
    predict_flower(img_path)

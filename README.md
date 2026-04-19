# Botanica.ai 🌿

**An AI-Powered Indian Medicinal Plant Recognition System**

Botanica.ai is a deep learning computer vision project designed to identify medicinal plants native to India from images and provide their detailed medicinal properties, scientific names, and traditional uses.

## 🚀 Features
- **Accurate Identification**: Powered by **EfficientNetB3** tailored for fine-grained image classification.
- **Medicinal Knowledge Base**: Instantly retrieves and displays the plant's scientific name, family, common names, parts used, key compounds, and medicinal uses upon recognition.
- **Safety Warnings**: Provides integrated precaution information for toxic or potent plants (like Datura).
- **Colab-Ready Training**: Includes an optimized, GPU-ready Google Colab notebook for fast (~35min) model training.

## 🛠️ Tech Stack
- **Framework**: TensorFlow / Keras
- **Model Architecture**: EfficientNetB3 (Transfer Learning, trained on ImageNet)
- **Data Augmentation**: `ImageDataGenerator`
- **Languages/Tools**: Python, Pandas, NumPy, Matplotlib

## 📦 Dataset
This project uses the **IMFI (Indian Medicinal Flower Image) Dataset**, which contains images of 28 specific Indian medicinal flowers.
- *Note: Due to GitHub's file size limits, the dataset and pre-trained `.keras` models are not included in this repository.*

### Supported Classes (28 Plants)
Adathoda, Banana, Bush Clock Vine, Champaka, Chitrak, Common Lanthana, Crown Flower, Datura, Four O' Clock Flower, Hibiscus, Honey Suckle, Indian Mallow, Jatropha, Malabar Melastome, Marigold, Nagapoovu, Nityakalyani, Pinwheel, Rose, Shankupushpam, Spider Lily, Sunflower, Thechi, Thumba, Touch Me Not, Tridax procumbens, Wild potato vine, Yellow Daisy.

## 💻 Getting Started

### 1. Training the Model (Recommended via Colab)
If you just cloned the repo and don't have the `medicinal_model.keras` file, you need to train the model first.
1. Download the IMFI dataset as a ZIP file.
2. Open `colab_train.ipynb` in Google Colab.
3. Upload the dataset to your Google Drive.
4. Set hardware accelerator to **T4 GPU** in Colab.
5. Run all cells. The trained model (`medicinal_model.keras`) and indices (`medicinal_class_indices.json`) will be saved directly to your Google Drive.

### 2. Local Inference
Once you have the trained `medicinal_model.keras` and `medicinal_class_indices.json` in your project folder, you can test the model locally.

```bash
# Install dependencies
pip install tensorflow numpy pandas matplotlib scikit-learn

# Run the inference script
python test_model.py
```
*The script will prompt you to enter the path to an image file. It will then display the image, the top 3 prediction confidences, and the plant's medicinal properties.*

## ⚠️ Disclaimer
The medicinal information provided by Botanica.ai is for **educational purposes only** and should **NOT** be considered professional medical advice. Always consult a qualified healthcare provider or Ayurvedic practitioner before using any plant for medicinal purposes. Some plants included in the dataset (e.g., Datura, Crown Flower) are highly toxic.

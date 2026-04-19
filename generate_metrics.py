import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

def generate_evaluation_metrics(model_path, data_dir, class_indices_path, output_dir="metrics_report"):
    """
    Creates comprehensive evaluation metrics (Confusion Matrix, F1/Precision/Recall bars, 
    and a text report) and saves them as PNG files.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Loading class indices from {class_indices_path}...")
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
        
    labels_map = {v: k for k, v in class_indices.items()}
    class_names = [labels_map[i] for i in range(len(labels_map))]
    
    # Load Test Data
    print(f"Loading test data from {data_dir}...")
    # NOTE: In a real scenario, you need your test_df here. 
    # Since we are running this post-training or independent, we will simulate loading all images as test
    # for metric generation. Ideally, you run this on your holdout test set.
    
    filepaths = []
    labels = []
    for cls_name in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir): continue
        for img in os.listdir(cls_dir):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                filepaths.append(os.path.join(cls_dir, img))
                labels.append(cls_name)
                
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    
    # In a real script, filter this df to only be the 10% test set. 
    # For now, we take 10% randomly just to generate reports.
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['labels'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    ts_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = ts_gen.flow_from_dataframe(
        test_df, x_col='filepaths', y_col='labels',
        target_size=(224, 224), class_mode='categorical',
        color_mode='rgb', shuffle=False, batch_size=32
    )
    
    print("Generating predictions...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # ============================================================
    # 1. Classification Report & F1-Score Chart
    # ============================================================
    print("Saving Classification Report...")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract metrics for plotting
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for cls in class_names:
        if cls in report_dict:
            classes.append(cls)
            precision.append(report_dict[cls]['precision'])
            recall.append(report_dict[cls]['recall'])
            f1.append(report_dict[cls]['f1-score'])
            
    # Plot Metrics Bar Chart
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#4C72B0')
    rects2 = ax.bar(x, recall, width, label='Recall', color='#55A868')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#C44E52')
    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Precision, Recall, and F1-Score by Class', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'class_metrics_bar_chart.png')
    plt.savefig(metrics_path, dpi=200)
    plt.close()
    
    # ============================================================
    # 2. Confusion Matrix
    # ============================================================
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8})
    plt.title('Confusion Matrix', fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200)
    plt.close()
    
    # ============================================================
    # 3. Overall Report File
    # ============================================================
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(output_dir, 'full_classification_report.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION METRICS REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report_text)
        f.write("\n\n")
        f.write(f"Accuracy: {report_dict['accuracy']:.4f}\n")
        f.write(f"Macro Avg F1: {report_dict['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted Avg F1: {report_dict['weighted avg']['f1-score']:.4f}\n")
        
    print(f"\n[OK] All metric reports saved in the '{output_dir}' directory!")
    print(f"   - {metrics_path}")
    print(f"   - {cm_path}")
    print(f"   - {os.path.join(output_dir, 'full_classification_report.txt')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate validation metrics")
    parser.add_argument("--model", type=str, default="medicinal_model.keras", help="Path to keras model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--classes", type=str, default="medicinal_class_indices.json", help="Path to class indices JSON")
    args = parser.parse_args()
    
    if os.path.exists(args.model) and os.path.exists(args.data):
        generate_evaluation_metrics(args.model, args.data, args.classes)
    else:
        print("Model or Data path not found!")

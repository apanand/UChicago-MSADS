import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load the best ensemble model
best_model = load_model('best_ensemble_model.h5')

classes = ['hatchback', 'mini_van', 'minibus', 'pickup', 'sedan', 'sports_convertible', 'station_wagon', 'suv_crossover']

def process_image(image_path, true_box):
    print(f"Processing image: {image_path}")
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to match the input size of your model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Perform classification
    predictions = best_model.predict(img_array)
    class_index = np.argmax(predictions[0])
    class_name = classes[class_index]
    confidence = float(predictions[0][class_index])
    
    result = {
        'box': true_box['box'],
        'class': class_name,
        'confidence': confidence
    }
    
    return [result], [true_box]

def display_image_with_boxes(image_path, results, true_boxes):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw predicted boxes
    for result in results:
        box = result['box']
        label = f"Pred: {result['class']} {result['confidence']:.2f}"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]-20), label, fill="red", font=font)

    # Draw true boxes
    for true_box in true_boxes:
        box = true_box['box']
        label = f"True: {true_box['class']}"
        draw.rectangle(box, outline="green", width=3)
        draw.text((box[0], box[1]-40), label, fill="green", font=font)

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def process_folder(folder_path, annotations_file):
    df = pd.read_csv(annotations_file)
    all_results = []
    true_labels = []
    predicted_labels = []
    unique_classes = set()

    for _, row in df.iterrows():
        filename = row['filename']
        image_path = os.path.join(folder_path, filename)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        true_class = row['class']
        unique_classes.add(true_class)
        true_box = {
            'box': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
            'class': true_class
        }
        true_labels.append(true_class)

        results, _ = process_image(image_path, true_box)
        
        if results:
            predicted_class = results[0]['class']
            unique_classes.add(predicted_class)
            predicted_labels.append(predicted_class)
            all_results.append((image_path, results, results[0]['confidence'], [true_box]))
            print(f"Processed {filename}: Predicted {predicted_class}, True {true_class}")
        else:
            print(f"No prediction for {filename}")

    return all_results, true_labels, predicted_labels, list(unique_classes)

# Process the 'valid' folder
folder_path = 'valid'
annotations_file = os.path.join(folder_path, '_annotations.csv')
all_results, true_labels, predicted_labels, unique_classes = process_folder(folder_path, annotations_file)

# Display classification report
print("\nClassification Report:")
report = classification_report(true_labels, predicted_labels, target_names=unique_classes, output_dict=True)
print(classification_report(true_labels, predicted_labels, target_names=unique_classes))

# Extract F1 scores and recall values for each class
f1_scores = {class_name: report[class_name]['f1-score'] for class_name in unique_classes if class_name in report}
recall_scores = {class_name: report[class_name]['recall'] for class_name in unique_classes if class_name in report}

# Sort classes by F1 score in descending order
sorted_classes = sorted(f1_scores.keys(), key=lambda x: f1_scores[x], reverse=True)
sorted_f1_scores = [f1_scores[class_name] for class_name in sorted_classes]
sorted_recall_scores = [recall_scores[class_name] for class_name in sorted_classes]

# Create a bar plot for F1 scores
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(sorted_classes)), sorted_f1_scores)
plt.title('F1 Scores by Car Class')
plt.xlabel('Car Class')
plt.ylabel('F1 Score')
plt.xticks(range(len(sorted_classes)), sorted_classes, rotation=45, ha='right')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Create a bar plot for recall scores
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(sorted_classes)), sorted_recall_scores)
plt.title('Recall Scores by Car Class')
plt.xlabel('Car Class')
plt.ylabel('Recall Score')
plt.xticks(range(len(sorted_classes)), sorted_classes, rotation=45, ha='right')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print the F1 scores and recall values
print("\nF1 Scores and Recall by Class:")
for class_name in sorted_classes:
    print(f"{class_name}: F1 = {f1_scores[class_name]:.4f}, Recall = {recall_scores[class_name]:.4f}")


# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(unique_classes))
plt.xticks(tick_marks, unique_classes, rotation=45, ha='right')
plt.yticks(tick_marks, unique_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Display top 5 images with highest confidence
top_5 = sorted(all_results, key=lambda x: x[2], reverse=True)[:5]
for image_path, results, confidence, true_boxes in top_5:
    print(f"\nImage: {image_path}")
    print(f"Max Confidence: {confidence}")
    display_image_with_boxes(image_path, results, true_boxes)

# Calculate and display statistics
class_counts = {class_name: sum(1 for _, results, _, _ in all_results if results[0]['class'] == class_name) for class_name in unique_classes}
total_detections = sum(class_counts.values())

print("\nDetection Statistics:")
for class_name, count in class_counts.items():
    percentage = (count / total_detections) * 100 if total_detections > 0 else 0
    print(f"{class_name}: {count} ({percentage:.2f}%)")

plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title('Car Class Distribution in Predicted Images')
plt.xlabel('Car Class')
plt.ylabel('Number of Predictions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

base_dir = 'sample_balanced'
classes = ['hatchback', 'mini_van', 'minibus', 'pickup', 'sedan', 'sports_convertible','station_wagon','suv_crossover'] 

# Verify the directory structure
for class_name in classes:
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_dir):
        print(f"Warning: Directory not found for class {class_name}")

# Custom learning rate schedule with warmup
def lr_schedule(epoch):
    initial_lr = 1e-3
    if epoch < 10:
        return initial_lr
    else:
        return initial_lr * tf.math.exp(0.1 * (10 - epoch))

# Mixup data augmentation
def mixup(x, y, alpha=0.2):
    if alpha > 0:
        lam = tf.random.uniform([], 0, alpha)
    else:
        lam = 1

    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))

    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)

    return mixed_x, mixed_y

# Set up data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Load all data
all_data = []
all_labels = []
for i, class_name in enumerate(classes):
    class_dir = os.path.join(base_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        all_data.append(img_array)
        all_labels.append(i)

all_data = np.array(all_data)
all_labels = np.array(all_labels)

# Implement k-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create lists to store metrics for all folds
all_train_accuracies = []
all_val_accuracies = []
models = []

for fold, (train_index, val_index) in enumerate(kf.split(all_data)):
    print(f"Training on fold {fold+1}/{n_splits}...")

    # Split data
    x_train, x_val = all_data[train_index], all_data[val_index]
    y_train, y_val = all_labels[train_index], all_labels[val_index]

    # Convert to TensorFlow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, tf.one_hot(y_train, depth=len(classes))))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32)
    train_dataset = train_dataset.map(mixup)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, tf.one_hot(y_val, depth=len(classes)))).batch(32)

    # Load the pre-trained EfficientNetV2L model
    base_model = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze all layers
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Unfreeze the last 100 layers
    for layer in model.layers[-100:]:
        layer.trainable = True

    # Compile the model
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        callbacks=[early_stopping, reduce_lr, lr_scheduler],
        verbose=1
    )

    # Print final accuracies for this fold
    print(f"Fold {fold+1} - Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Fold {fold+1} - Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Store accuracies for later plotting
    all_train_accuracies.append(history.history['accuracy'])
    all_val_accuracies.append(history.history['val_accuracy'])
    
    # Plot accuracies for this fold
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - Fold {fold+1}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Evaluate the model
    evaluation = model.evaluate(val_dataset)
    print(f"Fold {fold+1} - Validation Loss: {evaluation[0]:.4f}")
    print(f"Fold {fold+1} - Validation Accuracy: {evaluation[1]:.4f}")

    # Save the model for this fold
    model.save(f'car_classification_model_fold_{fold+1}.h5')
    print(f"Model for fold {fold+1} saved successfully.")
    
    # Add model to the list
    models.append(model)

# Plot average accuracies across all folds
plt.figure(figsize=(10, 6))
for i in range(n_splits):
    plt.plot(all_train_accuracies[i], label=f'Fold {i+1} Training', alpha=0.3)
    plt.plot(all_val_accuracies[i], label=f'Fold {i+1} Validation', alpha=0.3)

plt.plot(np.mean(all_train_accuracies, axis=0), label='Average Training', linewidth=2)
plt.plot(np.mean(all_val_accuracies, axis=0), label='Average Validation', linewidth=2)
plt.title('Model Accuracy Across All Folds')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Prepare a test set (using the validation set from the last fold for this example)
test_dataset = val_dataset

# Create a new model that represents the ensemble
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
ensemble_outputs = []

for model in models:
    ensemble_outputs.append(model(input_layer))

average_output = tf.keras.layers.Average()(ensemble_outputs)
ensemble_model = tf.keras.Model(inputs=input_layer, outputs=average_output)

# Save the ensemble model
ensemble_model.save('car_classification_ensemble_model.h5')
print("Ensemble model saved successfully.")

# Test the saved ensemble model
loaded_ensemble_model = tf.keras.models.load_model('car_classification_ensemble_model.h5')

# Prepare the test dataset
test_images = np.concatenate([x for x, _ in test_dataset], axis=0)

# Get predictions from the loaded ensemble model
loaded_ensemble_preds = loaded_ensemble_model.predict(test_images)
loaded_ensemble_class_preds = np.argmax(loaded_ensemble_preds, axis=1)

# Get true labels
y_test = np.concatenate([y for x, y in test_dataset], axis=0)
y_test = np.argmax(y_test, axis=1)  # Convert one-hot back to integer labels

# Evaluate loaded ensemble predictions
loaded_ensemble_accuracy = accuracy_score(y_test, loaded_ensemble_class_preds)
print(f"Loaded Ensemble Model Accuracy: {loaded_ensemble_accuracy:.4f}")

# Print detailed classification report for loaded ensemble
print("\nClassification Report (Loaded Ensemble):")
print(classification_report(y_test, loaded_ensemble_class_preds, target_names=classes))

# Print confusion matrix for loaded ensemble
print("\nConfusion Matrix (Loaded Ensemble):")
print(confusion_matrix(y_test, loaded_ensemble_class_preds))

# Visualize the confusion matrix for loaded ensemble
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix(y_test, loaded_ensemble_class_preds), interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Loaded Ensemble)')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Analyze misclassifications for loaded ensemble
misclassified = y_test != loaded_ensemble_class_preds
misclassified_indices = np.where(misclassified)[0]

print("\nMisclassified Samples (Loaded Ensemble):")
for idx in misclassified_indices[:10]:  # Print first 10 misclassifications
    true_label = classes[y_test[idx]]
    pred_label = classes[loaded_ensemble_class_preds[idx]]
    print(f"Sample {idx}: True: {true_label}, Predicted: {pred_label}")
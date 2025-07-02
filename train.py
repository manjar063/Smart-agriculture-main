import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from utils.preprocessor import create_data_generator
import json

def build_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(160, 160, 3)
    )
    base_model.trainable = True

    # Fine-tune the last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    # Paths
    data_dir = Path('data')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'

    # Parameters
    batch_size = 16
    image_size = (160, 160)
    epochs = 10

    # Data generators
    datagen = create_data_generator()

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Build model
    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
    ]

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save class indices
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    return history

def evaluate_model(model, test_generator):
    # Get predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Ensure correct label mapping
    class_indices = test_generator.class_indices
    labels = list(class_indices.values())
    target_names = list(class_indices.keys())

    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes, labels=labels)
    print("Confusion Matrix:")
    print(cm)

    # Generate classification report
    report = classification_report(true_classes, predicted_classes, labels=labels, target_names=target_names)
    print("Classification Report:")
    print(report)

    # Save report to file
    with open("models/evaluation_report.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

if __name__ == '__main__':
    history = train_model()
    if history is not None:
        model = tf.keras.models.load_model('models/best_model.h5')
        data_dir = Path('data')
        test_dir = data_dir / 'test'
        datagen = create_data_generator()
        validation_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(160, 160),
            batch_size=16,
            class_mode='categorical'
        )
        evaluate_model(model, validation_generator)

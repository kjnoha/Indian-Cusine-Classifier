"""
TEST MODEL ACCURACY
Check performance on unseen test data
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

def test_model():
    # Load the trained model
    model = keras.models.load_model('models/mobilenet_model.h5')
    
    # Create test data generator
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        'filtered_dataset/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['north_indian', 'south_indian'],
        shuffle=False  # Important for accurate evaluation
    )
    
    # Evaluate on test data
    print("🧪 Evaluating on TEST data...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"📊 Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"📉 Test Loss: {test_loss:.4f}")
    
    # Compare with validation accuracy
    print(f"\n📈 COMPARISON:")
    print(f"✅ Validation Accuracy: 93.30%")
    print(f"🧪 Test Accuracy: {test_accuracy * 100:.2f}%")
    
    if test_accuracy * 100 >= 90:
        print("🏆 EXCELLENT! Model generalizes well to unseen data!")
    elif test_accuracy * 100 >= 85:
        print("✅ GOOD! Model performs well on test data!")
    else:
        print("⚠️ Model might be overfitting to validation data")

if __name__ == "__main__":
    test_model()
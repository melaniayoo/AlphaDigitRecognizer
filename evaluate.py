from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from main import load_data
import numpy as np
import matplotlib.pyplot as plt

# Load data
(training_data, testing_data, mapping, num_classes) = load_data(
    r"C:\Users\mehak\HandwritingRecognition\Database\train\emnist-letters.mat", 28, 28, None, True
)

# Extract testing images and labels
testing_images, testing_labels = testing_data

# Load the trained model
model = load_model("cnn_emnist_model.h5")

# Evaluate on test data
y_test = to_categorical(testing_labels - 1, num_classes)
test_loss, test_accuracy = model.evaluate(testing_images, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict on a single sample
index = np.random.randint(len(testing_images))
image = testing_images[index]
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_label = int(np.argmax(prediction))  # Ensure integer

# Adjust actual label and ensure it's an integer
actual_label = int(testing_labels[index]) - 1

# Visualize the prediction
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted: {chr(predicted_label + ord('A'))}, Actual: {chr(actual_label + ord('A'))}")
plt.axis('off')
plt.show()

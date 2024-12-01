from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from main import load_data
import numpy as np
import matplotlib.pyplot as plt

# Load data for letters
(_, (testing_images_letters, testing_labels_letters), _, _) = load_data(
    r"C:\Users\mehak\HandwritingRecognition\Database\train\emnist-letters.mat", 28, 28, None, True
)

# Load data for digits
(_, (testing_images_digits, testing_labels_digits), _, _) = load_data(
    r"C:\Users\mehak\HandwritingRecognition\Database\train\emnist-digits.mat", 28, 28, None, True
)

# Adjust labels to ensure no overlap
testing_labels_letters = np.array(testing_labels_letters) + 9  # Offset letters to start from 10
testing_labels_digits = np.array(testing_labels_digits) - 1  # Ensure digits are 0-9

# Combine testing data
testing_images = np.concatenate((testing_images_letters, testing_images_digits), axis=0)
testing_labels = np.concatenate((testing_labels_letters, testing_labels_digits), axis=0)

# Flatten labels to ensure they are 1D arrays
testing_labels = testing_labels.flatten()

# Filter testing data
valid_test_indices = (testing_labels <= 35)
testing_images = testing_images[valid_test_indices]
testing_labels = testing_labels[valid_test_indices]

# Verify unique labels
print("Filtered unique testing labels:", np.unique(testing_labels))

# One-hot encode labels
num_classes = 36  # 10 digits + 26 letters
y_test = to_categorical(testing_labels, num_classes)

# Load the trained model
model = load_model("cnn_emnist_alphanumeric_model.keras")

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(testing_images, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict on a single sample
index = np.random.randint(len(testing_images))
image = testing_images[index]
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(prediction)
actual_label = testing_labels[index]

# Map numeric label to character
if predicted_label < 10:  # Digits 0-9
    predicted_char = str(predicted_label)
else:  # Letters A-Z
    predicted_char = chr(predicted_label - 10 + ord('A'))

if actual_label < 10:  # Digits 0-9
    actual_char = str(actual_label)
else:  # Letters A-Z
    actual_char = chr(actual_label - 10 + ord('A'))

# Visualize the prediction
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_char}, Actual: {actual_char}")
plt.axis('off')
plt.show()

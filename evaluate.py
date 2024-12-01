# evaluate.py
from tensorflow.keras.models import load_model
from data_loader import load_data
import numpy as np
import matplotlib.pyplot as plt

# Load data
(_, _), (testing_images, testing_labels), num_classes = load_data(
    "C:/path/to/emnist-letters.mat", 28, 28
)

# Load the trained model
model = load_model("cnn_emnist_model.h5")

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(testing_images, to_categorical(testing_labels, num_classes))
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict on a single sample
index = np.random.randint(len(testing_images))
image = testing_images[index]
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(prediction)

# Visualize the prediction
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_label}, Actual: {testing_labels[index]}")
plt.axis('off')
plt.show()

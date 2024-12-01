from main import load_data  # Or your data loader module
from model import build_cnn_model  # Import the model architecture
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load data
(training_images, training_labels), (testing_images, testing_labels),num_classes = load_data(
    "/Users/OWNER/SideProjects/HandwritingRecognition/Database/train/emnist-letters.mat", 28, 28
)

training_images, training_labels = training_data
testing_images, testing_labels = testing_data

# Ensure labels are in the correct range (convert 1-indexed to 0-indexed if necessary)
training_labels = np.array(training_labels) - 1
testing_labels = np.array(testing_labels) - 1

# Verify unique labels
print("Unique labels in training set:", np.unique(training_labels))
print("Unique labels in testing set:", np.unique(testing_labels))

# Convert labels to one-hot encoding
y_train = to_categorical(training_labels, num_classes)
y_test = to_categorical(testing_labels, num_classes)

# Build the model
input_shape = (28, 28, 1)
model = build_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, y_train, batch_size=128, epochs=10, validation_data=(testing_images, y_test))

# Evaluate and print results
test_loss, test_accuracy = model.evaluate(testing_images, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
model.save("cnn_emnist_model.h5")

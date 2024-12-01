from main import load_data  # Or your data loader module
from model import build_cnn_model  # Import the model architecture
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load both datasets
letters_data = load_data(
    "/Users/OWNER/SideProjects/HandwritingRecognition/Database/train/emnist-digits.mat", 28, 28, None, True
)
digits_data = load_data(
    "/Users/OWNER/SideProjects/HandwritingRecognition/Database/train/emnist-letters.mat", 28, 28, None, True
)

# Extract training and testing data for letters
(training_images_letters, training_labels_letters) = letters_data[0]
(testing_images_letters, testing_labels_letters) = letters_data[1]

# Extract training and testing data for digits
(training_images_digits, training_labels_digits) = digits_data[0]
(testing_images_digits, testing_labels_digits) = digits_data[1]

# Adjust labels to ensure no overlap
training_labels_letters = np.array(training_labels_letters) + 9  # Offset letters to start from 10
testing_labels_letters = np.array(testing_labels_letters) + 9
training_labels_digits = np.array(training_labels_digits) - 1  # Ensure digits are 0-9
testing_labels_digits = np.array(testing_labels_digits) - 1

# Combine training data
training_images = np.concatenate((training_images_letters, training_images_digits), axis=0)
training_labels = np.concatenate((training_labels_letters, training_labels_digits), axis=0)

# Debugging shapes
print("Shape of training_images:", training_images.shape)
print("Shape of training_labels:", training_labels.shape)

# Combine testing data
testing_images = np.concatenate((testing_images_letters, testing_images_digits), axis=0)
testing_labels = np.concatenate((testing_labels_letters, testing_labels_digits), axis=0)

# Flatten labels to ensure they are 1D arrays
training_labels = training_labels.flatten()
testing_labels = testing_labels.flatten()

# Debugging shapes
print("Shape of flattened training_labels:", training_labels.shape)
print("Shape of flattened testing_labels:", testing_labels.shape)

# Filter training data
valid_train_indices = (training_labels <= 35)
print("Shape of valid_train_indices:", valid_train_indices.shape)

training_images = training_images[valid_train_indices]
training_labels = training_labels[valid_train_indices]


# Filter testing data
valid_test_indices = (testing_labels <= 35)
print("Shape of valid_test_indices:", valid_test_indices.shape)

testing_images = testing_images[valid_test_indices]
testing_labels = testing_labels[valid_test_indices]

# Verify unique labels
print("Filtered unique training labels:", np.unique(training_labels))
print("Filtered unique testing labels:", np.unique(testing_labels))

# One-hot encode labels
num_classes = 36  # 10 digits + 26 letters
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
model.save("cnn_emnist_alphanumeric_model.keras")

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from main import load_data

training_data, testing_data, mapping, num_classes = load_data(
    r"/Users/OWNER/SideProjects/HandwritingRecognition/Database/train/emnist-letters.mat", 28, 28, None, True
)

training_images, training_labels = training_data
testing_images, testing_labels = testing_data


# Define the model
def build_cnn_model(input_shape, num_classes):
    model = Sequential()

    # Convolutional layer 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional layer 2
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer for multi-class classification

    return model

# Build the model
input_shape = (28, 28, 1)  # Input dimensions (28x28 grayscale images)
num_classes = 26  # Total classes (e.g., A-Z for EMNIST letters)
model = build_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Ensure labels are in the correct range
training_labels = np.array(training_labels) - 1
testing_labels = np.array(testing_labels) - 1

# Verify unique labels
print("Unique labels in training set:", np.unique(training_labels))
print("Unique labels in testing set:", np.unique(testing_labels))

# Convert labels to one-hot encoding
y_train = to_categorical(training_labels, num_classes)
y_test = to_categorical(testing_labels, num_classes)

# Train the model
model.fit(training_images, y_train, batch_size=128, epochs=10, validation_data=(testing_images, y_test))

test_loss, test_accuracy = model.evaluate(testing_images, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

model.save("cnn_emnist_model.keras")

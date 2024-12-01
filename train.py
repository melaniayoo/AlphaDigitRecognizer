# train.py
from main import load_data
from model import build_cnn_model
from tensorflow.keras.utils import to_categorical

# Load data
training_data, testing_data, mapping, num_classes = load_data(
    r"C:\Users\mehak\HandwritingRecognition\Database\train\emnist-letters.mat", 28, 28, None, True
)

training_images, training_labels = training_data
testing_images, testing_labels = testing_data

# Convert labels to one-hot encoding
y_train = to_categorical(training_labels, num_classes)
y_test = to_categorical(testing_labels, num_classes)

# Build the model
input_shape = (28, 28, 1)
model = build_cnn_model(input_shape, num_classes)

# Train the model
model.fit(training_images, y_train, batch_size=128, epochs=10, validation_data=(testing_images, y_test))

# Save the model
model.save("cnn_emnist_model.h5")

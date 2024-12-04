from main import load_data  # Or your data loader module
from model import build_cnn_model  # Import the model architecture
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.image import resize
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, images, labels, batch_size, num_classes):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        return batch_images, to_categorical(batch_labels, self.num_classes)


# Load both datasets
letters_data = load_data(
    r"C:\Users\mehak\HandwritingRecognition\Database\train\emnist-letters.mat", target_size=(150, 150)
)
digits_data = load_data(
    r"C:\Users\mehak\HandwritingRecognition\Database\train\emnist-digits.mat", target_size=(150, 150)
)

# Combine datasets
(training_images_letters, training_labels_letters) = letters_data[0]
(testing_images_letters, testing_labels_letters) = letters_data[1]
(training_images_digits, training_labels_digits) = digits_data[0]
(testing_images_digits, testing_labels_digits) = digits_data[1]

training_images = np.concatenate((training_images_letters, training_images_digits), axis=0)
training_labels = np.concatenate((training_labels_letters, training_labels_digits), axis=0)
testing_images = np.concatenate((testing_images_letters, testing_images_digits), axis=0)
testing_labels = np.concatenate((testing_labels_letters, testing_labels_digits), axis=0)

# Adjust labels to avoid overlap
training_labels[:len(training_labels_letters)] += 9
testing_labels[:len(testing_labels_letters)] += 9

# One-hot encoding is handled in DataGenerator
batch_size = 64
num_classes = 36
training_generator = DataGenerator(training_images, training_labels, batch_size, num_classes)
testing_generator = DataGenerator(testing_images, testing_labels, batch_size, num_classes)

# Build the InceptionV3-based model
input_shape = (150, 150, 3)
model = build_cnn_model(input_shape, num_classes)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training_generator, validation_data=testing_generator, epochs=10)

# Save the model
model.save("inceptionv3_emnist_alphanumeric_model.keras")
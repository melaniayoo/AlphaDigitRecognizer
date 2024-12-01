from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Commented out your original function
# def build_cnn_model(input_shape, num_classes):
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

# Define a new function using MobileNet
def build_cnn_model(input_shape, num_classes):
    # Load MobileNet pre-trained on ImageNet
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

    # Build the custom model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Global pooling to reduce dimensions
        Dense(128, activation='relu'),  # Add a dense layer for feature extraction
        Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])

    return model

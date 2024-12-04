from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import RMSprop

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
# def build_cnn_model(input_shape, num_classes):
#     # Load MobileNet pre-trained on ImageNet
#     base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

#     # Build the custom model
#     model = Sequential([
#         base_model,
#         GlobalAveragePooling2D(),  # Global pooling to reduce dimensions
#         Dense(128, activation='relu'),  # Add a dense layer for feature extraction
#         Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
#     ])

#     return model

# Define the model architecture with InceptionV3
def build_cnn_model(input_shape, num_classes):
    # Load the base InceptionV3 model
    base_model = InceptionV3(
        input_shape=input_shape,
        include_top=False,  # Exclude the top classification layers
        weights='imagenet'  # Use pre-trained weights
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


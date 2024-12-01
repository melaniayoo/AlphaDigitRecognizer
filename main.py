import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
'''
def plot_image(image1, image2, title1, title2):
    # Create a figure with 1 row and 2 columns for side-by-side images
    plt.figure(figsize=(10, 5))

    # Plot the first image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first image
    plt.imshow(image1, cmap='gray')
    plt.title(title1)
    plt.axis('off')  # Turn off axis for better visualization

    # Plot the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second image
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    plt.axis('off')  # Turn off axis for better visualization

    # Display the images
    plt.show()

# Load the image (replace 'image_path' with your image file path)
image = cv2.imread(r"C:\Users\mehak\HandwritingRecognition\Database\train\A.jpeg")
if image is None:
    print("Error: Image not found!")
else:
    # 1. Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Min pixel value:", gray_image.min())
    print("Max pixel value:", gray_image.max())
    height, width = gray_image.shape[:2]
    print(f"Image size: {width} x {height} pixels")
    resized_image = cv2.resize(gray_image, (200, 200))
    ret, new_image = cv2.threshold(resized_image,127, 255, cv2.THRESH_BINARY)
    normalized_image = new_image / 255.0
    #plt.imshow(normalized_image, cmap='gray')
    plot_image(image, normalized_image, "Original", "Image After Thresholding")
    #plt.show()
'''
def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = rotate(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = rotate(testing_images[i])
    if verbose == True: print('')

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)

load_data("/Users/OWNER/SideProjects/HandwritingRecognition/Database/train/emnist-letters.mat", 28, 28, None, True)

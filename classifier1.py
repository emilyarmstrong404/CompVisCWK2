import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
import os
from collections import Counter
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score

# Converts into tiny image representation and extracts feature vectors
# Output: matrix with one normalized flattened tiny image per row
def get_tiny_images(image_paths):
    image_dimension = 16
    d = image_dimension**2
    N = len(image_paths)
    image_features = np.zeros((N, d))
    
    for i in range(N):
        # Read image in
        image_path = image_paths[i]
        image = Image.open(image_path).convert('L')
        
        # Center crop to square
        w, h = image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        right = left + side
        bottom = top + side
        image = image.crop((left, top, right, bottom))
        
        # Resize image to 16x16
        resized_image = image.resize((image_dimension, image_dimension), resample=Image.Resampling.LANCZOS)
        image_np = np.asarray(resized_image).astype(np.float32)
        
        # Create feature vector
        feature_vector = image_np.flatten()
        
        # Normalise feature vector to zero mean and unit length
        zero_meaned_feature_vector = feature_vector - np.mean(feature_vector)
        vector_norm = np.linalg.norm(zero_meaned_feature_vector)
        if vector_norm > 0:
            normed_feature_vector = zero_meaned_feature_vector / vector_norm
        else:
            normed_feature_vector = zero_meaned_feature_vector
               
        # Feature vector stored as a row in image_features
        image_features[i] = normed_feature_vector
    return image_features


# Makes class predictions for test data by finding closest training image
# Ouput: List of class predictions for all test images
def k_nearest_neighbours(train_image_features, train_labels, test_image_features, k):
    # Finding the pythagorean distance between the test images and the train images
    D = pairwise_distances(test_image_features, train_image_features, metric='euclidean')
    
    # Finding the indexes of the closest train images for each of the test images
    dist_sorted_train_instances = np.argsort(D, axis=1)
    nearest_indexes = dist_sorted_train_instances[:, :k] #[:, :k] means all rows up to k
    
    # Predict class for each test sample using majority vote
    predicted_classes = []
    for index in nearest_indexes:
        
        # Get the labels of the k nearest neighbors
        nearest_labels = np.array(train_labels)[index]
        
        # Majority vote
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predicted_classes.append(str(most_common))
    
    # Predicting the class label for each test image using the index and training labels
    return predicted_classes


# Working on the assumption we can't use PyTorch??
def load_train_dataset(root_dir):
    image_paths = []
    image_labels = []
    
    # Loop through each class folder (folder names are the labels for images within)
    for label in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label)
        if os.path.isdir(class_dir):
            
            # Loop through all images in this class folder
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg')):
                    filepath = os.path.join(class_dir, filename)
                    image_paths.append(filepath)
                    image_labels.append(label)
    return image_paths, image_labels


# Same as load_train_dataset but with no subfolders or labels
def load_test_dataset(root_dir):
    test_paths = []
    for filename in sorted(os.listdir(root_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(root_dir, filename)
            test_paths.append(filepath)
    return test_paths
    
    
# Load and extract training data
train_root = "training"
train_image_paths, train_labels = load_train_dataset(train_root)
train_image_features = get_tiny_images(train_image_paths)


# Load and extract testing data
test_root = "testing"
test_image_paths = load_test_dataset(test_root)
test_image_features = get_tiny_images(test_image_paths)

# Classify using 13 nearest neighbours
predicted_labels = k_nearest_neighbours(train_image_features, train_labels, test_image_features, k=13)

print("Predicted Labels", predicted_labels[:5])


# Code used to find optimal k of 13
# X_train, X_val, y_train, y_val = train_test_split(
#     train_image_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
# )
# predicted_val_labels = k_nearest_neighbours(X_train, y_train, X_val, k=13)
# print("Predicted Labels", predicted_val_labels[:5])
# accuracy = accuracy_score(y_val, predicted_val_labels)
# print(f"Validation Accuracy: {accuracy:.4f}")
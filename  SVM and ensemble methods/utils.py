import os
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEED = 6  # DO NOT CHANGE
rng = np.random.default_rng(SEED)  # DO NOT CHANGE

class MNISTPreprocessor:
    def __init__(self, data_dir, config=None):
        """
        Initializes the MNISTPreprocessor.

        Parameters:
        - data_dir: str, the base directory where the MNIST images are stored.
        - config: config specific to your pre-processing methods, e.g., image size, normalization strategy
        """
        self.data_dir = data_dir
        self.classes = [str(i) for i in range(10)]
        self.image_paths = self._get_image_paths()
        
        # Default configuration for preprocessing
        self.config = {
            'resize': (28, 28),
            'normalize': True,
            'standardize': True,
            'pca': config['use_pca']
        }
        
    def _get_image_paths(self):
        """
        Collects all image file paths from the data directory for the specified classes.

        Returns:
        - image_paths: list of tuples (image_path, label)
        """
        image_paths = []
        for label in self.classes:
            class_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg'):
                    image_path = os.path.join(class_dir, fname)
                    image_paths.append((image_path, int(label)))
        return image_paths

    def _preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Resize if configured
        if 'resize' in self.config and self.config['resize']:
            image = image.resize(self.config['resize'])
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32)
        #print(f"Original Image (first 10 pixels): {image_array.flatten()[:50]}")
        # Check the range before normalization
        #print(f"Min/Max before normalization: {image_array.min()}, {image_array.max()}")
        
        if self.config.get('normalize', True):
            image_array /= 255.0  # Normalize to [0, 1]
        #print(f"Normaliseed Image (first 10 pixels): {image_array.flatten()[:50]}")
        # Check the range after normalization
        #print(f"Min/Max after normalization: {image_array.min()}, {image_array.max()}")
        
        return image_array.flatten()  # Flatten to a 1D vector
    def batch_generator(self, batch_size=32, shuffle=True):
        """
        Generator that yields batches of pre-processed images and their labels.

        Parameters:
        - batch_size: int, the number of images per batch.
        - shuffle: bool, whether to shuffle the data before each epoch.

        Yields:
        - X_batch: numpy array of shape (batch_size, flattened_image_size)
        - y_batch: numpy array of shape (batch_size,)
        """
        image_paths = self.image_paths.copy()
        if shuffle:
            np.random.shuffle(image_paths)

        num_samples = len(image_paths)
        for offset in range(0, num_samples, batch_size):
            batch_samples = image_paths[offset:offset+batch_size]
            X_batch = []
            y_batch = []
            for image_path, label in batch_samples:
                image_vector = self._preprocess_image(image_path)
                print(image_vector[:100])
                print(label)
                X_batch.append(image_vector)
                y_batch.append(label)
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch

    def get_all_data(self):
        """
        Loads all data into memory with pre-processing applied.

        Returns:
        - X: numpy array of shape (num_samples, flattened_image_size)
        - y: numpy array of shape (num_samples,)
        """
        X = []
        y = []
        #print("Resizing and Normalising")
        for image_path, label in self.image_paths:
            image_vector = self._preprocess_image(image_path)
            X.append(image_vector)
            y.append(label)
            #print(image_vector)
            #print(y)
        X = np.array(X)
        y = np.array(y)
        # Optionally standardize if specified in config
        #print("Standardizing")
        if self.config.get('standardize', True):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        #print("Reducing dimensionality")
        if self.config.get('pca', True):
            pca = PCA(n_components=0.95)
            X = pca.fit_transform(X)
        return X, y

def filter_dataset(X, y, entry_number_digit):
    """
    Filters the dataset to include only the target and three surrounding digits.

    Parameters:
    - X: numpy array of input data
    - y: numpy array of labels
    - entry_number_digit: last digit of the entry number
    
    Returns:
    - Filtered X and y arrays.
    """
    target_digits = [
        entry_number_digit % 10,
        (entry_number_digit - 1) % 10,
        (entry_number_digit + 1) % 10,
        (entry_number_digit + 2) % 10
    ]

    condition = np.isin(y, target_digits)
    X = X[condition]
    y = y[condition]
    #print(f"Labels (first 10): {y[:10]}")
    return X, y

def convert_labels_to_svm_labels(arr, svm_pos_label=0):
    """
    Converts labels to binary SVM labels.

    Parameters:
    - arr: numpy array of original labels
    - svm_pos_label: the label to be considered as positive class
    
    Returns:
    - Converted numpy array with SVM-compatible labels (1 and -1).
    """

    return np.where(arr == svm_pos_label, 1, -1)

def get_metrics_as_dict(preds, true):
    """
    Generates a classification report in dictionary format.

    Parameters:
    - preds: numpy array of predictions
    - true: numpy array of true labels

    Returns:
    - Dictionary of classification metrics.
    """
    assert(preds.shape == true.shape), "Shape Mismatch. Assigning 0 score"
    report = classification_report(true, preds, output_dict=True)
    return report

def val_score(preds, true):
    """
    Calculates F1 score for validation.

    Parameters:
    - preds: numpy array of predictions
    - true: numpy array of true labels

    Returns:
    - F1 score for the macro average.
    """
    assert(preds.shape == true.shape), "Shape Mismatch. Assigning 0 score"
    report = classification_report(true, preds, output_dict=True,zero_division=1)
    return report["macro avg"]['f1-score']

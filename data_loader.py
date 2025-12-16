import jax.numpy as jnp
import numpy as np
import gzip
import os
from typing import Tuple


def load_mnist_images(filename):
    """Load MNIST images from idx3-ubyte file"""
    with gzip.open(filename, 'rb') as f:
        # Read magic number
        magic = np.frombuffer(f.read(4), dtype=np.dtype('>i4'), count=1)[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number for MNIST image file: {magic}")

        # Read dimensions
        num_images = np.frombuffer(f.read(4), dtype=np.dtype('>i4'), count=1)[0]
        rows = np.frombuffer(f.read(4), dtype=np.dtype('>i4'), count=1)[0]
        cols = np.frombuffer(f.read(4), dtype=np.dtype('>i4'), count=1)[0]

        # Read image data
        buffer = f.read(rows * cols * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)

        return images


def load_mnist_labels(filename):
    """Load MNIST labels from idx1-ubyte file"""
    with gzip.open(filename, 'rb') as f:
        # Read magic number
        magic = np.frombuffer(f.read(4), dtype=np.dtype('>i4'), count=1)[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number for MNIST label file: {magic}")

        # Read number of labels
        num_labels = np.frombuffer(f.read(4), dtype=np.dtype('>i4'), count=1)[0]

        # Read label data
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)

        return labels


def load_mnist_subset(num_samples: int = 1000, input_dim: int = 784, data_dir: str = None) -> Tuple:

    # If data_dir is not provided, use current working directory
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'data', 'MNIST', 'raw')

    # MNIST file names
    train_images_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')

    # Check if files exist
    if not os.path.exists(train_images_file):
        raise FileNotFoundError(f"MNIST images file not found: {train_images_file}")
    if not os.path.exists(train_labels_file):
        raise FileNotFoundError(f"MNIST labels file not found: {train_labels_file}")

    print("Loading MNIST images from local files...")
    images = load_mnist_images(train_images_file)
    labels = load_mnist_labels(train_labels_file)

    # Flatten images and normalize to [0, 1]
    images_flat = images.reshape(images.shape[0], -1).astype(np.float64) / 255.0

    # Take random subset
    rng = np.random.default_rng(42)
    indices = rng.choice(len(images_flat), num_samples, replace=False)

    A_subset = images_flat[indices]
    Y_subset = labels[indices].astype(np.float64)

    # Convert to JAX arrays
    A = jnp.array(A_subset, dtype=jnp.float64)
    Y = jnp.array(Y_subset, dtype=jnp.float64)
    Y_normalized = (Y_subset - 4.5) / 4.5

    print(f"Loaded MNIST subset: {A.shape[0]} samples, {A.shape[1]} features")
    print(f"Target range: {Y.min():.0f} to {Y.max():.0f}")

    return A, Y_normalized
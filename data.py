import numpy as np
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os


# Cameras we will use
cameras = ['left', 'center', 'right']
cameras_steering_correction = [.25, 0., -.25]

def preprocess(image, top_offset=.375, bottom_offset=.125):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (66, 200, 3))
    return image

def generate_samples(data, root_path, augment=True):
    """
    Keras generator yielding batches of training/validation data.
    Applies data augmentation pipeline if `augment` is True.
    """
    while True:
        indices = np.random.permutation(len(data))
        batch_size = 128
        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]

            x_batch = []
            y_batch = []

            for i in batch_indices:
                # Select camera (random for augmentation, center only for validation)
                camera = np.random.randint(len(cameras)) if augment else 1

                # Load image and apply camera correction
                image_path = os.path.join(root_path, data[cameras[camera]].values[i].strip())
                image = mpimg.imread(image_path).copy()
                angle = data.steering.values[i] + cameras_steering_correction[camera]

                if augment:
                    # Add random shadow
                    h, w = image.shape[0], image.shape[1]
                    [x1, x2] = np.random.choice(w, 2, replace=False)
                    k = h / (x2 - x1)
                    b = -k * x1
                    for row in range(h):
                        c = int((row - b) / k)
                        image[row, :c, :] = (image[row, :c, :] * 0.5).astype(np.int32)

                # Crop and resize to (66, 200)
                v_delta = 0.05 if augment else 0
                image = preprocess(
                    image,
                    top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
                )

                x_batch.append(image)
                y_batch.append(angle)

            # Convert to numpy arrays
            x_batch = np.array(x_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.float32)

            # Random horizontal flip for augmentation
            if augment:
                flip_indices = random.sample(range(x_batch.shape[0]), x_batch.shape[0] // 2)
                x_batch[flip_indices] = x_batch[flip_indices, :, ::-1, :]
                y_batch[flip_indices] = -y_batch[flip_indices]

            yield x_batch, y_batch

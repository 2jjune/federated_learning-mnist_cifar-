import os
import set_epoch_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = "data/"
test_dir = os.path.join(base_dir, "test")


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(32, 32),
        batch_size=64,
        color_mode = 'rgb',
        class_mode='categorical')
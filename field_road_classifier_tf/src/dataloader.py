import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class FieldRoadDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_dir = data_path + '/train'
        self.val_dir = data_path + '/val'
        self.batch_size = 32
        self.image_size = (224, 224)
        self.num_classes = len(os.listdir(self.train_dir))
        self.train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.3,
            height_shift_range=0.3,
            dtype='float32',
            rescale=1.0 / 255,
        )
        self.val_datagen = ImageDataGenerator(dtype='float32', rescale=1.0 / 255,)

    def get_train_data(self):
        train_data = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
        print(train_data)
        return train_data

    def get_val_data(self):
        val_data = self.val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
        return val_data

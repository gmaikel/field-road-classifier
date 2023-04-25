import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


class FieldRoadPredictor:
    """
        This "FieldRoadPredictor" class allows loading a pre-trained 
        image classification model and predicting whether 
        a given image represents a field or a road using this model.
    """
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ["field", "road"]
        self.input_shape = self.model.layers[0].input_shape[1:3]

    def predict(self, image_path):
        img = image.load_img(image_path, target_size=self.input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = self.model.predict(img_array)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_class_names = [self.class_names[i] for i in predicted_classes]
        return predicted_class_names

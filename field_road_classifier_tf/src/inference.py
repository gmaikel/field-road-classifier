import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


class FieldRoadInference:
    """
    The "FieldRoadInference" class loads a trained model and evaluates
    its performance on a set of test images of field roads using binary classification.
    It generates a confusion matrix and calculates accuracy to evaluate
    the model's performance. It also saves the confusion matrix as an image file.
    """

    def __init__(self, data_path, model_path, output_path):
        self.data_path = data_path
        self.output_path = os.path.join(
            output_path, os.path.basename(model_path).split(".")[0]
        )
        self.model = tf.keras.models.load_model(model_path)
        self.num_classes = len(os.listdir(data_path))
        self.class_names = ["field", "road"]  # Add class names here

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def evaluate(self):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255
        )

        test_data = test_datagen.flow_from_directory(
            self.data_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode="binary",
            shuffle=False,
        )

        # Predict the labels of the test images using the loaded model
        y_pred_prob = self.model.predict(test_data)

        # Convert the predicted probabilities to binary predictions
        y_pred = np.zeros_like(y_pred_prob)
        y_pred[y_pred_prob >= 0.5] = 1

        # Get the true labels of the test images
        y_true = test_data.labels

        # Compute the confusion matrix
        cm = tf.math.confusion_matrix(y_true, y_pred)
        # Calculate true positives, true negatives, false positives, false negatives
        tp = tf.linalg.diag_part(cm)[1]
        tn = tf.linalg.diag_part(cm)[0]
        fp = tf.reduce_sum(cm, axis=0)[1] - tp
        fn = tf.reduce_sum(cm, axis=1)[1] - tp

        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Save the confusion matrix to file
        cm_fig = self.plot_confusion_matrix(cm)
        cm_fig.savefig(os.path.join(self.output_path, "confusion_matrix.png"))

        # Save some false prediction images
        # self.save_false_prediction_images(test_data, y_true, y_pred)

        return accuracy, cm

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        thresh = (
            cm.numpy().max() / 2.0
        )  # convert to NumPy array and then get the max value

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        cm_fig = plt.gcf()
        return cm_fig

    # def save_false_prediction_images(self, test_data, y_true, y_pred):
    #     """
    #     Save images of false predictions.
    #     """
    #     classes = ['field', 'road']
    #
    #     if not os.path.exists(os.path.join(self.output_path, 'false_predictions')):
    #         os.makedirs(os.path.join(self.output_path, 'false_predictions'))
    #
    #     for i, (img, true_label) in enumerate(zip(test_data, y_true)):
    #         true_label = classes[int(true_label)]
    #         pred_label = classes[int(y_pred[i])]
    #         if pred_label != true_label:
    #             filename = f"{true_label}_as_{pred_label}_img_{i}.png"
    #             filepath = os.path.join(self.output_path, 'false_predictions', filename)
    #             plt.imsave(filepath, img[0])

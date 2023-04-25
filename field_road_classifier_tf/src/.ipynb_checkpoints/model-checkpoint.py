import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class FieldRoadModel:
    def __init__(
            self,
            epochs,
            model_name,
            input_shape,
            alpha=0.25,
            gamma=2.0,
            learning_rate=0.001,
    ):
        self.input_shape = input_shape
        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_name=model_name

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.7))

        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(
            from_logits=False,
            alpha=self.alpha,
            gamma=self.gamma
        ),
            optimizer=SGD(learning_rate=learning_rate),
            metrics=['accuracy'],
        )

    def fit(self, train_data, val_data):
        # Create the checkpoint directory if it does not exist
        checkpoint_dir = './checkpoints/'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Define the checkpoint filename
        checkpoint_path = checkpoint_dir + f'{self.model_name}/best_{self.model_name}_model.h5'

        # Define the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=0
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=150)

        history = self.model.fit(
            train_data,
            epochs=self.epochs,
            validation_data=val_data,
            callbacks=[early_stop, checkpoint_callback]
        )
        return history

    def evaluate(self, test_data):
        loss, accuracy = self.model.evaluate(test_data)
        return loss, accuracy

    def plot_history(self, history):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(history.history['accuracy'], label='train')
        ax[0].plot(history.history['val_accuracy'], label='validation')
        ax[0].set_title('Model Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='train')
        ax[1].plot(history.history['val_loss'], label='validation')
        ax[1].set_title('Model Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()

        plt.show()



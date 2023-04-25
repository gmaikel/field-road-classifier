import argparse
import random
import numpy as np
import tensorflow as tf

# Local libraries
from src import (
    FieldRoadModel,
    FieldRoadDataLoader,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model(
    input_shape, alpha, gamma, learning_rate, data_path, epochs, model_name
):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        except RuntimeError as error:
            print(error)

    model = FieldRoadModel(
        input_shape=input_shape,
        alpha=alpha,
        gamma=gamma,
        learning_rate=learning_rate,
        epochs=epochs,
        model_name=model_name,
    )

    loader = FieldRoadDataLoader(data_path)
    train_data = loader.get_train_data()
    val_data = loader.get_val_data()

    history = model.fit(train_data, val_data)
    test_loss, test_accuracy = model.evaluate(val_data)

    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    model.plot_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Field Road Model")
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=[224, 224, 3],
        help="The input shape of the model",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="The alpha parameter for the Sigmoid Focal Cross-Entropy loss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="The gamma parameter for the Sigmoid Focal Cross-Entropy loss",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="The learning rate for the optimizer",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/mgali/Desktop/dataset",
        help="The path to the data directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed for random number generator"
    )
    parser.add_argument(
        "--model_name", type=str, default=1, help="Name of model saving"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    train_model(
        input_shape=args.input_shape,
        alpha=args.alpha,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        data_path=args.data_path,
        epochs=args.epochs,
        model_name=args.model_name,
    )

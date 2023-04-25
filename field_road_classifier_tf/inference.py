import argparse
from src import FieldRoadInference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/mgali/Desktop/dataset/val",
        help="path to validation data directory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/v2/best_v2_model.h5",
        help="path to saved model file",
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="path to output directory"
    )

    args = parser.parse_args()

    # Instantiate the Inference class
    inference = FieldRoadInference(
        data_path=args.data_path,
        model_path=args.model_path,
        output_path=args.output_path,
    )

    accuracy, cm = inference.evaluate()

    print(f"Accuracy: {accuracy}")
    print(f"Confusion matrix:\n{cm}")

import argparse
from src import FieldRoadPredictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="/home/mgali/Desktop/dataset/val/fields/1.jpeg",
        help="path to image to predict",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/v2/best_v2_model.h5",
        help="path to saved model file",
    )

    args = parser.parse_args()

    # Instantiate the Inference class
    predictor = FieldRoadPredictor(
        model_path=args.model_path,
    )

    print(predictor.predict(image_path=args.image_path))

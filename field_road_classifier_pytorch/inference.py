from inference import FieldsRoadsClassifier

if __name__=='__main__':

    img_path = '/home/mgali/PycharmProjects/trimble/data/dataset/train/fields/2.jpg'
    model_path = '/home/mgali/PycharmProjects/trimble/tb_logs/field_roads_model/version_1/checkpoints/epoch=29-step=150.ckpt'

    classifier = FieldsRoadsClassifier(model_path=model_path)

    print(classifier.classify(image_path=img_path))

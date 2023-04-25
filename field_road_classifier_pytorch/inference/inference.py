import torch
from torchvision import transforms
from PIL import Image
from models import FieldsRoadsModel


class FieldsRoadsClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = FieldsRoadsModel.load_from_checkpoint(model_path).to(self.device)
        self.model.freeze()

    def classify(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)
            if output > 0.5:
                return "road"
            else:
                return "field"
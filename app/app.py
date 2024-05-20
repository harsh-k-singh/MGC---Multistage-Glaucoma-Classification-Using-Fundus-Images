from flask import Flask, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from models import CA_Net, GlaucomaDataset  # Import from models.py

app = Flask(__name__)

# Load your pre-trained model
class_labels = ['early_glaucoma', 'normal_control', 'advanced_glaucoma']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CA_Net(num_classes=len(class_labels)).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        prediction = class_labels[predicted.item()]

    return prediction

if __name__ == '__main__':
    app.run(debug=True)
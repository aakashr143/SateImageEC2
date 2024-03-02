import torch
import torch.nn.functional as f
from flask import Flask, jsonify, request
from torchvision import transforms
from PIL import Image
import urllib.request
import uuid
import os

app = Flask(__name__)
model = torch.load('model.pth')
model.eval()

classes = [
    "Agricultural",
    "Airport",
    "Bareland",
    "Beach",
    "Bridge",
    "Buildings",
    "Cemetery",
    "Center",
    "Chaparral",
    "Coastal Mansion",
    "Desert",
    "Forest",
    "Freeway",
    "Golf Course",
    "Harbor",
    "Industrial Area",
    "Intersection",
    "Meadow",
    "Mobile Home Park",
    "Oil Gas Field",
    "Park",
    "Parking",
    "Railway",
    "Residential",
    "Runway",
    "Shipping Yard",
    "Sparse Residential",
    "Square",
    "Stadium",
    "Storage Tank",
    "Thermal Power Station",
    "Wastewater Treatment Plant",
    "Wetland"
]


def download_image_from_url(url: str) -> str | None:
    try:
        img_id = str(uuid.uuid1())
        urllib.request.urlretrieve(url, f"{img_id}.png")
        return f"{img_id}.png"
    except Exception as e:
        print(e)
        return None


def get_image_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')

    os.remove(image_path)

    return transform(img).unsqueeze(0)


def get_prediction(image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)
        probabilites = f.softmax(predictions, dim=1)

        return predictions.argmax(dim=1).item(), [t.item() for t in probabilites[0]]


@app.route('/')
def hello():
    return 'SateImageClassification'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if not data['image_url']:
            return jsonify({
                'error': True,
                'message': 'Missing image url'
            })

        image_path = download_image_from_url(data['image_url'])

        if image_path is None:
            return jsonify({
                'error': True,
                'message': 'Unable to download image'
            })

        image_tensor = get_image_tensor(image_path)
        pred_idx, probs = get_prediction(image_tensor)

        return jsonify({
            'error': False,
            'message': {
                'class': classes[pred_idx],
                'probablitiy': probs
            }
        })

    except Exception as e:
        return jsonify({
            'error': True,
            'message': e
        })


if __name__ == '__main__':
    app.run()

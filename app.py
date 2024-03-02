from flask import Flask, jsonify, request
import io
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Satellite Image Classification'




if __name__ == '__main__':
    app.run()
from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)

# 画像の前処理関数
def preprocess_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# 画像特徴を生成する関数
def generate_features(image_tensor):
    # ここでは仮にResNetモデルを使用
    model = models.resnet50(pretrained=True)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    # outputsを解析し、特徴をテキストで返す
    # この部分は具体的なモデルと要件に応じて異なる
    return "画像の特徴"

@app.route('/generate-features', methods=['POST'])
def handle_generate_features():
    data = request.json
    image_url = data['imageUrl']
    image_tensor = preprocess_image(image_url)
    features = generate_features(image_tensor)
    return jsonify({'features': features})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)

# Cargar modelo YOLOv5 desde archivo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='impresion.pt', force_reload=True)

@app.route("/inferencia", methods=["POST"])
def inferencia():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    image_file = request.files["file"]
    img = Image.open(image_file.stream)

    # Realiza inferencia
    results = model(img)

    # Convierte resultados a JSON
    result_json = results.pandas().xyxy[0].to_json(orient="records")
    return result_json
import argparse

import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision.models import ResNet101_Weights

app = Flask(__name__)

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path to the inference model")
parser.add_argument("--device", type=str, help="cpu or cuda")
args = parser.parse_args()

# Modell laden (ersetze 'dein_modell.pth' mit dem Pfad zu deinem Modell)
model = torch.load(
    args.model_path, weights_only=False, map_location=torch.device(args.device)
)
model.eval()  # Wichtig für Inferenz


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "Keine Datei hochgeladen"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Keine Datei ausgewählt"})

        try:
            # Bild verarbeiten
            image = Image.open(file).convert("RGB")

            longer_side = max(image.size[0], image.size[1])
            new_size = (longer_side, longer_side)
            new_im = Image.new("RGB", new_size)
            box = tuple((n - o) // 2 for n, o in zip(new_size, image.size))
            new_im.paste(image, box)

            preprocessed_image = ResNet101_Weights.IMAGENET1K_V2.transforms()(new_im)

            image_final = preprocessed_image.unsqueeze(0)  # Batch-Dimension hinzufügen

            # Inferenz
            with torch.no_grad():
                output = model(image_final)
                predicted_class = torch.argmax(output, 1)

            # Ergebnis zurückgeben
            return jsonify({"class": int(predicted_class)})

        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"error": "Falsche Methode"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

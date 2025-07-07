from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("D:\\Deep Learning\\Trained Model\\oxford_flowers_model.h5")

IMG_SIZE = 224

CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura",
    "geranium", "orange dahlia", "pink-yellow dahlia?", "cautleya spicata",
    "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily", "common tulip", "wild rose"
]

def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr = np.array(img) / 255.0  
    return np.expand_dims(arr, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"error": "No image found"}), 400

        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))

        input_tensor = preprocess(image)
        predictions = model.predict(input_tensor)
        top_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        response = {
            "prediction": f"{CLASS_NAMES[top_idx]} ({confidence:.2f}%)"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

from model import preprocess_image, load_class_names

UPLOAD_FOLDER = 'uploads'
FEEDBACK_FOLDER = 'feedback'
MODEL_PATH = 'models/model.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')

# load class names
CLASS_NAMES = load_class_names('models/class_names.txt')

# try to load model at startup (support .keras or .h5)
model = None
if os.path.exists('models/model.keras'):
    try:
        model = tf.keras.models.load_model('models/model.keras')
        print('Loaded model from models/model.keras')
    except Exception as e:
        print('Failed loading models/model.keras:', e)
elif os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print('Loaded model from', MODEL_PATH)
    except Exception as e:
        print('Failed loading model:', e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'no image file provided'}), 400
    f = request.files['image']
    if f.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    dest = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(dest)
    if model is None:
        return jsonify({'error': 'model not available. Train and save model to models/model.h5 or models/model.keras'}), 500
    img = preprocess_image(dest, target_size=(224, 224))
    preds = model.predict(np.expand_dims(img, axis=0))[0]
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    confidence = float(preds[idx])
    return jsonify({'label': label, 'confidence': confidence})


@app.route('/feedback', methods=['POST'])
def feedback():
    # Accepts an image file and a corrected label, saves to feedback/<label>/
    if 'image' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'image file and label are required'}), 400
    f = request.files['image']
    label = request.form['label'].strip()
    if f.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    if label == '':
        return jsonify({'error': 'empty label'}), 400
    safe_label = secure_filename(label)
    target_dir = os.path.join(FEEDBACK_FOLDER, safe_label)
    os.makedirs(target_dir, exist_ok=True)
    import time
    filename = secure_filename(f.filename)
    filename = f"{int(time.time())}_{filename}"
    dest = os.path.join(target_dir, filename)
    f.save(dest)
    return jsonify({'status': 'saved', 'path': dest})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
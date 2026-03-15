Coral Health Detector

Quick start:
1. Create a Python virtualenv and install dependencies:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Prepare data:
   Arrange images like:
   data/train/healthy/*.jpg
   data/train/unhealthy/*.jpg
   data/val/healthy/*.jpg
   data/val/unhealthy/*.jpg

3. Train a model:
   python train.py --data-dir data --epochs 5

   This saves the model to models/model.h5 and models/class_names.txt

4. Run the app:
   python app.py

5. Open http://localhost:5000 in your browser and upload coral images.

Notes:
- Use many diverse images of healthy coral for better generalization.
- For production, consider using GPU, batching, and a proper model registry.

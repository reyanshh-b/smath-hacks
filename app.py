"""
CoralScan - Coral Disease Detector with AI
app.py: Gradio web app that runs coral disease detection on uploaded video.

Run with:
    python app.py
"""

import cv2
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image
import tempfile
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "coral_model.h5"
IMG_SIZE     = (224, 224)
FRAME_SKIP   = 5       # process every Nth frame to keep things fast
CLASSES      = ["Healthy Coral", "Bleached Coral", "Dead Coral"]
COLORS = {
    "Healthy Coral":  (50,  205, 50),    # green
    "Bleached Coral": (255, 165, 0),     # orange
    "Dead Coral":     (220, 20,  60),    # red
}
DESCRIPTIONS = {
    "Healthy Coral":  "This coral appears healthy. Normal coloration and polyp activity detected.",
    "Bleached Coral": "Bleaching detected. The coral has expelled its algae, likely due to thermal stress or pollution.",
    "Dead Coral":     "This coral appears to be dead. Skeletal structure is visible with no living tissue.",
}

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_frame(frame):
    """Convert a BGR OpenCV frame to a normalized RGB tensor."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_frame(frame):
    """Return the predicted class label and confidence for a single frame."""
    tensor = preprocess_frame(frame)
    preds  = model.predict(tensor, verbose=0)[0]
    idx    = np.argmax(preds)
    return CLASSES[idx], float(preds[idx])


def annotate_frame(frame, label, confidence):
    """Draw a colored label + confidence bar onto the frame."""
    h, w = frame.shape[:2]
    color = COLORS[label]

    # colored border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness=6)

    # label background
    text       = f"{label}  {confidence * 100:.1f}%"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (10, 10), (tw + 20, th + 20 + baseline), color, cv2.FILLED)
    cv2.putText(frame, text, (15, th + 15), font, font_scale, (255, 255, 255), thickness)

    # confidence bar
    bar_y     = h - 25
    bar_total = w - 60
    cv2.rectangle(frame, (30, bar_y), (30 + bar_total, bar_y + 12), (50, 50, 50), cv2.FILLED)
    cv2.rectangle(frame, (30, bar_y), (30 + int(bar_total * confidence), bar_y + 12), color, cv2.FILLED)

    return frame


def process_video(video_path):
    """
    Process an uploaded video file frame by frame.
    Returns path to the annotated output video and a text summary.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: could not open video file."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path   = tempfile.mktemp(suffix=".mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx      = 0
    current_label  = CLASSES[0]
    current_conf   = 0.0
    label_counts   = {c: 0 for c in CLASSES}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # only run inference every FRAME_SKIP frames
        if frame_idx % FRAME_SKIP == 0:
            current_label, current_conf = predict_frame(frame)
            label_counts[current_label] += 1

        annotated = annotate_frame(frame.copy(), current_label, current_conf)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    # build summary
    dominant = max(label_counts, key=label_counts.get)
    summary  = (
        f"Analysis complete. {frame_idx} frames processed.\n\n"
        f"Dominant classification: {dominant}\n"
        f"{DESCRIPTIONS[dominant]}\n\n"
        f"Frame breakdown:\n"
        + "\n".join([f"  {k}: {v} frames" for k, v in label_counts.items()])
    )

    return out_path, summary


def process_image(image):
    """
    Process a single uploaded image.
    Returns annotated image and a text summary.
    """
    if image is None:
        return None, "Please upload an image."

    # convert PIL to BGR numpy array
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    label, conf = predict_frame(frame)
    annotated   = annotate_frame(frame.copy(), label, conf)
    annotated   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    summary = (
        f"Classification: {label} ({conf * 100:.1f}% confidence)\n\n"
        f"{DESCRIPTIONS[label]}"
    )

    return Image.fromarray(annotated), summary


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="CoralScan", theme=gr.themes.Ocean()) as demo:

    gr.Markdown(
        """
        # 🪸 CoralScan — Coral Disease Detector
        **Upload an underwater video or photo and CoralScan will analyze the coral health using AI.**
        Detects: Healthy Coral | Bleached Coral | Dead Coral
        """
    )

    with gr.Tabs():

        # Video tab
        with gr.TabItem("Video Detection"):
            gr.Markdown("Upload underwater footage and CoralScan will annotate each frame.")
            with gr.Row():
                video_input  = gr.Video(label="Upload Video")
                video_output = gr.Video(label="Annotated Output")
            video_summary = gr.Textbox(label="Analysis Summary", lines=8)
            video_btn     = gr.Button("Analyze Video", variant="primary")
            video_btn.click(
                fn=process_video,
                inputs=video_input,
                outputs=[video_output, video_summary],
            )

        # Image tab
        with gr.TabItem("Image Detection"):
            gr.Markdown("Upload a single underwater photo for a quick classification.")
            with gr.Row():
                img_input  = gr.Image(type="pil", label="Upload Image")
                img_output = gr.Image(type="pil", label="Annotated Output")
            img_summary = gr.Textbox(label="Result", lines=5)
            img_btn     = gr.Button("Analyze Image", variant="primary")
            img_btn.click(
                fn=process_image,
                inputs=img_input,
                outputs=[img_output, img_summary],
            )

    gr.Markdown(
        """
        ---
        Built at **SMathHacks 2026** | Theme: Under the Sea
        Model: MobileNetV2 fine-tuned on the Coral Reef Health Dataset (Kaggle)
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)   # share=True gives a public link for the demo

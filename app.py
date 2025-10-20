import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# ==============================================================================
# 0. App Configuration (Robust Path Handling)
# ==============================================================================

# --- Get the directory where this app.py script is located ---
# This assumes app.py, models/, and code/ are all inside the project root directory.
PROJECT_ROOT = Path(__file__).parent.resolve()

# --- Define paths relative to the project root ---
MODELS_DIR = PROJECT_ROOT / "models"
CODE_DIR = PROJECT_ROOT / "code"

# --- Add the 'code' directory to Python's path to allow imports ---
if CODE_DIR.exists():
    sys.path.append(str(PROJECT_ROOT))
else:
    # Handle the case where the 'code' folder might be inside another subfolder, just in case.
    # This makes the code more robust.
    # For your structure, the first part of the 'if' will execute.
    alt_code_dir = PROJECT_ROOT / "glaucoma_app" / "code"
    if alt_code_dir.exists():
        sys.path.append(str(PROJECT_ROOT / "glaucoma_app"))
        CODE_DIR = alt_code_dir

# --- Model-specific optimal thresholds ---
MODEL_THRESHOLDS = {
    "A_Benchmark_EfficientNetB3_model.keras": 0.03,
    "B_SRA_EfficientNetB3_model.keras": 0.79,
    "C_SRA_Attention_EfficientNetB3_model.keras": 0.13,
}

# ==============================================================================
# 1. Load Models and Dependencies
# ==============================================================================
print("--- Initializing Glaucoma Detection App ---")

# --- Load the custom CBAM module ---
try:
    from code.cbam import cbam_block, MaxAcrossChannel, MeanAcrossChannel

    custom_objects_dict = {
        'cbam_block': cbam_block,
        'MaxAcrossChannel': MaxAcrossChannel,
        'MeanAcrossChannel': MeanAcrossChannel
    }
    print("   - CBAM custom module loaded successfully.")
except ImportError:
    print(f"   [WARNING] Could not find cbam.py in '{CODE_DIR}'. Attention models may fail to load.")
    custom_objects_dict = {}

# --- Discover model files dynamically ---
if not MODELS_DIR.exists():
    raise FileNotFoundError(f"The 'models' directory was not found. Expected at: {MODELS_DIR}")

available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
if not available_models:
    raise FileNotFoundError(f"No .keras model files found in the '{MODELS_DIR}' directory.")
print(f"   - Found {len(available_models)} models: {available_models}")

# --- Cache for loaded models to improve efficiency ---
loaded_models_cache = {}


def get_model(model_name):
    """Dynamically loads and caches the selected model."""
    if model_name not in loaded_models_cache:
        print(f"   - Loading model for the first time: {MODELS_DIR}/{model_name}")
        model_path = os.path.join(MODELS_DIR, model_name)

        loaded_models_cache[model_name] = tf.keras.models.load_model(model_path, custom_objects=custom_objects_dict)
        print(f"   - {model_name} loaded and cached successfully.")
    return loaded_models_cache[model_name]


# ==============================================================================
# 2. Define Prediction Function
# ==============================================================================

def predict_glaucoma(model_name, input_image):
    """
    Receives a model name and an image input, returns the prediction.
    """
    if input_image is None:
        return None, "Please upload an image first."

    model = get_model(model_name)
    threshold = MODEL_THRESHOLDS.get(model_name, 0.5)

    img = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img = img.resize((300, 300))
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.efficientnet.preprocess_input(img_batch)

    prediction_prob = model.predict(preprocessed_img, verbose=0)[0][0]

    prob_normal = float(prediction_prob)
    prob_glaucoma = 1.0 - prob_normal

    if prob_normal > threshold:
        final_diagnosis = "Normal"
    else:
        final_diagnosis = "Glaucoma"

    confidence_scores = {'Glaucoma': prob_glaucoma, 'Normal': prob_normal}
    diagnosis_text = f"Diagnosis: {final_diagnosis}\n(Model: {model_name}, Threshold: {threshold})"

    return confidence_scores, diagnosis_text


# ==============================================================================
# 3. Create and Launch Gradio Interface
# ==============================================================================
print("\n--- Building Gradio interface... ---")

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown(
        """
        # Automated Glaucoma Detection System
        Upload a fundus image, select an analysis model, and the system will provide a preliminary diagnostic suggestion.
        This result is for informational purposes only and cannot replace a professional medical diagnosis.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Fundus Image", type="numpy")
            model_selector = gr.Dropdown(
                choices=available_models,
                value=available_models[-1],
                label="Select Analysis Model"
            )
            submit_button = gr.Button("Run Diagnosis", variant="primary")

        with gr.Column(scale=1):
            diagnosis_output = gr.Textbox(label="Final Diagnosis", lines=3)
            confidence_output = gr.Label(num_top_classes=2, label="Prediction Confidence")

    submit_button.click(
        fn=predict_glaucoma,
        inputs=[model_selector, image_input],
        outputs=[confidence_output, diagnosis_output]
    )

print("\n--- Gradio app is ready. Launching now... ---")
interface.launch()


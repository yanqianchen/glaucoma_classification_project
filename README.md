# Automated Glaucoma Detection System

This project presents a deep learning-based framework for the automated detection of glaucoma from retinal fundus images. The core of this research is a comparative analysis of three models to evaluate the impact of a Systematic Realistic Augmentation (SRA) strategy and a Convolutional Block Attention Module (CBAM) on model robustness and generalization.

The final deliverable is a web-based user interface, built with Gradio, that allows for the real-time classification of uploaded fundus images using the trained models.

## Directory Structure

The project is organized with a clear and scalable structure to separate code, models, and the application logic.
```
glaucoma_classification_project/
│
├── 📂 code/
│   ├── cbam.py               # Custom CBAM module implementation
│   └── glaucoma_code.ipynb   # Jupyter Notebook for all research and training
│
├── 📂 models/
│   ├── A_Benchmark_EfficientNetB3_model.keras
│   ├── B_SRA_EfficientNetB3_model.keras
│   └── C_SRA_Attention_EfficientNetB3_model.keras
│
├── 🖼️ images/
│   └── (Contains sample images, UI screenshots, etc.)
│
├── 🚀 app.py                  # The main Gradio application script
├── 📄 README.md                # This file
└── 📋 requirements.txt        # Python dependencies for the project
```


## How to Run the Application

Clone the repository:
```
git clone [your-repository-url]
cd glaucoma_classification_project
```


Install dependencies:
It is recommended to use a virtual environment.
```
pip install -r requirements.txt
```


Run the Gradio app:
```
python app.py
```


Open your web browser and navigate to the local URL provided in the terminal (usually http://127.0.0.1:7860).

## Demonstration

The user interface allows for easy image upload and model selection to perform the diagnosis.

![system.png](images/readme/system.png)
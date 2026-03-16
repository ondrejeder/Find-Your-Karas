import os
# FORCE CPU-only mode: Hide any ghost GPUs from PyTorch/Ultralytics
# This must be set before any other imports that might initialize torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gradio as gr
from ultralytics import RTDETR
from PIL import Image
import numpy as np

# Global model variable
model = None

def get_model():
    global model
    if model is None:
        model_path = 'best.pt'
        if os.path.exists(model_path):
            try:
                # Load specifying cpu device
                print("--- Loading Model (CPU) ---")
                model = RTDETR(model_path)
                model.to('cpu')
            except Exception as e:
                print(f"Error loading model: {e}")
    return model

def predict_species(image, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run RT-DETR inference on an image to identify Karas species (CPU Optimized)
    """
    if image is None:
        return None, "Please upload an image."
    
    current_model = get_model()
    if current_model is None:
        return np.array(image), "⚠️ **Model file (best.pt) not found.**"
    
    # Run prediction explicitly on CPU
    try:
        results = current_model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            device='cpu',  # Hard-enforce CPU
            verbose=False
        )
    except Exception as e:
        return np.array(image), f"❌ **Error during inference:** {str(e)}"
    
    # Get the first result
    result = results[0]
    
    # Plot results on image
    plotted_img = result.plot()
    
    # Convert BGR to RGB (Ultralytics plot returns BGR numpy array)
    plotted_img_rgb = plotted_img[..., ::-1]
    
    # Extract detection information for the summary
    boxes = result.boxes
    species_counts = {}
    
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            species_counts[class_name] = species_counts.get(class_name, 0) + 1
    
    if not species_counts:
        return plotted_img_rgb, "ℹ️ No fish detected in the image."
    
    # Format output text
    summary = "## 🐟 Detection Results\n\n"
    for species, count in species_counts.items():
        summary += f"- **{species}**: {count} detected\n"
    
    return plotted_img_rgb, summary

# Create Gradio interface
with gr.Blocks(title="Find Your Karas (RT-DETR)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐟 Find Your Karas
        ### Powered by RT-DETR
        
        Upload an image of a fish to identify if it is a **Karas obecný** or a **Karas stříbřitý**.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="📤 Upload Image")
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Lower = more detections"
                )
                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.45,
                    step=0.05,
                    label="IoU Threshold",
                    info="Higher = fewer overlapping boxes"
                )
            
            predict_btn = gr.Button("🔍 Identify Species", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="🎯 Identification Results")
            output_text = gr.Markdown(label="📊 Summary")
    
    gr.Markdown(
        """
        ---
        ### 📋 How to Use:
        1. **Upload an image** of the fish you want to identify.
        2. Click **"Identify Species"**.
        3. The model will highlight any detected Karas fish and tell you the species.
        
        *Note: This app requires a trained `best.pt` model file to be present in the root folder.*
        ---
        """
    )
    
    predict_btn.click(
        fn=predict_species,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)

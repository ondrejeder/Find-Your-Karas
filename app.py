import os
# Hard-enforce CPU-only mode before ANY other imports (like torch/ultralytics)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gradio as gr
from ultralytics import RTDETR
from PIL import Image
import numpy as np

# Global model variable
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            print("--- Loading Model (CPU) ---")
            _model = RTDETR('best.pt')
            _model.to('cpu')
        except Exception as e:
            print(f"Error loading model: {e}")
    return _model

def predict_species(image, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run RT-DETR inference on an image to identify Karas species
    """
    if image is None:
        return None, "Please upload an image."
    
    # Lazy load only when identification is clicked
    model = get_model()
    if model is None:
        return np.array(image), "⚠️ **Model file (best.pt) not found.**"
    
    # Run prediction
    try:
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            device='cpu',  # Hard-enforced CPU for free tier stability
            verbose=False
        )
    except Exception as e:
        return np.array(image), f"❌ **Error during inference:** {str(e)}"

    
    # Get the first result
    result = results[0]
    
    # Plot results on image
    plotted_img = result.plot()
    
    # Convert BGR to RGB
    plotted_img_rgb = plotted_img[..., ::-1]
    
    # Extract detection information
    boxes = result.boxes
    species_counts = {}
    
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            species_counts[class_name] = species_counts.get(class_name, 0) + 1
    
    if not species_counts:
        return plotted_img_rgb, "ℹ️ No fish detected in the image."
    
    # Format summary
    summary = "## 🐟 Detection Results\n\n"
    for species, count in species_counts.items():
        summary += f"- **{species}**: {count} detected\n"
    
    return plotted_img_rgb, summary

# Build the same clean interface
with gr.Blocks(title="Find Your Karas (RT-DETR)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐟 Find Your Karas\nIdentify if a fish is a **Karas obecný** or a **Karas stříbřitý**.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="📤 Upload Image")
            conf_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
            iou_slider = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU")
            btn = gr.Button("🔍 Identify Species", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(type="numpy", label="🎯 Results")
            output_txt = gr.Markdown(label="📊 Summary")
    
    btn.click(predict_species, inputs=[input_img, conf_slider, iou_slider], outputs=[output_img, output_txt])

if __name__ == "__main__":
    demo.launch()

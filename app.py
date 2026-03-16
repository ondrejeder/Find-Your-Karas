import gradio as gr
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import os

# FORCE CPU-only mode: Hide any ghost GPUs from PyTorch/Ultralytics
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load your trained RT-DETR model
# We load it directly like the working Fish Tracking app
model = RTDETR('best.pt')
model.to('cpu')

def predict_species(image, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run RT-DETR inference on an image to identify Karas species
    """
    if image is None:
        return None, "Please upload an image."
    
    # Run prediction explicitly on CPU
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        device='cpu',
        verbose=False
    )
    
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
        return plotted_img_rgb, "ℹ️ **No fish detected in the image.**"
    
    # Format a nice species summary
    summary = "## 🐟 Identification Results\n\n"
    
    # Check for specific Karas species based on your model classes
    # Assuming classes are 'karas obecny' and 'karas stribrity'
    for species, count in species_counts.items():
        icon = "🐠" if "stříbřitý" in species.lower() else "🐟"
        summary += f"### {icon} {species}: **{count}** detected\n"
    
    summary += "\n---\n*Identification powered by RT-DETR*"
    
    return plotted_img_rgb, summary

# Create Gradio interface mirroring the Fish Tracking style but simplified
# Using theme=gr.themes.Soft() as in the working app
with gr.Blocks(title="Find Your Karas (RT-DETR)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐟 Find Your Karas
        ### Species Identification powered by RT-DETR
        
        Upload an image of fish to automatically identify and count **Karas obecný** and **Karas stříbřitý**.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="📤 Upload Image")
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Lower = more detections (but more false positives)"
                )
                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.45,
                    step=0.05,
                    label="IoU Threshold (NMS)",
                    info="Cleanup overlapping boxes"
                )
            
            predict_btn = gr.Button("🔍 Identify Species", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="🎯 Detection Results")
            output_text = gr.Markdown(label="📊 Analysis Output")
    
    gr.Markdown(
        """
        ---
        ### 📋 How to Use:
        1. **Upload an image** containing Karas fish.
        2. Click **"Identify Species"**.
        3. View the highlighted fish and the count for each species below the image.
        
        ---
        """
    )
    
    predict_btn.click(
        fn=predict_species,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch()

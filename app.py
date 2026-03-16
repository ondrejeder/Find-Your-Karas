import gradio as gr
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import os

# FORCE CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load model
model = RTDETR('best.pt')
model.to('cpu')

# Localization strings
I18N = {
    "cz": {
        "title": "🐟 Najdi svého Karase",
        "subtitle": "Identifikace druhů pomocí RT-DETR",
        "description": "Nahrajte fotografii ryby a zjistěte, zda se jedná o **Karase obecného** nebo **Karase stříbřitého**.",
        "input_label": "📤 Nahrát obrázek",
        "conf_label": "Práh spolehlivosti",
        "conf_info": "Vyšší = přesnější (ale méně detekcí)",
        "iou_label": "Práh překryvu (IoU)",
        "btn_label": "🔍 Identifikovat druh",
        "output_label": "🎯 Výsledky detekce",
        "summary_label": "📊 Výsledek",
        "no_fish": "ℹ️ **V obrázku nebyl detekován žádný karas.**",
        "detected_prefix": "## 🐟 Výsledek identifikace\n\n",
        "detected_suffix": " nalezen!",
        "footer": "### 📋 Jak používat:\n1. **Nahrajte obrázek** s karasem.\n2. Klikněte na **\"Identifikovat druh\"**.\n3. Aplikace označí rybu v obrázku a potvrdí druh pod ním.",
        "lang_toggle": "Language / Jazyk"
    },
    "en": {
        "title": "🐟 Find Your Karas",
        "subtitle": "Species Identification powered by RT-DETR",
        "description": "Upload a fish photo to identify if it is a **Karas obecný** (Crucian Carp) or **Karas stříbřitý** (Prussian Carp).",
        "input_label": "📤 Upload Image",
        "conf_label": "Confidence Threshold",
        "conf_info": "Higher = more precise (but fewer detections)",
        "iou_label": "IoU Threshold",
        "btn_label": "🔍 Identify Species",
        "output_label": "🎯 Detection Results",
        "summary_label": "📊 Analysis Output",
        "no_fish": "ℹ️ **No Karas detected in the image.**",
        "detected_prefix": "## 🐟 Identification Results\n\n",
        "detected_suffix": " detected!",
        "footer": "### 📋 How to Use:\n1. **Upload an image** containing fish.\n2. Click **\"Identify Species\"**.\n3. View the highlighted fish and species confirmation below.",
        "lang_toggle": "Language / Jazyk"
    }
}

def predict_species(image, lang, conf_threshold=0.7, iou_threshold=0.45):
    """
    Run RT-DETR inference and return localized results
    """
    if image is None:
        return None, ""
    
    # Run prediction
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        device='cpu',
        verbose=False
    )
    
    result = results[0]
    plotted_img_rgb = result.plot()[..., ::-1]
    
    # Extract unique species detected
    detected_species = set()
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detected_species.add(result.names[class_id])
    
    lang_key = "cz" if lang == "Czech / Čeština" else "en"
    
    if not detected_species:
        return plotted_img_rgb, I18N[lang_key]["no_fish"]
    
    # Format summary without counts
    summary = I18N[lang_key]["detected_prefix"]
    for species in sorted(list(detected_species)):
        icon = "🐠" if "stříbř" in species.lower() or "prussian" in species.lower() else "🐟"
        summary += f"### {icon} {species}{I18N[lang_key]['detected_suffix']}\n"
    
    return plotted_img_rgb, summary

def update_ui(lang):
    """Update UI labels based on language selection"""
    lang_key = "cz" if lang == "Czech / Čeština" else "en"
    texts = I18N[lang_key]
    return (
        gr.update(value=f"# {texts['title']}\n### {texts['subtitle']}\n{texts['description']}"),
        gr.update(label=texts["input_label"]),
        gr.update(label=texts["conf_label"], info=texts["conf_info"]),
        gr.update(label=texts["iou_label"]),
        gr.update(value=texts["btn_label"]),
        gr.update(label=texts["output_label"]),
        gr.update(label=texts["summary_label"]),
        gr.update(value=f"---\n{texts['footer']}\n---")
    )

with gr.Blocks(title="Find Your Karas", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        lang_selector = gr.Radio(
            choices=["Czech / Čeština", "English"],
            value="Czech / Čeština",
            label="Language / Jazyk",
            interactive=True
        )
    
    header = gr.Markdown(f"# {I18N['cz']['title']}\n### {I18N['cz']['subtitle']}\n{I18N['cz']['description']}")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label=I18N["cz"]["input_label"])
            
            with gr.Accordion("⚙️ Advanced / Pokročilé", open=False):
                conf_slider = gr.Slider(0.1, 0.95, value=0.7, step=0.05, label=I18N["cz"]["conf_label"])
                iou_slider = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label=I18N["cz"]["iou_label"])
            
            predict_btn = gr.Button(I18N["cz"]["btn_label"], variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label=I18N["cz"]["output_label"])
            output_text = gr.Markdown(label=I18N["cz"]["summary_label"])
    
    footer = gr.Markdown(f"---\n{I18N['cz']['footer']}\n---")
    
    # Event listeners
    lang_selector.change(
        update_ui, 
        inputs=[lang_selector], 
        outputs=[header, input_image, conf_slider, iou_slider, predict_btn, output_image, output_text, footer]
    )
    
    predict_btn.click(
        fn=predict_species,
        inputs=[input_image, lang_selector, conf_slider, iou_slider],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch()

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
    "🇨🇿": {
        "title": "Najdi svého Karase",
        "subtitle": "Identifikace druhů pomocí RT-DETR",
        "description": "Nahrajte fotografii ryby a zjistěte, zda se jedná o **Karase obecného** nebo **Karase stříbřitého**.",
        "input_label": "Nahrát obrázek",
        "conf_label": "Práh spolehlivosti",
        "conf_info": "Vyšší = přesnější (ale méně detekcí)",
        "iou_label": "Práh překryvu (IoU)",
        "btn_label": "Identifikovat druh",
        "output_label": "Výsledky detekce",
        "summary_label": "Výsledek",
        "no_fish": "ℹ️ **V obrázku nebyl detekován žádný karas.**",
        "detected_prefix": "# Výsledek identifikace\n\n",
        "detected_suffix": " nalezen!",
        "footer": "### 📋 Jak používat:\n1. **Nahrajte obrázek** s karasem.\n2. Klikněte na **\"Identifikovat druh\"**.\n3. Aplikace označí rybu v obrázku a potvrdí druh pod ním.",
        "advanced_label": "Pokročilé",
        "species_map": {
            "karas obecny": "Karas obecný",
            "karas stribrity": "Karas stříbřitý"
        }
    },
    "🇬🇧": {
        "title": "Find Your Karas",
        "subtitle": "Species Identification powered by RT-DETR",
        "description": "Upload a fish photo to identify if it is a **Karas obecný** (Crucian Carp) or **Karas stříbřitý** (Prussian Carp).",
        "input_label": "Upload Image",
        "conf_label": "Confidence Threshold",
        "conf_info": "Higher = more precise (but fewer detections)",
        "iou_label": "IoU Threshold",
        "btn_label": "Identify Species",
        "output_label": "Detection Results",
        "summary_label": "Analysis Output",
        "no_fish": "ℹ️ **No Karas detected in the image.**",
        "detected_prefix": "# Identification Results\n\n",
        "detected_suffix": " detected!",
        "footer": "### 📋 How to Use:\n1. **Upload an image** containing fish.\n2. Click **\"Identify Species\"**.\n3. View the highlighted fish and species confirmation below.",
        "advanced_label": "Advanced",
        "species_map": {
            "karas obecny": "Crucian Carp",
            "karas stribrity": "Prussian Carp"
        }
    }
}

def format_summary(detected_species, lang):
    """Format the detection summary based on language and detected species"""
    if not detected_species:
        return I18N[lang]["no_fish"]
    
    summary = I18N[lang]["detected_prefix"]
    # Sort for deterministic output
    for species_id in sorted(list(detected_species)):
        icon = "🐠" if "stribrity" in species_id.lower() or "prussian" in species_id.lower() else "🐟"
        friendly_name = I18N[lang]["species_map"].get(species_id, species_id)
        # Using # for extra large text as requested
        summary += f"# {icon} {friendly_name}{I18N[lang]['detected_suffix']}\n"
    
    summary += "\n---\n*Identification powered by RT-DETR*"
    return summary

def predict_species(image, lang, conf_threshold=0.7, iou_threshold=0.45):
    """
    Run RT-DETR inference and return localized results + state
    """
    if image is None:
        return None, "", set()
    
    # Run prediction explicitly on CPU
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
    
    # Extract unique species detected (internal IDs)
    detected_species = set()
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            species_internal = result.names[class_id]
            detected_species.add(species_internal)
    
    summary = format_summary(detected_species, lang)
    
    return plotted_img_rgb, summary, detected_species

def translate_ui(lang, detected_species_state):
    """Update UI labels and translate current results if they exist"""
    texts = I18N[lang]
    
    # Update all fixed UI elements
    # 1. header, 2. input_image, 3. conf_slider, 4. iou_slider, 5. predict_btn, 6. output_image, 7. output_text (summary_label), 8. footer, 9. advanced_accordion (label)
    updates = [
        gr.update(value=f"# 🐟 {texts['title']}\n### {texts['subtitle']}\n{texts['description']}"),
        gr.update(label=texts["input_label"]),
        gr.update(label=texts["conf_label"], info=texts["conf_info"]),
        gr.update(label=texts["iou_label"]),
        gr.update(value=texts["btn_label"]),
        gr.update(label=texts["output_label"]),
        gr.update(label=texts["summary_label"]),
        gr.update(value=f"---\n{texts['footer']}\n---"),
        gr.update(label=texts["advanced_label"])
    ]
    
    # 10. If we have a past detection, translate it instantly in the Markdown value
    if detected_species_state:
        updates.append(gr.update(value=format_summary(detected_species_state, lang)))
    else:
        updates.append(gr.update()) # Keep current (empty) value
    
    return updates

with gr.Blocks(title="Find Your Karas", theme=gr.themes.Soft()) as demo:
    # State to keep track of detections for live translation
    last_detected_species = gr.State(set())
    
    with gr.Row():
        lang_selector = gr.Radio(
            choices=["🇨🇿", "🇬🇧"],
            value="🇨🇿",
            label="Jazyk / Language",
            interactive=True
        )
    
    header = gr.Markdown(f"# 🐟 {I18N['🇨🇿']['title']}\n### {I18N['🇨🇿']['subtitle']}\n{I18N['🇨🇿']['description']}")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label=I18N["🇨🇿"]["input_label"])
            
            with gr.Accordion(I18N["🇨🇿"]["advanced_label"], open=False) as advanced_accordion:
                conf_slider = gr.Slider(0.1, 0.95, value=0.7, step=0.05, label=I18N["🇨🇿"]["conf_label"])
                iou_slider = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label=I18N["🇨🇿"]["iou_label"])
            
            predict_btn = gr.Button(I18N["🇨🇿"]["btn_label"], variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label=I18N["🇨🇿"]["output_label"])
            output_text = gr.Markdown(label=I18N["🇨🇿"]["summary_label"])
    
    footer = gr.Markdown(f"---\n{I18N['🇨🇿']['footer']}\n---")
    
    # Event listeners
    lang_selector.change(
        translate_ui, 
        inputs=[lang_selector, last_detected_species], 
        outputs=[
            header, input_image, conf_slider, iou_slider, predict_btn, 
            output_image, output_text, footer, advanced_accordion, output_text
        ]
    )
    
    predict_btn.click(
        fn=predict_species,
        inputs=[input_image, lang_selector, conf_slider, iou_slider],
        outputs=[output_image, output_text, last_detected_species]
    )

if __name__ == "__main__":
    demo.launch()

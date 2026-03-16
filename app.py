import os

# 1. SET ENVIRONMENT VARIABLES FIRST
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gradio as gr
from ultralytics import RTDETR

# 2. LOAD MODEL
model = RTDETR('best.pt')
model.to('cpu')

# 3. LOCALIZATION DICTIONARY
I18N = {
    "CZ": {
        "title": "Najdi svého Karase",
        "subtitle": "Identifikace druhů pomocí RT-DETR",
        "description": "Nahrajte fotografii ryby a zjistěte, zda se jedná o **Karase obecného** nebo **Karase stříbřitého**.",
        "input_label": "Nahrát obrázek",
        "conf_label": "Práh spolehlivosti",
        "conf_info": "Vyšší = přesnější",
        "iou_label": "Práh překryvu (IoU)",
        "btn_label": "Identifikovat druh",
        "output_label": "Výsledky detekce",
        "summary_label": "Výsledek identifikace",
        "no_fish": "## ℹ️ V obrázku nebyl detekován žádný karas.",
        "detected_prefix": "## 🐟 Výsledek:\n\n",
        "detected_suffix": " nalezen!",
        "footer": "### 📋 Jak používat:\n1. **Nahrajte obrázek** s karasem.\n2. Klikněte na **\"Identifikovat druh\"**.\n3. Aplikace označí rybu v obrázku a potvrdí druh pod ním.",
        "advanced_label": "Pokročilé nastavení",
        "lang_label": "Jazyk",
        "species_map": {
            "karas obecny": "Karas obecný",
            "karas stribrity": "Karas stříbřitý"
        }
    },
    "EN": {
        "title": "Find Your Karas",
        "subtitle": "Species Identification powered by RT-DETR",
        "description": "Upload a fish photo to identify if it is a **Crucian Carp** or a **Prussian Carp**.",
        "input_label": "Upload Image",
        "conf_label": "Confidence Threshold",
        "conf_info": "Higher = more precise",
        "iou_label": "IoU Threshold",
        "btn_label": "Identify Species",
        "output_label": "Detection Results",
        "summary_label": "Identification Summary",
        "no_fish": "## ℹ️ No Karas detected in the image.",
        "detected_prefix": "## 🐟 Results:\n\n",
        "detected_suffix": " detected!",
        "footer": "### 📋 How to Use:\n1. **Upload an image** containing fish.\n2. Click **\"Identify Species\"**.\n3. View the highlighted fish and species confirmation below.",
        "advanced_label": "Advanced Settings",
        "lang_label": "Language",
        "species_map": {
            "karas obecny": "Crucian Carp",
            "karas stribrity": "Prussian Carp"
        }
    }
}

def format_summary(detected_species, lang):
    """Generates localized summary text for detected species using large headers."""
    # Ensure lang is valid, default to CZ
    l_key = "CZ" if lang == "CZ" else "EN"
    
    if detected_species is None:
        return ""
    
    # Handle the "No Fish" case
    if not detected_species:
        return str(I18N[l_key]["no_fish"])
    
    # Start building the summary string explicitly
    final_text = str(I18N[l_key]["detected_prefix"])
    suffix = str(I18N[l_key]["detected_suffix"])
    species_map = I18N[l_key]["species_map"]
    
    # Add each species to the text
    for species_id in sorted(list(detected_species)):
        icon = "🐠" if "stribrity" in str(species_id).lower() or "prussian" in str(species_id).lower() else "🐟"
        friendly_name = str(species_map.get(species_id, species_id))
        # Use # for extra large header
        line = f"# {icon} {friendly_name}{suffix}\n"
        final_text = final_text + line
    
    final_text = final_text + "\n---\n*RT-DETR Identification*"
    return final_text

def predict_species(image, lang, conf_threshold=0.7, iou_threshold=0.45):
    """Run model and return results along with detected classes for state tracking."""
    if image is None:
        return None, "", None
    
    # Ensure lang is valid for internal logic
    l_key = "CZ" if lang == "CZ" else "EN"
    
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
    
    # Extract unique species detected
    detected_species = set()
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            species_id = result.names[class_id]
            detected_species.add(species_id)
    
    summary = format_summary(detected_species, l_key)
    return plotted_img_rgb, summary, detected_species

def translate_ui(lang, last_species):
    """Swap app language and re-translate existing results."""
    l_key = "CZ" if lang == "CZ" else "EN"
    texts = I18N[l_key]
    
    # Update UI components
    updates = [
        gr.update(value=f"# 🐟 {texts['title']}\n### {texts['subtitle']}\n{texts['description']}"), # header
        gr.update(label=texts["input_label"]), # input_image
        gr.update(label=texts["lang_label"]), # lang_selector
        gr.update(label=texts["advanced_label"]), # advanced_settings
        gr.update(label=texts["conf_label"], info=texts["conf_info"]), # conf_slider
        gr.update(label=texts["iou_label"]), # iou_slider
        gr.update(value=texts["btn_label"]), # predict_btn
        gr.update(label=texts["output_label"]), # output_image
        gr.update(value=format_summary(last_species, l_key)), # output_text
        gr.update(value=f"---\n{texts['footer']}\n---") # footer
    ]
    return updates

# 4. DEFINE INTERFACE
with gr.Blocks(title="Find Your Karas", theme=gr.themes.Soft()) as demo:
    # State tracking for instant translation
    last_detections = gr.State(None)
    
    with gr.Row():
        lang_selector = gr.Radio(
            choices=["CZ", "EN"],
            value="CZ",
            label=I18N["CZ"]["lang_label"],
            interactive=True
        )
    
    header = gr.Markdown(f"# 🐟 {I18N['CZ']['title']}\n### {I18N['CZ']['subtitle']}\n{I18N['CZ']['description']}")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label=I18N["CZ"]["input_label"])
            
            with gr.Accordion(I18N["CZ"]["advanced_label"], open=False) as advanced_settings:
                conf_slider = gr.Slider(0.1, 0.95, value=0.7, step=0.05, label=I18N["CZ"]["conf_label"])
                iou_slider = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label=I18N["CZ"]["iou_label"])
            
            predict_btn = gr.Button(I18N["CZ"]["btn_label"], variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label=I18N["CZ"]["output_label"])
            output_text = gr.Markdown(value="", show_label=False)
    
    footer = gr.Markdown(f"---\n{I18N['CZ']['footer']}\n---")
    
    # Event: Language Switch
    lang_selector.change(
        fn=translate_ui,
        inputs=[lang_selector, last_detections],
        outputs=[
            header, input_image, lang_selector, advanced_settings,
            conf_slider, iou_slider, predict_btn, output_image, output_text, footer
        ]
    )
    
    # Event: Inference
    predict_btn.click(
        fn=predict_species,
        inputs=[input_image, lang_selector, conf_slider, iou_slider],
        outputs=[output_image, output_text, last_detections]
    )

if __name__ == "__main__":
    demo.launch()

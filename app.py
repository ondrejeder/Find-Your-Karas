import os

# 1. SET ENVIRONMENT VARIABLES FIRST
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gradio as gr
from ultralytics import RTDETR

# 2. LOAD MODEL
model = RTDETR('best.pt')
model.to('cpu')

# 3. LOCALIZATION - Separated for clear type inference to resolve linter errors
LABELS = {
    "CZ": {
        "title": "Najdi svého Karase",
        "subtitle": "Identifikace druhů pomocí RT-DETR",
        "description": "Nahrajte fotografii ryby a zjistěte, zda se jedná o **Karase obecného** nebo **Karase stříbřitého**.",
        "input": "Nahrát obrázek",
        "conf": "Práh spolehlivosti",
        "conf_info": "Vyšší = přesnější",
        "iou": "Práh překryvu (IoU)",
        "btn": "Identifikovat druh",
        "output": "Výsledky detekce",
        "summary": "Výsledek identifikace",
        "no_fish": "## ℹ️ V obrázku nebyl detekován žádný karas.",
        "prefix": "## 🐟 Výsledek:\n\n",
        "suffix": " nalezen!",
        "footer": "### 📋 Jak používat:\n1. **Nahrajte obrázek** s karasem.\n2. Klikněte na **\"Identifikovat druh\"**.\n3. Aplikace označí rybu v obrázku a potvrdí druh pod ním.",
        "advanced": "Pokročilé nastavení",
        "lang": "Jazyk"
    },
    "EN": {
        "title": "Find Your Karas",
        "subtitle": "Species Identification powered by RT-DETR",
        "description": "Upload a fish photo to identify if it is a **Crucian Carp** or a **Prussian Carp**.",
        "input": "Upload Image",
        "conf": "Confidence Threshold",
        "conf_info": "Higher = more precise",
        "iou": "IoU Threshold",
        "btn": "Identify Species",
        "output": "Detection Results",
        "summary": "Identification Summary",
        "no_fish": "## ℹ️ No Karas detected in the image.",
        "prefix": "## 🐟 Results:\n\n",
        "suffix": " detected!",
        "footer": "### 📋 How to Use:\n1. **Upload an image** containing fish.\n2. Click **\"Identify Species\"**.\n3. View the highlighted fish and species confirmation below.",
        "advanced": "Advanced Settings",
        "lang": "Language"
    }
}

# Species name mapping separated from UI labels
SPECIES_NAMES = {
    "CZ": {
        "karas obecny": "Karas obecný",
        "karas stribrity": "Karas stříbřitý"
    },
    "EN": {
        "karas obecny": "Crucian Carp",
        "karas stribrity": "Prussian Carp"
    }
}

def format_summary(detected_species, lang):
    """Generates localized summary text for detected species using large headers."""
    # Defensive key check
    l_key = str(lang) if str(lang) in ["CZ", "EN"] else "CZ"
    
    if detected_species is None:
        return ""
    
    # Handle the "No Fish" case
    if not detected_species:
        return str(LABELS[l_key]["no_fish"])
    
    # Building summary string
    final_text = str(LABELS[l_key]["prefix"])
    suffix = str(LABELS[l_key]["suffix"])
    mapping = SPECIES_NAMES[l_key]
    
    # Process each detection
    for species_id in sorted(list(detected_species)):
        icon = "🐠" if "stribrity" in str(species_id).lower() or "prussian" in str(species_id).lower() else "🐟"
        friendly_name = str(mapping.get(str(species_id), str(species_id)))
        # Extra large header (#) as per user request
        line = f"# {icon} {friendly_name}{suffix}\n"
        final_text = final_text + line
    
    final_text = final_text + "\n---\n*RT-DETR Identification*"
    return final_text

def predict_species(image, lang, conf_threshold=0.7, iou_threshold=0.45):
    """Run model and return results along with detected classes for state tracking."""
    if image is None:
        return None, "", None
    
    # Normalize lang key
    l_key = "CZ" if str(lang) == "CZ" else "EN"
    
    # Inference
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
    
    # Track detected classes
    detected_species = set()
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]
            detected_species.add(name)
    
    summary = format_summary(detected_species, l_key)
    return plotted_img_rgb, summary, detected_species

def translate_ui(lang, last_species):
    """Live translate UI and results when language toggle is clicked."""
    l_key = "CZ" if str(lang) == "CZ" else "EN"
    texts = LABELS[l_key]
    
    # UI updates mapping to the 10 outputs defined below
    updates = [
        gr.update(value=f"# 🐟 {texts['title']}\n### {texts['subtitle']}\n{texts['description']}"), # header
        gr.update(label=texts["input"]), # input_image
        gr.update(label=texts["lang"]), # lang_selector
        gr.update(label=texts["advanced"]), # advanced_settings
        gr.update(label=texts["conf"], info=texts["conf_info"]), # conf_slider
        gr.update(label=texts["iou"]), # iou_slider
        gr.update(value=texts["btn"]), # predict_btn
        gr.update(label=texts["output"]), # output_image
        gr.update(value=format_summary(last_species, l_key)), # output_text
        gr.update(value=f"---\n{texts['footer']}\n---") # footer
    ]
    return updates

# 4. DEFINE INTERFACE
with gr.Blocks(title="Find Your Karas", theme=gr.themes.Soft()) as demo:
    # State for instant translation
    last_detections = gr.State(None)
    
    with gr.Row():
        lang_selector = gr.Radio(
            choices=["CZ", "EN"],
            value="CZ",
            label=LABELS["CZ"]["lang"],
            interactive=True
        )
    
    header = gr.Markdown(f"# 🐟 {LABELS['CZ']['title']}\n### {LABELS['CZ']['subtitle']}\n{LABELS['CZ']['description']}")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label=LABELS["CZ"]["input"])
            
            with gr.Accordion(LABELS["CZ"]["advanced"], open=False) as advanced_settings:
                conf_slider = gr.Slider(0.1, 0.95, value=0.7, step=0.05, label=LABELS["CZ"]["conf"])
                iou_slider = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label=LABELS["CZ"]["iou"])
            
            predict_btn = gr.Button(LABELS["CZ"]["btn"], variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label=LABELS["CZ"]["output"])
            output_text = gr.Markdown(value="", show_label=False)
    
    footer = gr.Markdown(f"---\n{LABELS['CZ']['footer']}\n---")
    
    # Interactions
    lang_selector.change(
        fn=translate_ui,
        inputs=[lang_selector, last_detections],
        outputs=[
            header, input_image, lang_selector, advanced_settings,
            conf_slider, iou_slider, predict_btn, output_image, output_text, footer
        ]
    )
    
    predict_btn.click(
        fn=predict_species,
        inputs=[input_image, lang_selector, conf_slider, iou_slider],
        outputs=[output_image, output_text, last_detections]
    )

if __name__ == "__main__":
    demo.launch()

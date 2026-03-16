import gradio as gr
from ultralytics import RTDETR
import os

# FORCE CPU-only mode: Hide GPUs from PyTorch before any model init
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load your trained RT-DETR model
# The model is loaded directly for stability on HF CPU tier
model = RTDETR('best.pt')
model.to('cpu')

# Dictionary containing localized text for the interface
I18N = {
    "🇨🇿": {
        "title": "Najdi svého Karase",
        "subtitle": "Identifikace karasů pomocí RT-DETR",
        "description": "Nahrajte fotografii a zjistěte, zda jde o **Karase obecného** nebo **Karase stříbřitého**.",
        "input_label": "Nahrát obrázek",
        "conf_label": "Práh spolehlivosti",
        "conf_info": "Vyšší práh = vyšší jistota detekce",
        "iou_label": "Práh IoU (NMS)",
        "btn_label": "🔍 Identifikovat druh",
        "output_label": "Detekce v obrázku",
        "summary_label": "Výsledek identifikace",
        "no_fish": "## ℹ️ V obrázku nebyl detekován žádný karas.",
        "detected_prefix": "## 🐟 Výsledek:\n\n",
        "detected_suffix": " nalezen!",
        "footer": "### 📋 Jak používat:\n1. **Nahrajte obrázek** s karasem.\n2. Klikněte na **\"Identifikovat druh\"**.\n3. Aplikace označí rybu a potvrdí druh.",
        "advanced_label": "Pokročilé nastavení",
        "lang_label": "Jazyk",
        "species_map": {
            "karas obecny": "Karas obecný",
            "karas stribrity": "Karas stříbřitý"
        }
    },
    "🇬🇧": {
        "title": "Find Your Karas",
        "subtitle": "Karas Identification powered by RT-DETR",
        "description": "Upload a photo to see if it's a **Crucian Carp** or a **Prussian Carp**.",
        "input_label": "Upload Image",
        "conf_label": "Confidence Threshold",
        "conf_info": "Higher threshold = more confident detections",
        "iou_label": "IoU Threshold (NMS)",
        "btn_label": "🔍 Identify Species",
        "output_label": "Detected Image",
        "summary_label": "Identification Result",
        "no_fish": "## ℹ️ No Karas detected in the image.",
        "detected_prefix": "## 🐟 Results:\n\n",
        "detected_suffix": " detected!",
        "footer": "### 📋 How to Use:\n1. **Upload an image** of a Karas.\n2. Click **\"Identify Species\"**.\n3. The app will label the fish and confirm the species.",
        "advanced_label": "Advanced Settings",
        "lang_label": "Language",
        "species_map": {
            "karas obecny": "Crucian Carp",
            "karas stribrity": "Prussian Carp"
        }
    }
}

def format_summary(detected_species, lang):
    """Generates localized summary text for detected species using large font."""
    if detected_species is None:
        return ""
    if not detected_species:
        return I18N[lang]["no_fish"]
    
    summary = I18N[lang]["detected_prefix"]
    # Species list is sorted for consistent display
    for species_id in sorted(list(detected_species)):
        icon = "🐠" if "stribrity" in species_id.lower() or "prussian" in species_id.lower() else "🐟"
        friendly_name = I18N[lang]["species_map"].get(species_id, species_id)
        # Using # for extra large header as requested by the user
        summary += f"# {icon} {friendly_name}{I18N[lang]['detected_suffix']}\n"
    
    summary += "\n---\n*RT-DETR Inference*"
    return summary

def predict_species(image, lang, conf_threshold=0.7, iou_threshold=0.45):
    """Core prediction logic returning image, summary text, and detected classes state."""
    if image is None:
        return None, "", None
    
    # Run inference explicitly on CPU
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        device='cpu',
        verbose=False
    )
    
    result = results[0]
    # Convert BGR (OpenCV) to RGB (Gradio)
    plotted_img = result.plot()[..., ::-1]
    
    # Identify unique detected classes
    detected_species = set()
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]
            detected_species.add(name)
    
    summary = format_summary(detected_species, lang)
    return plotted_img, summary, detected_species

def translate_ui(lang, last_species):
    """Updates all UI labels and re-renders current results in the new language."""
    texts = I18N[lang]
    
    # Returns 10 update objects mapping to the 10 outputs in lang_selector.change
    updates = [
        gr.update(value=f"# 🐟 {texts['title']}\n### {texts['subtitle']}\n{texts['description']}"), # header
        gr.update(label=texts["input_label"]), # input_image
        gr.update(label=texts["lang_label"]), # lang_selector
        gr.update(label=texts["advanced_label"]), # advanced_settings
        gr.update(label=texts["conf_label"], info=texts["conf_info"]), # conf_slider
        gr.update(label=texts["iou_label"]), # iou_slider
        gr.update(value=texts["btn_label"]), # predict_btn
        gr.update(label=texts["output_label"]), # output_image
        gr.update(label=texts["summary_label"], value=format_summary(last_species, lang)), # output_text
        gr.update(value=f"---\n{texts['footer']}\n---") # footer
    ]
    return updates

# Define the user interface
with gr.Blocks(title="Find Your Karas", theme=gr.themes.Soft()) as demo:
    # state to keep track of detections for instant translation
    detected_species_state = gr.State(None)
    
    with gr.Row():
        lang_selector = gr.Radio(
            choices=["🇨🇿", "🇬🇧"],
            value="🇨🇿",
            label=I18N["🇨🇿"]["lang_label"],
            interactive=True
        )
    
    header = gr.Markdown(f"# 🐟 {I18N['🇨🇿']['title']}\n### {I18N['🇨🇿']['subtitle']}\n{I18N['🇨🇿']['description']}")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label=I18N["🇨🇿"]["input_label"])
            
            with gr.Accordion(I18N["🇨🇿"]["advanced_label"], open=False) as advanced_settings:
                conf_slider = gr.Slider(0.1, 0.95, value=0.7, step=0.05, label=I18N["🇨🇿"]["conf_label"])
                iou_slider = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label=I18N["🇨🇿"]["iou_label"])
            
            predict_btn = gr.Button(I18N["🇨🇿"]["btn_label"], variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label=I18N["🇨🇿"]["output_label"])
            output_text = gr.Markdown(label=I18N["🇨🇿"]["summary_label"])
    
    footer = gr.Markdown(f"---\n{I18N['🇨🇿']['footer']}\n---")
    
    # Bind translation event
    lang_selector.change(
        fn=translate_ui,
        inputs=[lang_selector, detected_species_state],
        outputs=[
            header, input_image, lang_selector, advanced_settings,
            conf_slider, iou_slider, predict_btn, output_image, output_text, footer
        ]
    )
    
    # Bind prediction event
    predict_btn.click(
        fn=predict_species,
        inputs=[input_image, lang_selector, conf_slider, iou_slider],
        outputs=[output_image, output_text, detected_species_state]
    )

if __name__ == "__main__":
    demo.launch()

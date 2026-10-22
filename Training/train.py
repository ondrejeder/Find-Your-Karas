from ultralytics import RTDETR, YOLO
import os
import argparse

def train_model():
    parser = argparse.ArgumentParser(description="Train a YOLO/RT-DETR model for fish detection.")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", 
                        help="Model file to use (e.g., rtdetr-l.pt, rtdetr-x.pt, yolov9e.pt, yolov10x.pt)")
    
    # Dataset arguments
    parser.add_argument("--data", type=str, default="data.yaml", 
                        help="Path to data.yaml file.")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default 640).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (optimize for your VRAM).")
    parser.add_argument("--device", type=str, default="0", help="Device to use (e.g., 0 or cpu).")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--amp", type=str, default="True", help="Use Automatic Mixed Precision ('True' or 'False').")
    
    # Output parameters
    parser.add_argument("--project", type=str, default="runs/detect", help="Project name (folder).")
    parser.add_argument("--name", type=str, default="train_karas", help="Run name.")
    
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"⚠️ Warning: {args.data} not found.")
        print("Please ensure your dataset is correctly pathing to the YAML file.")
        return

    # Load the requested model
    print(f"🚀 Loading model: {args.model}")
    if "rtdetr" in args.model.lower():
        model = RTDETR(args.model)
    else:
        # Fallback to standard YOLO loader (handles v8, v9, v10)
        model = YOLO(args.model)

    # Train the model
    print(f"📈 Starting training: {args.name} in {args.project}...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        amp=(args.amp.lower() == "true"),
        patience=50,       # Increased: richer augmentation needs more epochs to converge
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,     # If name exists, increment or use existing
        # On-the-fly augmentations (applied per-batch, not saved to disk)
        degrees=15,        # Random rotation ±15° — fish are photographed at all angles
        scale=0.5,         # Random scale 50-150% — handles fish at different distances
        perspective=0.0005 # Slight perspective warp — handles different shooting angles
    )

    print("✅ Training complete!")
    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"Your best model is saved at: {best_pt}")

    # Export best.pt → best.onnx for faster CPU inference
    print("📦 Exporting best.pt to ONNX format...")
    try:
        if "rtdetr" in args.model.lower():
            export_model = RTDETR(best_pt)
        else:
            export_model = YOLO(best_pt)
        export_path = export_model.export(format="onnx", imgsz=args.imgsz, dynamic=False)
        print(f"✅ ONNX model saved at: {export_path}")
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
        print("You can manually export later with: model.export(format='onnx')")

if __name__ == "__main__":
    train_model()

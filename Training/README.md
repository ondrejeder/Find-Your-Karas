# Trénování detekce karasů

Archiv původního `train.py` a konfigurací Roboflow datasetů. Fotografie, virtuální prostředí, váhy a adresář `runs` nejsou součástí této zálohy.

1. Vytvořte Python venv a nainstalujte PyTorch pro svůj hardware a knihovny Ultralytics/ONNX. Původní pozorované verze jsou v `requirements-observed.txt`; PyTorch tehdy používal CUDA 12.8 nightly build.
2. Stáhněte dataset podle odkazu `roboflow.url` ve vybraném `data.yaml`. Nastavte cesty `train`, `val` a `test` podle skutečného umístění fotografií. Původní názvy složek i YAML zůstaly zachované; číslo v názvu složky nemusí odpovídat verzi Roboflow.
3. Spusťte například z tohoto adresáře:

```sh
python train.py --model rtdetr-l.pt --data data.yaml --epochs 100 --imgsz 640 --batch 8 --device 0
```

Pro jiný Ultralytics model změňte `--model`; skript rozlišuje RT-DETR a YOLO podle názvu. Model `.pt` musí být dostupný lokálně nebo ke stažení přes Ultralytics. Pro CPU použijte `--device cpu`.

Výstupem jsou váhy pod `runs/detect/<name>/weights` a pokus o export nejlepšího modelu do ONNX. Archivace neověřovala nový trénink ani jeho kvalitu. Původní lokální dataset a výsledky trénování zůstaly zachované.

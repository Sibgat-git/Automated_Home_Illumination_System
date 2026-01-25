#!/usr/bin/env python3
"""
Download pre-trained TensorFlow Lite models for person detection.
"""

import urllib.request
import os
from pathlib import Path

# Model URLs - using TensorFlow official model zoo
MODELS = {
    "efficientdet_lite0": {
        "url": "https://raw.githubusercontent.com/google-coral/test_data/master/efficientdet_lite0_320_ptq.tflite",
        "filename": "efficientdet_lite0.tflite",
        "description": "EfficientDet-Lite0 - Fast, good accuracy (recommended for Pi 4)"
    },
    "ssd_mobilenet_v2": {
        "url": "https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite",
        "filename": "ssd_mobilenet_v2.tflite",
        "description": "SSD MobileNet V2 - Very fast, moderate accuracy"
    }
}

# COCO labels (class 0 = person)
COCO_LABELS = """0 person
1 bicycle
2 car
3 motorcycle
4 airplane
5 bus
6 train
7 truck
8 boat
9 traffic light
10 fire hydrant
11 stop sign
12 parking meter
13 bench
14 bird
15 cat
16 dog
17 horse
18 sheep
19 cow
20 elephant
21 bear
22 zebra
23 giraffe
24 backpack
25 umbrella
26 handbag
27 tie
28 suitcase
29 frisbee
30 skis
31 snowboard
32 sports ball
33 kite
34 baseball bat
35 baseball glove
36 skateboard
37 surfboard
38 tennis racket
39 bottle
40 wine glass
41 cup
42 fork
43 knife
44 spoon
45 bowl
46 banana
47 apple
48 sandwich
49 orange
50 broccoli
51 carrot
52 hot dog
53 pizza
54 donut
55 cake
56 chair
57 couch
58 potted plant
59 bed
60 dining table
61 toilet
62 tv
63 laptop
64 mouse
65 remote
66 keyboard
67 cell phone
68 microwave
69 oven
70 toaster
71 sink
72 refrigerator
73 book
74 clock
75 vase
76 scissors
77 teddy bear
78 hair drier
79 toothbrush"""


def download_file(url: str, filepath: Path) -> bool:
    """Download a file with progress indication."""
    print(f"Downloading: {filepath.name}")
    print(f"From: {url}")

    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, filepath, show_progress)
        print(f"\n  Saved to: {filepath}")
        return True

    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False


def main():
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("TensorFlow Lite Model Downloader")
    print("=" * 60)

    # Download default model (EfficientDet-Lite0)
    model_info = MODELS["efficientdet_lite0"]
    model_path = models_dir / model_info["filename"]

    if model_path.exists():
        print(f"\nModel already exists: {model_path}")
    else:
        print(f"\nDownloading {model_info['description']}...")
        if not download_file(model_info["url"], model_path):
            print("Failed to download model!")
            return 1

    # Save labels file
    labels_path = models_dir / "coco_labels.txt"
    if not labels_path.exists():
        print(f"\nSaving labels to: {labels_path}")
        with open(labels_path, "w") as f:
            f.write(COCO_LABELS)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Model: {model_path}")
    print(f"Labels: {labels_path}")
    print("=" * 60)
    print("\nYou can now run: python main.py")

    return 0


if __name__ == "__main__":
    exit(main())

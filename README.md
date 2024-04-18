# Driver Awareness Analysis

This is a small part of my master's project to analyze driver awareness while driving using a camera.

## Installation

Please use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following Python library.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe
pip install opencv-python
pip install ffmpegcv
pip install numpy
```

## Usage
Run using the following commands
```bash
# If using a webcam
python driver_awareness.py 
# If using video
python driver_awareness.py -v ./data/video/Driving_video.mp4 
```
Note: Video file not included
## Disclaimer

This project is not fully developed and very inaccurate. This is only to showcase the idea behind the project.
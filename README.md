# Real Time Object Detection and live streaming using OpenCV, Yolo5, PyTorch and Flask
TL;DR: Python application for real time object detection on video feed. live stream accessible over the network.

## Requirements
- Python/Anaconda fresh environment is ideal

## Installation
1. Create a new environment using conda
   ````
   conda create -n <env_name> python=3.8
   ````
2. Switch to the new environment
   ````
   conda activate <env_name>
   ````
4. Add open source package channels
   ````
   conda config --add channels conda-forge
   conda config --add channels pytorch
   conda config --set channel_priority strict
   ````
3. Install Requirements
   ````
   conda install --file requirements.txt
   ````

## Usage
Run `app.py`. This runs a flask server on RPI and the live stream is accessible on `http://127.0.0.1:5000` or `http://RPI_LOCAL_IP:5000`
   ````
   python app.py
   ````

## ToDo
- Controls on Web Interface

:blue_heart:

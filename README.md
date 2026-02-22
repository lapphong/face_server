# Face Server API (Flask + DeepFace)

A simple face recognition REST API built with Flask and DeepFace.
The API compares an uploaded face image against a local image database and returns the best match with a confidence score.

## Info
 - Python version: `3.9`
 - Face detection & recognition using DeepFace (VGG-Face)
 - Simple REST API (/compare)
 - CORS enabled (ready for frontend integration)

## How to install Build

 ### Clone repository
```bash
git clone https://github.com/lapphong/face_server.git
cd face_server
```

 ### Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install tf-keras
python3 server.py
```

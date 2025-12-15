from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import os

app = Flask(__name__)
CORS(app)

images_dir = "images"
known_faces = []
known_names = []

# Load known faces từ thư mục images
for filename in os.listdir(images_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(images_dir, filename)
        try:
            faces = DeepFace.extract_faces(img_path=path, enforce_detection=True)
            if faces:
                known_faces.append(path)
                name = os.path.splitext(filename)[0]
                known_names.append(name)
                print(f"[+] Loaded face: {name}")
        except Exception as e:
            print(f"[!] No face detected in {filename}: {e}")

if not known_faces:
    raise Exception("❌ No images with detectable faces found in images/ directory")

@app.route('/compare', methods=['POST'])
def compare_face():
    if 'image' not in request.files:
        return jsonify({'error': '❌ Missing image file with key \"image\"'}), 400

    uploaded_file = request.files['image']
    upload_path = "temp_uploaded_image.jpg"
    uploaded_file.save(upload_path)

    try:
        result_list = DeepFace.find(
            img_path=upload_path,
            db_path=images_dir,
            enforce_detection=True,
            model_name="VGG-Face"
        )

        result = result_list[0] if isinstance(result_list, list) else result_list

        if result is None or len(result) == 0:
            return jsonify({'error': '❌ No matching face found'}), 400

        best_match = result.iloc[0]
        matched_identity = best_match["identity"]
        matched_name = os.path.splitext(os.path.basename(matched_identity))[0]

        distance = best_match["distance"]
        confidence = round((1 - distance) * 100, 2)

        print(f"[DEBUG] Match: True | Name: {matched_name} | Confidence: {confidence}%")

        return jsonify({
            'match': True,
            'name': matched_name,
            'confidence': confidence
        })

    except Exception as e:
        print("[ERROR] Image processing error:", e)
        return jsonify({'error': f'❌ Image processing error: {str(e)}'}), 500
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)

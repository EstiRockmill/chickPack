import json
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
#יצירת קובץ לשמירת הנתונים
CALIBRATION_FILE = "calibration.json"

# בפעם הראשונה נטעין אוביקט אמיתי וככה הוא ידע מה יחס ההמרה
def save_calibration(pixel_volume, real_volume):
    ratio = real_volume / pixel_volume  # יחס המרה מפיקסלים לסמ"ק
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"ratio": ratio}, f)
    print(f"✔ יחס המרה נשמר לקובץ: {ratio:.5f}")

# אם כבר יצרנו קובץ מנסה להכנס אליו ולהמיר את הנתונינם...
def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
            ratio = data.get("ratio", None)
            print(f"✔ יחס המרה נטען מקובץ: {ratio:.5f}")
            return ratio
    print("⚠ קובץ קליברציה לא קיים.")
    return None

# טוען את מודל YOLO
print("🚀 טוען מודל YOLO...")
model = YOLO("yolov8n.pt")
print("✔ מודל YOLO נטען")

# פונקציה לחישוב מפת עומק
def compute_depth_map(img1, img2):
    print("🧮 מחשב מפת עומק...")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    depth_map = stereo.compute(gray1, gray2)

    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)
    print("✔ מפת עומק נוצרה (צורת depth_map):", depth_map.shape)

    return depth_map

# פונקציה לחישוב נפח לכל אובייקט בנפרד
def compute_object_volumes(depth_map, results):
    print("📏 מחשב נפח לכל אובייקט...")
    object_volumes = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]

            object_depth = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(object_depth) if object_depth.size > 0 else 0

            area_pixels = (x2 - x1) * (y2 - y1)
            volume_pixels = area_pixels * avg_depth

            object_volumes.append({"label": label, "volume_pixels": volume_pixels, "bbox": (x1, y1, x2, y2)})

            print(f" - {label}: area={area_pixels}, avg_depth={avg_depth:.2f}, volume_pixels={volume_pixels:.2f}")

    print(f"✔ נמצאו {len(object_volumes)} אובייקטים עם נפח")
    return object_volumes

# API Flask
app = Flask(__name__)

@app.route('/analyze-images', methods=['POST'])
def analyze_images():
    print("\n=== קיבלנו בקשה ל-analyze-images ===")

    print("request.files.keys():", list(request.files.keys()))
    print("request.content_type:", request.content_type)

    if 'image1' not in request.files or 'image2' not in request.files:
        print("❌ חסר image1 או image2 בבקשה!")
        return jsonify({'error': 'שתי תמונות חייבות להישלח'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    print("✔ image1 filename:", image1.filename)
    print("✔ image2 filename:", image2.filename)

    try:
        img1_data = image1.read()
        img2_data = image2.read()

        print("גודל image1 (bytes):", len(img1_data))
        print("גודל image2 (bytes):", len(img2_data))

        img1 = np.frombuffer(img1_data, np.uint8)
        img2 = np.frombuffer(img2_data, np.uint8)

        img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            print("❌ לא הצלחנו לפענח את התמונות!")
            return jsonify({'error': 'שגיאה בקריאת התמונות'}), 400
        else:
            print("✔ התמונות הומרו בהצלחה ל־NumPy arrays")
    except Exception as e:
        print("❌ שגיאה בקריאת התמונות:", str(e))
        return jsonify({'error': 'שגיאה בקריאת התמונות'}), 400

    print("🚀 מריץ YOLO לזיהוי אובייקטים...")
    results = model(img1)
    total_boxes = sum(len(r.boxes) if r.boxes else 0 for r in results)
    print(f"✔ YOLO זיהה {total_boxes} אובייקטים")

    depth_map = compute_depth_map(img1, img2)

    object_volumes = compute_object_volumes(depth_map, results)

    ratio = load_calibration()

    if ratio is None:
        print("⚠ יחס המרה לא קיים. נדרשת קליברציה ידנית.")
        known_object = input("הזן שם של אובייקט ידוע (כפי שמופיע בזיהוי): ").strip()
        known_objects = [obj for obj in object_volumes if obj["label"] == known_object]

        if known_objects:
            real_volume = float(input(f"הזן נפח אמיתי של {known_object} בסמ״ק: "))
            pixel_volume = np.mean([obj["volume_pixels"] for obj in known_objects])
            save_calibration(pixel_volume, real_volume)
            ratio = real_volume / pixel_volume
        else:
            print("❌ האובייקט שהוזן לא נמצא בזיהוי!")
            return jsonify({"error": "האובייקט לא נמצא בזיהוי!"}), 404

    object_volumes_cm3 = []
    for obj in object_volumes:
        obj["volume_cm3"] = obj["volume_pixels"] * ratio
        object_volumes_cm3.append(obj)

    print("🎯 נפחים לאחר המרה לסמ״ק:")
    for obj in object_volumes_cm3:
        print(f" - {obj['label']} | {obj['volume_cm3']:.2f} סמ״ק | bbox={obj['bbox']}")

    result = []
    for obj in object_volumes_cm3:
        print(f"chechk{obj['volume_cm3']}")
        result.append({
            'label': obj['label'],
            'volume_cm3': obj['volume_cm3']
            # 'bbox': obj['bbox']
        })

    print("✅ מחזיר תשובה ללקוח")
    return jsonify(result)

if __name__ == '__main__':
    print("📡 מפעיל שרת Flask על פורט 5000...")
    app.run(debug=True, host="0.0.0.0", port=5000)
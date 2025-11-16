import cv2
import time
import os
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8x.pt")

# Initialize OpenAI client (optional - only if API key is available)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = None
if openai_api_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        print("✅ OpenAI API key found - AI descriptions enabled")
    except Exception as e:
        print(f"⚠️ OpenAI initialization failed: {e}")
else:
    print("ℹ️ OpenAI API key not found - AI descriptions disabled (motion detection and YOLO detection will still work)")

# Open webcam
cap = cv2.VideoCapture(0)
prev_frame = None
last_description_time = 0
description_interval = 10  # seconds
save_dir = "captured_frames"
os.makedirs(save_dir, exist_ok=True)

print("🚀 AI Vision Motion Detection Started — Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    # Detect motion
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
    motion_detected = cv2.countNonZero(thresh) > 50000
    prev_frame = gray

    if motion_detected:
        results = model(frame)
        annotated_frame = results[0].plot()

        current_time = time.time()
        if current_time - last_description_time > description_interval:
            img_path = os.path.join(save_dir, f"motion_{int(current_time)}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"📸 Captured frame: {img_path}")

            # Get AI description if OpenAI client is available
            if client:
                try:
                    # Send image to GPT-4o for visual understanding
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an AI vision assistant. Describe clearly what objects are visible."},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Describe what objects are visible in this image."},
                                {"type": "image_url", "image_url": f"file://{os.path.abspath(img_path)}"}
                            ]}
                        ]
                    )
                    description = response.choices[0].message.content.strip()
                    print(f"🧠 AI Description: {description}")
                except Exception as e:
                    print(f"⚠️ Error from OpenAI: {e}")
            else:
                # Print detected objects from YOLO results
                if results and len(results) > 0:
                    detected_objects = []
                    for box in results[0].boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        detected_objects.append(f"{class_name} ({confidence:.2f})")
                    if detected_objects:
                        print(f"🔍 Detected objects: {', '.join(detected_objects)}")

            last_description_time = current_time

        cv2.imshow("AI Vision Detection", annotated_frame)
    else:
        cv2.imshow("AI Vision Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

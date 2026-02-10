import cv2, torch, torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

# Config
MODEL_PATH = "mask_detector_v3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder order: with_mask=0, without_mask=1
LABELS = {0: "Mask Detected", 1: "No Mask!"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}

# Load the older but reliable Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(model.last_channel, 128),
        nn.ReLU(), nn.Linear(128, 2)
    )
    # Loading the V3 model you just trained
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.eval().to(DEVICE)

def start_detect():
    model = load_model()
    cap = cv2.VideoCapture(0)
    print("[INFO] Using Haar Cascade Detector. Press 'Q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Faces (tuned to be faster and less "hangy")
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(80, 80))

        for (x, y, w, h) in faces:
            # Predict
            crop = frame[y:y+h, x:x+w]
            if crop.size > 0:
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                with torch.no_grad():
                    # Move image to device and predict
                    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
                    out = model(input_tensor)
                    idx = out.argmax(1).item()
                    prob = torch.softmax(out, 1)[0][idx].item()

                # Results
                label_text = f"{LABELS[idx]} ({prob*100:.1f}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), COLORS[idx], 3)
                cv2.putText(frame, label_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[idx], 2)

        cv2.imshow("Mask Detector v3", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detect()
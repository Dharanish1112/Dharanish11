import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime

# Load Pre-trained Model
model = load_model('polyp_detection_model.h5')  # Replace with your trained model file

# Load an Example Image
image_path = 'example_medical_image.jpg'  # Replace with your medical image
image = cv2.imread(image_path)
input_image = cv2.resize(image, (256, 256))  # Resize to match model input size
input_image = np.expand_dims(input_image, axis=0) / 255.0  # Normalize

# Detect Polyps
predictions = model.predict(input_image)
polyp_mask = (predictions[0] > 0.5).astype(np.uint8)  # Binary mask

# Extract Polyp Features
contours, _ = cv2.findContours(polyp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_info = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50:  # Threshold to filter noise
        x, y, w, h = cv2.boundingRect(contour)
        diameter = round(np.sqrt(4 * area / np.pi), 2)  # Equivalent circular diameter
        stage = "Early" if diameter < 5 else "Advanced"
        infection = "Detected" if diameter > 7 else "None"  # Simplistic logic for demo
        deadline = (datetime.now() + 
                    (datetime.timedelta(days=30) if stage == "Early" else datetime.timedelta(days=7))).strftime("%Y-%m-%d")
        
        output_info.append({
            "Position": (x, y),
            "Diameter (mm)": diameter,
            "Stage": stage,
            "Infection Status": infection,
            "Removal Deadline": deadline
        })
        
        # Annotate Image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255) if stage == "Advanced" else (255, 0, 0), 2)
        cv2.putText(image, f"Stage: {stage}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Visualize Results
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Polyp Detection Results")
plt.show()

# Output Report
print("\n=================== POLYP DETECTION REPORT ===================")
for i, polyp in enumerate(output_info, start=1):
    print(f"\nPolyp {i}:")
    print(f"  Position: {polyp['Position']}")
    print(f"  Diameter: {polyp['Diameter (mm)']} mm")
    print(f"  Stage: {polyp['Stage']}")
    print(f"  Infection Status: {polyp['Infection Status']}")
    print(f"  Recommended Removal Deadline: {polyp['Removal Deadline']}")
print("\n==============================================================")

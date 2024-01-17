import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
from PIL import Image, ImageTk

# Load the pre-trained model
def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(weights_file)
    return model

# Function to detect emotion from an image
def detect_emotion(file_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use your face detection logic here
    # For example, you can use Haar cascades or any other face detection method
    # faces = your_face_detection_function(gray_image)
    # ...

    # Assuming you have a region of interest (ROI) containing the face
    # roi = extract_roi(gray_image, face_coordinates)
    # ...

    # Resize the ROI to match the model input size
    roi = cv2.resize(roi, (48, 48))

    # Preprocess the image
    roi = np.expand_dims(roi, axis=0)
    roi = roi.reshape(1, 48, 48, 1)

    # Make a prediction
    emotion_label = EMOTIONS_LIST[np.argmax(model.predict(roi))]
    
    return emotion_label

# Function to open file dialog and get the file path
def browse_file():
    file_path = filedialog.askopenfilename()
    emotion = detect_emotion(file_path)
    result_label.config(text=f"Predicted Emotion: {emotion}")

# GUI setup
root = tk.Tk()
root.title("Emotion Detector")

# Load the pre-trained model
model = load_model("model_a1.json", "model_weights.h5")

# Define your emotions list
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create and configure GUI elements
browse_button = tk.Button(root, text="Browse Image", command=browse_file)
browse_button.pack(pady=20)

result_label = Label(root, text="Predicted Emotion: ")
result_label.pack()

# Run the GUI
root.mainloop()

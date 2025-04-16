#!/usr/bin/env python
# coding: utf-8

# # IBM Gesture Trainer

# In[3]:


# === Importing Required Libraries ===
import cv2  # For webcam video capture and display
import mediapipe as mp  # For hand landmark detection
import numpy as np  # For array manipulations
import pandas as pd  # For saving and reading data in CSV format
import os  # For file existence checks
from sklearn.ensemble import RandomForestClassifier  # ML model for gesture classification
from tkinter import *  # For building the GUI
from tkinter import simpledialog  # For user input dialog box in GUI
from threading import Thread  # For running functions in the background
import joblib  # For saving and loading trained ML models

# === Mediapipe Hand Tracking Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,  # Video stream, not static images
                       max_num_hands=1,  # Only one hand at a time
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils  # For drawing hand landmarks

# === File Paths for Dataset and Model ===
DATA_FILE = "gesture_data.csv"  # CSV file to save gesture features with labels
MODEL_FILE = "gesture_model.pkl"  # Trained model file path

# === Global Flags and Variables ===
collecting = False  # Flag to control data collection thread
predicting = False  # Flag to control live prediction
current_label = ""  # Gesture label user provides during data collection

# === Function to Extract Features from Hand Landmarks ===
def extract_features(landmarks):
    """
    Converts list of 21 hand landmarks (each with x, y, z) into a flat array of 63 features.
    """
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

# === Data Collection Thread Function ===
def collect_data():
    global collecting, current_label
    cap = cv2.VideoCapture(0)  # Open webcam
    collected = 0  # Number of samples collected

    while collecting:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame for natural feel
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                features = extract_features(handLms.landmark)
                data_row = list(features) + [current_label]  # Append label to features
                df = pd.DataFrame([data_row])

                # Save to CSV
                if os.path.exists(DATA_FILE):
                    df.to_csv(DATA_FILE, mode='a', header=False, index=False)
                else:
                    cols = [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']] + ["label"]
                    df.to_csv(DATA_FILE, mode='w', header=cols, index=False)

                collected += 1

        # Show live feedback
        cv2.putText(frame, f"Collecting: {current_label} ({collected})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Train ML Model Function ===
def train_model():
    if not os.path.exists(DATA_FILE):
        print("No data to train.")
        return

    df = pd.read_csv(DATA_FILE)
    X = df.drop('label', axis=1)  # Features
    y = df['label']  # Labels

    model = RandomForestClassifier(n_estimators=100)  # Simple, effective model
    model.fit(X, y)  # Train the model
    joblib.dump(model, MODEL_FILE)  # Save it for future use
    print("Model trained and saved!")

# === Live Prediction Thread Function ===
def predict_live():
    global predicting
    if not os.path.exists(MODEL_FILE):
        print("Train a model first!")
        return

    model = joblib.load(MODEL_FILE)  # Load trained model
    cap = cv2.VideoCapture(0)

    while predicting:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                features = extract_features(handLms.landmark).reshape(1, -1)
                pred = model.predict(features)[0]
                cv2.putText(frame, f"Prediction: {pred}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        cv2.imshow("Live Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === GUI Functions ===
def start_collection():
    global collecting, current_label
    label = simpledialog.askstring("Gesture Label", "Enter label for gesture:")
    if label:
        current_label = label
        collecting = True
        Thread(target=collect_data).start()  # Run data collection in background

def stop_collection():
    global collecting
    collecting = False

def start_prediction():
    global predicting
    predicting = True
    Thread(target=predict_live).start()  # Run live prediction in background

def stop_prediction():
    global predicting
    predicting = False

def train_model_btn():
    train_model()

# === GUI Setup Using Tkinter ===
root = Tk()
root.title("AI Hand Gesture Trainer")
root.geometry("300x300")
root.config(bg="#eef")

Label(root, text="AI Hand Gesture Trainer", font=("Arial", 16, "bold"), bg="#eef").pack(pady=10)

Button(root, text="Start Data Collection", command=start_collection, width=25, bg="green", fg="white").pack(pady=5)
Button(root, text="Stop Data Collection", command=stop_collection, width=25, bg="darkgreen", fg="white").pack(pady=5)

Button(root, text="Train Model", command=train_model_btn, width=25, bg="blue", fg="white").pack(pady=10)

Button(root, text="Start Live Prediction", command=start_prediction, width=25, bg="purple", fg="white").pack(pady=5)
Button(root, text="Stop Prediction", command=stop_prediction, width=25, bg="indigo", fg="white").pack(pady=5)

Button(root, text="Exit", command=root.quit, width=25, bg="red", fg="white").pack(pady=20)

root.mainloop()  # Start GUI loop


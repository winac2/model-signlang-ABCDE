import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import pickle

# load model
MODEL_PATH = './hand_gesture_model.pkl'
model = pickle.load(open(MODEL_PATH, 'rb'))

# define label
LABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}  # dựa trên dữ liệu của model đã train
#0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
# khởi tạo mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")
        self.root.geometry("800x600")

        # hiện video
        self.video_label = Label(self.root)
        self.video_label.pack()

        # dự đoán label
        self.prediction_label = Label(self.root, text="Gesture: None", font=("Helvetica", 16))
        self.prediction_label.pack(pady=20)

        # nút
        self.start_button = Button(self.root, text="Start Camera", command=self.start_camera, font=("Helvetica", 14))
        self.start_button.pack(side="left", padx=10)

        self.stop_button = Button(self.root, text="Stop Camera", command=self.stop_camera, font=("Helvetica", 14))
        self.stop_button.pack(side="left", padx=10)

        # khởi tạo video
        self.cap = None
        self.running = False

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.process_frame()

    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')

    def process_frame(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # xác định tay
                results = hands.process(frame_rgb)
                prediction = "None"
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # vẽ các landmark trên tay
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # xuất landmark
                        x_ = []
                        y_ = []

                        for landmark in hand_landmarks.landmark:
                            x_.append(landmark.x)
                            y_.append(landmark.y)

                        # tạo bounding box xung quanh tay
                        x_min, x_max = min(x_), max(x_)
                        y_min, y_max = min(y_), max(y_)

                        # Convert sang toạ độ pixel
                        h, w, _ = frame.shape
                        x_min, x_max = int(x_min * w), int(x_max * w)
                        y_min, y_max = int(y_min * h), int(y_max * h)

                        # vẽ bounding box khi hiển thị
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # xuất data landmark để dự đoán
                        data_aux = []
                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                            data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                        # dự đoán với ngưỡng (threshold)
                        data_aux = np.array(data_aux).reshape(1, -1)
                        try:
                            pred_prob = model.predict_proba(data_aux)
                            if max(pred_prob[0]) > 0.7:  # đặt ngưỡng là 80%
                                pred = np.argmax(pred_prob, axis=1)
                                prediction = LABELS.get(int(pred[0]), "Unknown Gesture")
                            else:
                                prediction = "None"
                        except Exception as e:
                            prediction = "Error"

                # hiện dự đoán
                self.prediction_label.config(text=f"Gesture: {prediction}")

                # hiện khung
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # lên khung tiếp
            self.root.after(10, self.process_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()

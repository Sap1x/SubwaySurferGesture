# 🎮 Subway Surfer Gesture Control with OpenCV & MediaPipe

Control **Subway Surfer** (or any game with arrow keys) using **hand gestures** in real-time!  
This project uses **OpenCV** for webcam input, **MediaPipe** for hand tracking, and **PyAutoGUI** to simulate keyboard inputs.

---

## 📌 Features
- 🖐 Real-time **hand landmark detection** using MediaPipe.
- ↔️ Swipe left/right to move.
- ⬆️ Swipe up to jump.
- ⬇️ Swipe down to roll.
- ⏱ Gesture cooldown to prevent accidental repeats.
- Works with any PC game that uses **arrow keys**.

---

## 🛠 Tech Stack
- **Python 3.9+**
- [OpenCV](https://opencv.org/) (`cv2`)
- [MediaPipe](https://developers.google.com/mediapipe)
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)

---

## 📦 Installation
Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/SubwaySurferGesture.git
cd SubwaySurferGesture

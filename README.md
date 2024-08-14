
---

# 🎨 Air Canvas: Drawing in the Air with Hand Gestures

![Air Canvas](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen) ![Mediapipe](https://img.shields.io/badge/Mediapipe-0.8.6-yellow)

## 🚀 Project Overview

**Air Canvas** is an innovative project that allows you to draw in the air using hand gestures, all captured through your webcam. Imagine waving your hands and creating digital art in real-time, without ever touching a screen or using a physical pen! This project combines the power of Python, OpenCV, and MediaPipe to bring your ideas to life, right in the air.

## ✨ Features

- **Hand Gesture Detection**: Utilize the power of MediaPipe to accurately detect and track your hand movements.
- **Color Palette & Eraser**: Switch between various colors, an eraser tool, and a "Clear All" button to wipe the canvas clean.
- **Both Hands Support**: Draw using either your left or right hand, providing flexibility and ease of use.
- **Last Point Indicator**: Easily continue your drawing with a marker indicating your last drawing point.
- **Smooth Drawing Experience**: Drawing only activates when the index finger is extended, ensuring precise control.
- **Real-Time Updates**: See your drawings appear instantly on the canvas as you move your hand.

## 🛠️ Technologies Used

- **Python** 🐍: The core language powering the project.
- **OpenCV** 📸: For capturing the webcam feed and handling image processing.
- **MediaPipe** ✋: To track and detect hand gestures in real-time.
- **NumPy** 🧮: For efficient handling of array operations.

## 🎯 How It Works

1. **Setup the Webcam Feed**: The webcam captures your hand movements in real-time.
2. **Hand Detection**: MediaPipe processes the frames to detect hand landmarks and identifies the tip of your index finger.
3. **Gesture Recognition**: Depending on the position and gesture of your hand, the system determines whether to draw, erase, or switch colors.
4. **Drawing on Canvas**: Your gestures are translated into drawings on a separate canvas, displayed side-by-side with the webcam feed.
5. **Color and Tool Selection**: Easily switch between different colors, the eraser, or clear the entire canvas using intuitive hand gestures or button clicks.


## 🖥️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sriramkrish68/Gesture-Canvas.git
   cd Gesture-Canvas
   ```

2. **Install the Required Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Start Drawing**: Use your webcam to start drawing in the air!

## 📚 Future Enhancements

- **Add Shapes & Text**: Introduce gestures for drawing shapes like circles, squares, and adding text.
- **Multi-Canvas Support**: Allow switching between multiple canvases.
- **AI-Assisted Drawing**: Integrate AI to enhance or auto-correct your drawings.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.



## 💬 Let's Connect

- [My LinkedIn](https://www.linkedin.com/in/sriramkrish379/)



---

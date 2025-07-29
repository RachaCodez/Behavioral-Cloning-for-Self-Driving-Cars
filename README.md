
# 🧠 Behavioral Cloning for Self-Driving Cars

This project implements **behavioral cloning using a Convolutional Neural Network (CNN)** to autonomously drive a car in the [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim).

It uses a deep learning model based on the NVIDIA architecture to predict steering angles from center camera images.

---

## 🚗 Project Structure



behavioral-cloning/
├── data.py                   # Preprocessing & batch generator
├── drive.py                 # Inference server (SocketIO + Flask)
├── model.py                 # CNN training script (NVIDIA-based)
├── model.h5                 # Trained model weights
├── model.json               # (Optional) Keras model config
├── weights\_logger\_callback.py # Custom Keras callback for logging
├── venv/                    # Python virtual environment
├── README.md                # Project description and usage
└── .gitignore



---

## 📦 Requirements

```bash
Python 3.8+
TensorFlow 2.x
Keras
NumPy
Flask
eventlet
python-socketio
Pillow
matplotlib
scikit-image
pandas
````

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🧠 Model Architecture

The CNN model is based on **NVIDIA's end-to-end architecture**, featuring:

* 3× Convolutional layers
* 2× MaxPooling layers
* Fully connected Dense layers
* Dropout for regularization

Input image is cropped, resized to 66×200, and normalized between \[0, 1].

---

## 🏁 How to Train

Make sure your `driving_log.csv` and image dataset are in the `data/` directory.

```bash
python model.py
```

This script:

* Loads data
* Applies preprocessing and augmentation
* Trains the model
* Saves the model to `model.h5` and `model.json`

---

## 🧪 How to Run in Simulator

1. Launch the Udacity simulator in **Autonomous Mode**
2. In terminal, run:

```bash
python drive.py model.h5
```

3. Simulator will send camera images to the model.
4. Model returns steering and throttle in real time via Flask + SocketIO.

---

## 🖼️ Sample Prediction Loop

```python
steering_angle = float(model.predict(transformed_image_array, batch_size=1))
throttle = .2 if float(speed) > 5 else 1.0
```

---

## 📂 Dataset Format

The simulator exports a `driving_log.csv` with columns:

```csv
center,left,right,steering,throttle,brake,speed
```

Images are stored in `IMG/`.

---

## 📈 Future Work

* Add road segmentation or lane detection
* Integrate RNN/LSTM for temporal dependencies
* Drive in real-world testbed (Raspberry Pi + PiCam)

---

## 👨‍💻 Author

**Ruthvik Racha**
📧 [ruthvik@example.com](mailto:ruthvik@example.com)
🌐 GitHub: [github.com/ruthvikracha](https://github.com/ruthvikracha)

---

## 📜 License

MIT License — see `LICENSE` for details.

---


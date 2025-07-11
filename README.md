
# 🧠 CNN for Image Recognition on MNIST (Day 5)

A Convolutional Neural Network (CNN) built with TensorFlow and Keras for recognizing handwritten digits from the MNIST dataset.  
This project introduces the concept of convolutional layers and pooling layers for image classification tasks.

---

## 📄 Project Overview

This project performs the following:

✅ Loads and preprocesses the MNIST dataset.  
✅ Builds a CNN with two convolutional + pooling layers, followed by dense layers.  
✅ Trains the CNN on the training dataset.  
✅ Evaluates the model’s performance on the test dataset.

The use of convolutional layers enables the model to extract spatial features more effectively than a simple dense network.

---

## 🗃️ Dataset

- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)
- 60,000 training images, 10,000 test images.
- Images: 28×28 grayscale.
- Labels: Digits 0–9.

The dataset is downloaded automatically using `keras.datasets.mnist.load_data()`.

---

## 🔧 Technologies Used

- [Python 3.x](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/) & Keras
- Jupyter Notebook / Python script compatible

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yashpalsince2004/CNN_For_ImageRecognition.git
cd cnn-mnist-recognition
```

### 2️⃣ Install dependencies
It is recommended to use a virtual environment.
```bash
pip install tensorflow
```

### 3️⃣ Run the script
```bash
python Day5_CNN_for_Image_Recognition.py
```

---

## 🧪 Model Architecture

| Layer                 | Details                              |
|-----------------------|--------------------------------------|
| Input Layer           | Reshape 28×28 → 28×28×1             |
| Conv2D                | 32 filters, 3×3 kernel, ReLU        |
| MaxPooling2D          | 2×2 pool size                       |
| Conv2D                | 64 filters, 3×3 kernel, ReLU        |
| MaxPooling2D          | 2×2 pool size                       |
| Flatten               | Converts 2D → 1D                   |
| Dense                 | 64 neurons, ReLU                   |
| Output Dense          | 10 neurons, softmax                |

---

## 📈 Results

After training for **5 epochs**, the CNN typically achieves test accuracy in the range of **~98% or higher**.

Sample output:
```
Epoch 1/5 …
…
CNN Test Accuracy: 0.989
```

---

## 🌟 Key Learning Points

- Understanding CNN layers: `Conv2D`, `MaxPooling2D`.
- Data reshaping for CNN input.
- Training & evaluating a CNN with Keras.
- Improvement of CNN performance over simple dense networks.

---

## 📚 References

- MNIST dataset by Yann LeCun et al.
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)

---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).

---

## 🤝 Contributions

Contributions and improvements are welcome. Please open an issue or submit a pull request.

---

## 🌱 Future Work

- Add dropout layers to prevent overfitting.
- Experiment with more epochs and batch normalization.
- Try more complex datasets such as CIFAR-10 or Fashion-MNIST.

---

## 👨‍💻 Author

**Yash**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/yash-pal-since2004)

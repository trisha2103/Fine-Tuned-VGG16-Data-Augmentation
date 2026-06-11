# 🐶🐱 Dogs vs Cats — Fine-Tuned VGG16 & Data Augmentation

> A deep learning project that classifies images as **Dogs** or **Cats** using a custom CNN baseline, transfer learning with fine-tuned **VGG16**, and image augmentation to boost performance on limited data.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-VGG16-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

This project solves a classic **binary image classification** problem — distinguishing dogs from cats — using two deep learning approaches and a data augmentation pipeline:

1. **Custom CNN** — a lightweight 2-block convolutional network trained from scratch as a baseline
2. **Fine-Tuned VGG16** ⭐ — transfer learning with ImageNet-pretrained VGG16, frozen layers, and a custom softmax output head
3. **Data Augmentation** — 8 image transforms applied via `ImageDataGenerator` to synthetically expand the training set

---

## 🗂️ Dataset Structure

```
dogs-vs-cats/
├── train/
│   ├── cat/        # 500 images
│   └── dog/        # 500 images
├── valid/
│   ├── cat/        # 100 images
│   └── dog/        # 100 images
└── test/
    ├── cat/        # 50 images
    └── dog/        # 50 images
```

> All images are resized to **224×224** and preprocessed using VGG16's built-in normalization.

---

## 🔄 Data Augmentation

To combat overfitting on a small dataset, the following transforms are applied using Keras `ImageDataGenerator`:

| Transform | Value |
|---|---|
| Rotation range | ±10° |
| Width shift | ±10% |
| Height shift | ±10% |
| Shear range | 15% |
| Zoom range | ±10% |
| Channel shift | ±10.0 |
| ZCA whitening | Enabled |
| Horizontal flip | Enabled |

```python
gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.1,
    channel_shift_range=10.,
    zca_whitening=True,
    horizontal_flip=True
)

# Generate and save 10 augmented copies per image
for x, val in zip(gen.flow(image,
                            save_to_dir='train/dog',
                            save_prefix='aug-image-',
                            save_format='jpeg'), range(10)):
    pass
```

---

## 🧠 Models

### Model 1 — Custom CNN (Baseline)

| Layer | Config |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU, same padding |
| MaxPool2D | 2×2, stride 2 |
| Conv2D | 64 filters, 3×3, ReLU, same padding |
| MaxPool2D | 2×2, stride 2 |
| Flatten | — |
| Dense (output) | 2 units, Softmax |

- Trained for **10 epochs** · Batch size **10** · Adam lr **0.0001**

### Model 2 — Fine-Tuned VGG16 ⭐

| Property | Details |
|---|---|
| **Base Model** | `VGG16` pretrained on ImageNet |
| **Frozen Layers** | All VGG16 layers (13 conv + 3 FC) |
| **Custom Head** | `Dense(2, activation='softmax')` |
| **Input Shape** | `(224, 224, 3)` |
| **Classes** | `cat`, `dog` |

- Trained for **5 epochs** · Batch size **10** · Adam lr **0.0001**

---

## 📖 How It Works

### Step 1 — Organize the Dataset

```python
for i in random.sample(glob.glob('cat*'), 500):
    shutil.move(i, 'train/cat')
for i in random.sample(glob.glob('dog*'), 500):
    shutil.move(i, 'train/dog')
# repeated for valid (100) and test (50)
```

### Step 2 — Augment Training Data

```python
gen = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                          horizontal_flip=True, zca_whitening=True, ...)
aug_images = [next(gen.flow(image))[0].astype(np.uint8) for i in range(10)]
```

### Step 3 — Create Data Generators

```python
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(train_path, target_size=(224, 224),
                      classes=['cat', 'dog'], batch_size=10)
```

### Step 4 — Build & Train Fine-Tuned VGG16

```python
vgg16_model = tf.keras.applications.vgg16.VGG16()

model = Sequential()
for layer in vgg16_model.layers[:-1]:  # strip original output layer
    model.add(layer)

for layer in model.layers:
    layer.trainable = False            # freeze all VGG16 weights

model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)
```

### Step 5 — Evaluate with Confusion Matrix

```python
predictions = model.predict(x=test_batches, verbose=0)
cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))
plot_confusion_matrix(cm=cm, classes=['cat', 'dog'], title='Confusion Matrix')
```

---

## 📊 Training Configuration

| Parameter | Custom CNN | VGG16 Fine-Tuned |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning Rate | 0.0001 | 0.0001 |
| Loss | Categorical Crossentropy | Categorical Crossentropy |
| Batch Size | 10 | 10 |
| Epochs | 10 | 5 |
| Image Size | 224 × 224 | 224 × 224 |

---

## 🚀 Features

- ✅ Automated dataset splitting into `train / valid / test`
- ✅ 8-transform **data augmentation** pipeline with save-to-disk support
- ✅ **VGG16 transfer learning** with frozen pretrained weights
- ✅ Custom CNN baseline for performance comparison
- ✅ **Confusion matrix** visualization for both models
- ✅ VGG16-specific preprocessing pipeline

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/trisha2103/Dogs-vs-Cats-VGG16.git
cd Dogs-vs-Cats-VGG16

# 2. Install dependencies
pip install tensorflow scikit-learn matplotlib numpy

# 3. Add your dataset
# Place raw cat*.jpg and dog*.jpg images inside dogs-vs-cats/train/
# Then run the data organization cell in the notebook

# 4. Launch notebooks
jupyter notebook "Data_Augmentation_-_image_-_CNN_.ipynb"
jupyter notebook "CNN_-_Fine_Tuned_VGG16_model.ipynb"
```

> 💡 A GPU is recommended for faster training. TensorFlow will fall back to CPU automatically.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🧠 `TensorFlow / Keras` | Model building & training |
| 🏛️ `VGG16` | Pretrained feature extractor |
| 🔄 `ImageDataGenerator` | Augmentation & data streaming |
| 📊 `scikit-learn` | Confusion matrix evaluation |
| 📈 `matplotlib` | Visualization |
| 🐍 `NumPy` | Numerical operations |

---

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- GPU recommended for faster training

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👩‍💻 Author

**Trisha** — [@trisha2103](https://github.com/trisha2103)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using TensorFlow, VGG16 & Data Augmentation</p>

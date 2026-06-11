#  Dogs vs Cats — Fine-Tuned VGG16 & Data Augmentation

> Classify images as **Dogs** or **Cats** using a custom CNN baseline, transfer learning with fine-tuned **VGG16**, and an 8-transform image augmentation pipeline.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-VGG16-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

This project solves the classic **Dogs vs Cats** binary classification problem using two deep learning approaches and a data augmentation pipeline:

1. **Custom CNN** — a lightweight 2-block convolutional network trained from scratch as a baseline
2. **Fine-Tuned VGG16** ⭐ — transfer learning with ImageNet-pretrained VGG16, frozen layers, and a custom softmax output head
3. **Data Augmentation** — 8 image transforms to synthetically expand the training set and reduce overfitting

---

## 🖼️ Sample Dataset Images

### 🐱 Cats

| | | | | |
|:---:|:---:|:---:|:---:|:---:|
| ![cat693](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/cat.693.jpg) | ![cat814](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/cat.814.jpg) | ![cat836](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/cat.836.jpg) | ![cat903](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/cat.903.jpg) | ![cat1009](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/cat.1009.jpg) |
| cat.693 | cat.814 | cat.836 | cat.903 | cat.1009 |

### 🐶 Dogs

| | | | |
|:---:|:---:|:---:|:---:|
| ![dog4](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/dog.4.jpg) | ![dog32](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/dog.32.jpg) | ![dog1160](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/dog.1160.jpg) | ![dog1289](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/dog.1289.jpg) |
| dog.4 | dog.32 | dog.1160 | dog.1289 |

### 🔄 Augmented Sample

| Original → Augmented |
|:---:|
| ![aug](https://raw.githubusercontent.com/trisha2103/Dogs-vs-Cats-VGG16/main/aug-image-_0_9718.jpeg) |
| `aug-image-_0_9718.jpeg` |

> All images are resized to **224×224** and preprocessed with VGG16's built-in normalization before training.

---

## 🚀 Features

- ✅ Automated dataset splitting into `train / valid / test`
- ✅ **8-transform data augmentation** pipeline with save-to-disk support
- ✅ **VGG16 transfer learning** with all base layers frozen
- ✅ Custom CNN baseline for comparison
- ✅ **Confusion matrix** visualization for both models
- ✅ VGG16-specific preprocessing pipeline

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

---

## 🔄 Data Augmentation

8 transforms applied via Keras `ImageDataGenerator`, with augmented copies saved back to the training folder:

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

# Save 10 augmented copies per image back to training folder
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

Trained for **10 epochs** · Batch size **10** · Adam lr **0.0001**

### Model 2 — Fine-Tuned VGG16 ⭐

| Property | Details |
|---|---|
| **Base Model** | `VGG16` pretrained on ImageNet |
| **Frozen Layers** | All VGG16 layers (13 conv + 3 FC) |
| **Custom Head** | `Dense(2, activation='softmax')` |
| **Input Shape** | `(224, 224, 3)` |
| **Classes** | `cat`, `dog` |

Trained for **5 epochs** · Batch size **10** · Adam lr **0.0001**

```python
vgg16_model = tf.keras.applications.vgg16.VGG16()

model = Sequential()
for layer in vgg16_model.layers[:-1]:  # remove original output layer
    model.add(layer)

for layer in model.layers:
    layer.trainable = False            # freeze all VGG16 weights

model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)
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

## 📖 How It Works

### Step 1 — Organize Dataset
```python
for i in random.sample(glob.glob('cat*'), 500):
    shutil.move(i, 'train/cat')
for i in random.sample(glob.glob('dog*'), 500):
    shutil.move(i, 'train/dog')
# repeated for valid (100) and test (50)
```

### Step 2 — Augment Training Data
```python
aug_images = [next(gen.flow(image))[0].astype(np.uint8) for i in range(10)]
```

### Step 3 — Create Data Generators
```python
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(train_path, target_size=(224, 224),
                      classes=['cat', 'dog'], batch_size=10)
```

### Step 4 — Evaluate with Confusion Matrix
```python
cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))
plot_confusion_matrix(cm=cm, classes=['cat', 'dog'], title='Confusion Matrix')
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🧠 `TensorFlow / Keras` | Model building & training |
| 🏛️ `VGG16` | Pretrained feature extractor |
| 🔄 `ImageDataGenerator` | Augmentation & data streaming |
| 📊 `scikit-learn` | Confusion matrix |
| 📈 `matplotlib` | Visualization |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/trisha2103/Dogs-vs-Cats-VGG16.git
cd Dogs-vs-Cats-VGG16

# 2. Install dependencies
pip install tensorflow scikit-learn matplotlib numpy

# 3. Place raw cat*.jpg and dog*.jpg images in dogs-vs-cats/train/
# Then run the data organization cell in the notebook

# 4. Launch notebooks
jupyter notebook "Data Augmentation - image - CNN .ipynb"
jupyter notebook "CNN - Fine Tuned VGG16 model.ipynb"
```
---

## 👩‍💻 Author

**Trisha** — [@trisha2103](https://github.com/trisha2103)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using TensorFlow, VGG16 & Data Augmentation</p>

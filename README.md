# 🌾 Rice Leaf Disease Classification Using Deep Learning

### An AI-powered Computer Vision Project for Automated Rice Leaf Disease Identification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Rice Leaf Disease Classification is a deep learning project that uses Computer Vision and Convolutional Neural Networks (CNNs) to identify diseases affecting rice leaves.

The project classifies rice leaf images into three disease categories:

1. Bacterial Leaf Blight
2. Brown Spot
3. Leaf Smut

Multiple deep learning approaches were explored, including a baseline CNN, an augmented CNN and transfer learning using MobileNetV2.

The best-performing experimental model, MobileNetV2 with a 320 × 320 input resolution, achieved **83.33% observed test accuracy**.

The project also incorporates Explainable AI using Grad-CAM to understand model attention and investigates potential background sensitivity.

## 🎯 Business Problem

Rice diseases can significantly affect agricultural productivity and crop quality. Traditional disease identification often relies on manual observation and agricultural expertise.

This project explores an AI-assisted approach to identify rice leaf diseases from photographs, potentially supporting earlier disease recognition and agricultural decision-making.

## 🎯 Project Objectives

- Analyse and understand rice leaf image data.
- Perform image preprocessing and data augmentation.
- Develop CNN-based classification models.
- Apply transfer learning using MobileNetV2.
- Compare model performance.
- Interpret predictions using Grad-CAM.
- Investigate classification errors and background sensitivity.
- Prepare the selected model for future application deployment.

## 📂 Dataset Description

The dataset contains **119 rice leaf images** across three disease categories.

| Disease Category | Number of Images |
|---|---:|
| Bacterial Leaf Blight | 40 |
| Brown Spot | 40 |
| Leaf Smut | 39 |
| **Total** | **119** |

The original dataset specification indicated 120 images, but only 119 valid images were available in the extracted dataset.

### Dataset Distribution

![Dataset Distribution](outputs/dataset_class_distribution.png)

### Sample Rice Leaf Images

![Rice Leaf Disease Samples](outputs/sample_rice_leaf_images.png)

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming |
| TensorFlow | Deep Learning |
| Keras | CNN and transfer learning |
| MobileNetV2 | Pretrained image classification |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Model evaluation |
| PIL | Image processing |
| Jupyter Notebook | Development environment |

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset structure, image characteristics and class distribution.

The analysis included:

- Disease class distribution.
- Image width and height distribution.
- Image resolution analysis.
- Aspect ratio analysis.
- File size distribution.
- Image format and colour mode validation.

### 1. Image Dimension Distribution

![Image Dimensions](outputs/image_dimension_distribution.png)

### 2. Image Resolution Analysis

![Image Resolution](outputs/image_resolution_distribution.png)

### 3. Aspect Ratio Distribution

![Aspect Ratio](outputs/aspect_ratio_distribution.png)

### 4. File Size Distribution

![File Size](outputs/image_file_size_distribution.png)

### Key EDA Findings

- 119 valid RGB images were identified.
- No corrupted images were detected.
- No byte-identical duplicate images were found.
- 34 unique image dimension combinations were identified.
- Bacterial Leaf Blight images had consistent dimensions, unlike the other two categories.
- The images had wide aspect ratios, requiring careful resizing to avoid distortion.

## 🧹 Image Preprocessing

The images were processed using the following pipeline:

1. Load images using PIL.
2. Convert images to RGB.
3. Preserve the original aspect ratio.
4. Resize images using LANCZOS interpolation.
5. Add white padding to obtain uniform dimensions.
6. Normalise images using MobileNetV2 preprocessing.

**Final input dimensions:** 320 × 320 × 3

### Image Preprocessing Visualisation

![Image Preprocessing](outputs/image_preprocessing_visualization.png)

## 🔄 Data Augmentation

Data augmentation was explored to improve model generalisation.

The following transformations were applied:

- Random horizontal flipping.
- Random rotation.
- Random zoom.
- Random contrast adjustment.

### Augmentation Examples

![Data Augmentation](outputs/data_augmentation_visualization.png)

## 🧠 Model Development

Four model configurations were developed and evaluated.

### 1. Baseline CNN

A custom CNN architecture was developed using convolutional layers, max pooling, global average pooling and dropout.

### 2. Augmented CNN

The baseline CNN architecture was extended with data augmentation layers to investigate the effect of image transformations.

### 3. MobileNetV2 – 224 × 224

A pretrained MobileNetV2 model using ImageNet weights was used as a feature extractor with a custom classification head.

### 4. MobileNetV2 – 320 × 320

A higher-resolution MobileNetV2 model was developed to investigate whether increased input resolution could improve disease classification.

## 📊 Model Performance Comparison

The following table summarises the experimental results.

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|---|---:|---:|---:|
| Baseline CNN | 63.86% | 44.44% | Not evaluated |
| Augmented CNN | 62.65% | 44.44% | Not evaluated |
| MobileNetV2 – 224 | 72.29% | 83.33% | 66.67% |
| **MobileNetV2 – 320** | **79.52%** | **83.33%** | **83.33%** |

*Note: The baseline and augmented CNN used different padding configurations from the later MobileNetV2 experiments. Their results should therefore not be interpreted as a perfectly controlled comparison.*

### Model Accuracy Comparison

![Model Accuracy Comparison](outputs/model_accuracy_comparison.png)

### Training and Validation Curves

![Training Curves](outputs/training_validation_curves.png)

## 🏆 Best Model – MobileNetV2 (320 × 320)

MobileNetV2 with 320 × 320 input resolution was selected as the provisional leading candidate.

### Evaluation Metrics

| Metric | Result |
|---|---:|
| Test Accuracy | 83.33% |
| Macro Precision | 0.86 |
| Macro Recall | 0.83 |
| Macro F1-score | 0.83 |
| Test Images | 18 |
| Correct Predictions | 15 |
| Incorrect Predictions | 3 |
| Total Parameters | 2,261,827 |

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix_mobilenet320.png)

### Classification Report

![Classification Report](outputs/classification_report_mobilenet320.png)

### Class-wise Performance

| Disease | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bacterial Leaf Blight | 0.75 | 1.00 | 0.86 |
| Brown Spot | 1.00 | 0.67 | 0.80 |
| Leaf Smut | 0.83 | 0.83 | 0.83 |

## 🔬 Explainable AI – Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented to visualise the image regions contributing to model predictions.

This helps investigate whether the model focuses on relevant leaf regions and visible disease symptoms.

### Grad-CAM Visualisations

![Grad-CAM Analysis](outputs/gradcam_analysis.png)

### Key Observations

- Some activation regions overlapped with visible leaf symptoms.
- Background activation was also observed.
- The misclassified example showed relatively diffuse attention.
- Grad-CAM provided useful insights into model behaviour, although it does not prove biological correctness.

## ⚠️ Error Analysis

Three out of 18 test images were incorrectly classified.

| Actual Class | Predicted Class | Confidence |
|---|---|---:|
| Leaf Smut | Bacterial Leaf Blight | 50.77% |
| Brown Spot | Leaf Smut | 38.28% |
| Brown Spot | Bacterial Leaf Blight | 43.51% |

### Misclassification Visualisation

![Misclassified Images](outputs/misclassified_images.png)

Brown Spot accounted for two of the three classification errors, indicating that further improvement is needed in distinguishing this disease from the other categories.

## 🧪 Background and Padding Sensitivity Analysis

A sensitivity experiment was conducted to investigate how modifying near-white image regions influences model predictions.

Three input conditions were compared:

- Original white padding.
- Gray-modified near-white regions.
- Black-modified near-white regions.

### Experimental Results

| Metric | White | Gray | Black |
|---|---:|---:|---:|
| Accuracy | 83.33% | 88.89% | 88.89% |
| Average Confidence | 61.33% | 61.89% | 54.04% |
| Prediction Stability | 100% | 72.22% | 72.22% |

![Padding Sensitivity Analysis](outputs/padding_sensitivity_analysis.png)

### Interpretation

Five of the 18 predictions changed under each modified condition.

This indicates sensitivity to changes in near-white image regions. However, because the modified pixels may include naturally white image regions, the experiment does not independently establish background bias.

## 🚀 Production Recommendation

MobileNetV2 with 320 × 320 input resolution is recommended as the provisional model for further development.

The model was:

- Saved in Keras format.
- Independently loaded and validated.
- Tested using a production-style preprocessing pipeline.
- Accompanied by a production configuration file.

### Proposed Deployment Architecture

```text
User / Farmer
      |
      v
Streamlit Web Application
      |
      v
Python Image Preprocessing
      |
      v
MobileNetV2 – 320 × 320
      |
      v
Disease Classification
      |
      v
Prediction + Confidence Score
```

A user-facing application and independent real-world validation remain future development activities.

## 📁 Project Structure

```text
Rice-Leaf-Disease-Classification/
│
├── notebooks/
│   └── Rice_Leaf_Disease_Classification.ipynb
│
├── models/
│   ├── mobilenetv2_320_best.keras
│   └── production_config.json
│
├── outputs/
│   ├── dataset_class_distribution.png
│   ├── sample_rice_leaf_images.png
│   ├── image_dimension_distribution.png
│   ├── image_resolution_distribution.png
│   ├── aspect_ratio_distribution.png
│   ├── image_file_size_distribution.png
│   ├── image_preprocessing_visualization.png
│   ├── data_augmentation_visualization.png
│   ├── model_accuracy_comparison.png
│   ├── training_validation_curves.png
│   ├── confusion_matrix_mobilenet320.png
│   ├── classification_report_mobilenet320.png
│   ├── gradcam_analysis.png
│   ├── misclassified_images.png
│   └── padding_sensitivity_analysis.png
│
├── reports/
│
├── README.md
├── requirements.txt
└── .gitignore
```

*The visualisation filenames above are suggested names. Save the corresponding notebook plots into the `outputs/` folder using these names.*

## ⚙️ Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Rice-Leaf-Disease-Classification.git
```

### 2. Navigate to the project

```bash
cd Rice-Leaf-Disease-Classification
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the Jupyter Notebook

```bash
jupyter notebook
```

Open the Rice Leaf Disease Classification notebook and execute the cells in order.

### Requirements

Example dependencies:

```text
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
jupyter
```

Use the actual package versions from your working environment for reproducibility.

## ⚠️ Limitations

- The dataset contains only 119 images.
- The test set contains only 18 images.
- The test set was used during model comparison and subsequent investigations; therefore, the reported results are exploratory rather than independent final validation.
- Brown Spot classification requires improvement.
- Background and preprocessing sensitivity was observed.
- Real-world performance has not yet been established.

## 🔮 Future Enhancements

- Expand the dataset with more diverse rice leaf images.
- Explore reliable leaf segmentation techniques.
- Fine-tune pretrained architectures.
- Investigate EfficientNet and other transfer learning models.
- Improve Brown Spot classification.
- Perform independent external validation.
- Develop a Streamlit-based prediction application.
- Evaluate the solution with agricultural experts.

## 🎓 Key Learnings

This project provided practical experience in:

- Exploratory Data Analysis.
- Image processing and preprocessing.
- CNN architecture development.
- Data augmentation.
- Transfer learning.
- Model evaluation and error analysis.
- Explainable AI using Grad-CAM.
- Background sensitivity investigation.
- Model saving and inference validation.

## 👩‍💻 Author

**Brindha Jayaraj**

AI / Machine Learning Enthusiast

This project was developed as part of the DataMites Capstone Project.

---

⭐ If you find this project useful, feel free to explore the repository and share your feedback.

**Disclaimer:** This project is an educational machine learning prototype. Its predictions are not a substitute for agricultural expert assessment or laboratory diagnosis.

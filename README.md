# Retinal OCT Image Classification with VGG16

## **Overview**
This project focuses on using a pre-trained VGG16 model to classify retinal OCT (Optical Coherence Tomography) images into four categories:

- **CNV** (Choroidal Neovascularization)
- **DME** (Diabetic Macular Edema)
- **Drusen**
- **Normal**

By fine-tuning the VGG16 model and training it on the **Kermany OCT 2018 dataset**, we achieve high accuracy in retinal image classification. This approach is crucial for assisting ophthalmologists in diagnosing retinal diseases effectively.

---

## **Dataset**

- **Source**: [Kaggle OCT 2018 Dataset]
- **Classes**: CNV, DME, Drusen, Normal
- **Training Set**: 34,464 images
- **Validation Set**: 3,223 images

---

## **Workflow**

### **1. Data Augmentation**
To enhance model generalization, the following augmentations were applied:
- Rescaling
- Rotation
- Width and Height Shifting
- Horizontal Flipping

### **2. Model Architecture**

**Base Model**: VGG16 pre-trained on ImageNet. The top fully connected (FC) layers were removed, and new layers were added:

1. **Flatten Layer**: Flattens feature maps.
2. **Dense Layer**: Fully connected layer with 256 neurons.
3. **Dropout Layer**: Regularization to prevent overfitting.
4. **Output Layer**: Softmax activation with four neurons for classification.

### **3. Training Configuration**
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 16
- **Epochs**: 10
- **Callbacks**:
  - Early Stopping
  - Model Checkpoint

---

## **Results**

### **Performance Metrics**
- **Validation Accuracy**: 87.22%
- **Validation Loss**: 0.374


### Classification Report
| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| CNV        | 0.79      | 0.91   | 0.85     |
| DME        | 0.95      | 0.76   | 0.85     |
| Drusen     | 0.87      | 0.92   | 0.89     |
| Normal     | 0.84      | 0.93   | 0.88     |

---

## **Technologies Used**
- **Python**: Core programming language.
- **TensorFlow/Keras**: For deep learning model development.
- **Matplotlib**: For visualizations.
- **NumPy**: For numerical operations.

---

## **How to Run the Project**

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy pandas tensorflow keras matplotlib
  ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/retinal-vgg16-classifier.git
   cd retinal-vgg16-classifier
   ```
2. Download the dataset from Kaggle and place it in the project directory.
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Test on new images:
   ```bash
   python test_model.py
   ```

---

## **Future Work**
- Explore more advanced architectures like EfficientNet.
- Implement explainable AI techniques for better model interpretability.
- Deploy the model in a web or mobile application for real-time inference.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.

---

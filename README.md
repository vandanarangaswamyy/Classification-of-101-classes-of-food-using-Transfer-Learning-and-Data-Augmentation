# 🍽️ Classification of 101 Food Classes using Transfer Learning and Data Augmentation

This project demonstrates how **Transfer Learning** and **Data Augmentation** significantly improve the performance of deep learning models in classifying images from the **Food-101 dataset**. The model is trained to accurately recognize food across 101 distinct categories by leveraging pre-trained CNN architectures and augmentation strategies.

## 📌 What the Project Does

- Classifies food images into one of 101 classes from the **Food-101** dataset.
- Builds multiple models trained on different subsets of data (1%, 10%, and 100%).
- Uses **Transfer Learning** (e.g., ResNet50, InceptionV3) for faster convergence and better accuracy.
- Applies **Data Augmentation** to enhance generalization and prevent overfitting.
- Compares models trained with and without transfer learning and augmentation.
- Achieves improved accuracy over traditional approaches (~50.76%).

## 🔍 Key Concepts Explored

- Convolutional Neural Networks (CNNs)
- Transfer Learning with pre-trained models
- Image preprocessing and augmentation (rotation, flipping, cropping)
- Fine-tuning layers and freezing weights
- Performance visualization (loss/accuracy plots)
- Evaluation using confusion matrix and accuracy scores

## 📁 Dataset

- **Food-101** dataset by Lukas Bossard et al.
- Training examples per class: 750
- Testing examples per class: 250
- Subsets used: 1%, 10%, 100% for both 10-class and 101-class variations

## 🧪 Model Variants Trained

| Dataset Name                 | Classes     | Training Images/Class | Test Images/Class |
|-----------------------------|-------------|------------------------|-------------------|
| `10_food_classes_1_percent` | 10          | 7                      | 250               |
| `10_food_classes_10_percent`| 10          | 75                     | 250               |
| `10_food_classes_100_percent`| 10         | 750                    | 250               |
| `101_food_classes_10_percent`| 101        | 75                     | 250               |
| `101_food_classes_100_percent`| 101       | 750                    | 250               |

## 🧠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- Matplotlib, NumPy, Scikit-learn
- Jupyter Notebook

## 🛠 How to Run

1. Clone this repository.
2. Download and extract the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
3. Open `CLASSIFICATION OF 101 CLASSES OF FOOD USING TRANSFER LEARNING AND DATA AUGMENTATION.ipynb` in Jupyter Notebook.
4. Run the cells in order to train and evaluate the models.
5. Track performance using TensorBoard (if enabled in notebook).

## 📈 Sample Results

- Baseline accuracy (from literature): **50.76%**
- Improved accuracy (with Transfer Learning + Augmentation): *Update this with your best result*

## 📄 Files in Repository

- `CLASSIFICATION OF 101 CLASSES...ipynb`: Main notebook containing all code and experiments.
- `CLASSIFICATION OF 101 CLASSES...docx`: Full project report including literature survey, architecture, and findings.
- `README.md`: This file!

## 📚 References

- [Food-101 Dataset - ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- Bossard et al., *Food-101: Mining Discriminative Components with Random Forests*, ECCV 2014
- InceptionV3, ResNet, MobileNet - TensorFlow/Keras model zoo


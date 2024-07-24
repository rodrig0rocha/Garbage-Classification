# Overview
This project addresses the critical challenge of waste management and recycling by utilizing deep learning techniques to automate the sorting of household garbage. Manual sorting is labor-intensive, time-consuming, and prone to errors that can contaminate recyclable materials, compromising their usability and overall recycling effectiveness. I leverage advanced artificial intelligence to accurately classify and segregate different types of waste, significantly improving sorting efficiency and reducing the reliance on manual labor. By enhancing the accuracy of waste classification, this approach not only streamlines the recycling process but also supports environmental sustainability by ensuring a higher percentage of materials are correctly recycled and diverted from landfills.

# Dataset 
Source: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
Description: The dataset contains images of various types of waste, organized into 12 categories. These categories include battery, biological, brown-glass, cardboard, clothes, green-glass, metal, plastic, paper, shoes, trash and white-glass, which are crucial for effective waste sorting and recycling.

# Models
This project utilized a range of models to classify images of garbage into 12 distinct categories. Both pre-trained models and custom Convolutional Neural Networks (CNNs) were employed to achieve robust performance.
### Pre-Trained Models
Several pre-trained models were explored to leverage existing deep learning architectures and fine-tune them for our specific classification task:

- ResNet-18 and ResNet-34: Residual networks designed to enable the training of very deep models by using skip connections to address the vanishing gradient problem.
- VGGNet-16 and VGGNet-19: Convolutional networks known for their depth and ability to capture detailed features through small convolutional filters.
- DenseNet-121 and DenseNet-169: Dense convolutional networks that improve feature propagation and reuse through dense connections between layers.

### Custom CNN Models 
Custom Convolutional Neural Networks (CNNs) were built using the Keras library, with each model incorporating the following components:

#### Architecture
- Convolutional Layers: Multiple layers for feature extraction from images.
- Max-Pooling Layers: Layers to reduce spatial dimensions and retain important features.
- Flattening Layer: Converts 2D feature maps into a 1D vector for dense layers.
- Dense Layers: Fully connected layers that interpret features and make predictions.
- Output Layer: Comprises 12 neurons with a softmax activation function to classify the images into 12 categories.
  
#### Model Training
- Batch Size: Different batch sizes were tested to determine the optimal configuration
- Data Augmentation: Applied techniques such as rotation, scaling, and flipping to enhance model robustness and address class imbalance.
- Class Weights: Applied class weights to handle class imbalance, providing a way to adjust the model's sensitivity to underrepresented classes.
- Batch Normalization and Dropout: Incorporated in more complex models to stabilize training and reduce overfitting.

# Evaluation Metric
Given the severe class imbalance in the dataset, accuracy alone was not sufficient to reflect the model’s performance comprehensively. Therefore, I prioritized the following metric:

F1 Score: Chosen as a key evaluation metric for its ability to provide a balanced measure of precision and recall. The F1 score is particularly useful for handling imbalanced data, as it considers both false positives and false negatives. Since Keras does not natively support a class-wise F1 score, and the scikit-learn implementation has compatibility issues with Keras/TensorFlow, I implemented a custom F1 score calculation to address these challenges.

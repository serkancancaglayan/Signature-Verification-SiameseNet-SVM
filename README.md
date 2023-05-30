# Signature-Verification-SiameseNet-SVM


![Screenshot](WorkFlow.jpg)


Signature verification is an important task in various domains such as banking, legal documentation, and forensic analysis. Traditional methods for signature verification often rely on manual examination by experts, making the process time-consuming and subjective. In recent years, deep learning techniques have shown great potential in automating signature verification processes.

The aim of this project is to develop a robust signature verification system that can accurately determine whether two signatures belong to the same person or not. The system will utilize a combination of a Siamese Convolutional Neural Network (CNN) and a Support Vector Machine (SVM) classifier to achieve this task. By extracting image embeddings from input signature images using the CNN, followed by classification using the SVM, the system will provide a reliable means of verifying signatures.


# Methods

***Data Collection and Preprocessing***

-Collect a dataset of signature images from different individuals.

-Preprocess the dataset by removing noise, resizing images, and normalizing intensities.

-Divide the dataset into pairs of genuine signatures and pairs of forged signatures.



***Siamese Convolutional Neural Network***

-Design a Siamese CNN architecture that takes two signature images as input.

-Train the CNN to learn image embeddings that capture the unique characteristics of signatures.

-Implement triplet loss as the training objective to encourage the CNN to minimize the distance between embeddings of genuine signatures and maximize the distance between embeddings of forged signatures.



***Feature Extraction***

-Extract image embeddings from signature pairs using the trained Siamese CNN.

-Generate a feature vector representing the similarity or dissimilarity between the signatures.



***Support Vector Machine Classifier***

-Train an SVM classifier using the extracted feature vectors.

-Utilize a suitable kernel function (e.g., linear, radial basis function) to map the feature vectors to a higher-dimensional space.

-Optimize SVM hyperparameters through techniques such as grid search or cross-validation.



***Signature Verification***

-Given a pair of signature images, pass them through the Siamese CNN to obtain their embeddings.

-Use the SVM classifier to predict whether the signatures belong to the same person or not.


# TODO List
- [x] Implement Siamese Net 
- [x] Implement Data Loader and Contrastive Loss
- [x] Train Siamese Net on ICDAR dataset
- [ ] Implement Feature Extractor from Image Embeddings
- [ ] Train a SVM to Signature Verification

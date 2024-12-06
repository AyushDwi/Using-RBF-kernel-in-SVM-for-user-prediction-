# Using-RBF-kernel-in-SVM-for-user-prediction-
ML project to demonstrate the use of RBF kernel in SVM

This code implements a machine learning pipeline for user prediction using Support Vector Machine (SVM). Here's a step-by-step explanation of its key components:

1. Libraries and Modules
The code imports various essential libraries:

Numpy & Pandas: For data manipulation and numerical computations.
Matplotlib & Seaborn: For visualization (e.g., ROC curve, confusion matrix).
Sklearn: Includes preprocessing, SVM classifier, model evaluation metrics, and hyperparameter tuning tools.

2. Data Import and Preprocessing
Dataset: The dataset is loaded from a URL and stored in a DataFrame (df). It contains user data, including age, salary, and purchase decisions.
Independent Variables (X): Age and Estimated Salary are extracted as features for the model.
Dependent Variable (y): The "Purchased" column serves as the target.

3. Feature Scaling
StandardScaler: Scales X to have a mean of 0 and standard deviation of 1. SVMs are sensitive to feature scaling, making this step essential for better performance.

4. Polynomial Features
Degree-2 Polynomial Transformation: Polynomial features are added to capture interactions and non-linear relationships between features.

5. Stratified K-Fold Cross-Validation
StratifiedKFold: Splits the dataset into 10 folds while preserving the class distribution in each fold. This ensures balanced evaluation across classes.
Training and Testing: In each fold:
Model trains on k-1 folds and tests on the remaining fold.
True (y_true) and predicted (y_pred) values are stored for overall evaluation.

6. Model Training
SVM with RBF Kernel: The SVM classifier uses the Radial Basis Function (RBF) kernel to capture non-linear relationships.
Balanced Class Weights: Adjusts weights to handle imbalanced data, ensuring fair treatment of both classes.

7. Evaluation Metrics
Confusion Matrix: Displays the number of true positives, false positives, true negatives, and false negatives.
Accuracy: Calculates the proportion of correctly predicted instances.
ROC Curve and AUC: Plots the true positive rate against the false positive rate and computes the Area Under the Curve (AUC), a measure of model performance.

8. Hyperparameter Tuning
GridSearchCV: Performs a grid search over two hyperparameters:
C: Regularization parameter controlling the trade-off between margin size and misclassification.
gamma: Controls the influence of a single training sample in the RBF kernel.
Best Parameters: The optimal combination of C and gamma is identified.

9. New Prediction
Example Input: Predicts the purchasing likelihood for a user with age 40 and salary 50,000:
The input is scaled and transformed to match the model's preprocessing.
The model predicts whether the user is likely to purchase the product.
Visualization
Confusion Matrix: Heatmap to visualize performance on true/false positive/negative predictions.
ROC Curve: Illustrates the trade-off between sensitivity (true positive rate) and specificity (false positive rate).


# Credit Card Fraud Detection

## Project Description
This project aims to detect fraudulent credit card transactions using various machine learning models. The dataset used for this project is a simulated credit card transaction dataset containing legitimate and fraudulent transactions from January 1, 2019, to December 31, 2020. The dataset includes transactions from 1000 customers and 800 merchants, generated using the Sparkov Data Generation tool by Brandon Harris.

## Models
The following models are trained and evaluated in this project:
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Network

## Performance Metrics
Each model is evaluated using the following metrics:
- Precision
- Recall
- F1-Score
- Accuracy

### Test Results
#### Random Forest
              precision    recall  f1-score   support

           0       1.00      0.90      0.95    553574
           1       0.02      0.59      0.04      2145

    accuracy                           0.90    555719
    macro avg      0.51      0.75      0.50    555719
    weighted avg   0.99      0.90      0.95    555719


Accuracy: 0.9033540332434198

#### Gradient Boosting
              precision    recall  f1-score   support

           0       1.00      0.93      0.97    553574
           1       0.04      0.77      0.08      2145

    accuracy                           0.93    555719
    macro avg      0.52      0.85      0.52    555719
    weighted avg   1.00      0.93      0.96    555719

Accuracy: 0.9335941366050108

#### XGBoost
              precision    recall  f1-score   support

           0       1.00      0.93      0.96    553574
           1       0.04      0.76      0.08      2145

    accuracy                           0.93    555719
    macro avg      0.52      0.85      0.52    555719
    weighted avg   1.00      0.93      0.96    555719

Accuracy: 0.92918183470423


#### Neural Network
          precision    recall  f1-score   support

       0       1.00      1.00      1.00    553574
       1       0.00      0.00      0.00      2145

       accuracy                        1.00    555719
       macro avg 0.50 0.50 0.50 555719
       weighted avg 0.99 1.00 0.99 555719
       Accuracy: 0.9961401355721147


### Analysis
From the test results, it is evident that the models perform very well in predicting the majority class (non-fraudulent transactions) but struggle significantly with the minority class (fraudulent transactions). Here is a detailed analysis of each model's performance:

- **Random Forest**: Achieves high overall accuracy (90.33%) and performs reasonably well in detecting fraudulent transactions (recall of 59%). However, its precision for fraud detection is very low (2%), indicating many false positives.
  
- **Gradient Boosting**: Shows an improved performance over Random Forest with an accuracy of 93.36%. It also has a better recall (77%) for the fraudulent class but still suffers from low precision (4%).

- **XGBoost**: Similar to Gradient Boosting, XGBoost achieves a high overall accuracy (92.92%) and high recall (76%) for the fraudulent class but low precision (4%).

- **Neural Network**: Achieves the highest overall accuracy (99.61%) but fails to detect any fraudulent transactions, with precision, recall, and F1-score all at 0 for the fraudulent class. This indicates the model is not useful for fraud detection in this context.

### Conclusion
The models show a significant imbalance in detecting fraudulent transactions versus non-fraudulent transactions. This is a common challenge in fraud detection due to the imbalance in the dataset. The results suggest that further tuning and potentially more sophisticated techniques are needed to improve fraud detection performance. Techniques such as advanced sampling methods, anomaly detection, or ensemble methods combining multiple algorithms might help improve the results.

Acknowledgements

    Brandon Harris for creating the Sparkov Data Generation tool, which was used to generate the simulated credit card transaction dataset.
    The dataset used in this project was sourced from Kaggle https://www.kaggle.com/datasets/kartik2112/fraud-detection/data

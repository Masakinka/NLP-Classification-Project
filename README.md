## NLP Classification Project: Sentiment Analysis with Machine Learning

### Overview
This project focuses on building a machine learning model to classify the sentiment of text data. Using various machine learning algorithms, the goal is to predict whether a text expresses positive, negative, or neutral sentiment. The dataset used consists of labeled text data from multiple sources, providing a diverse set of examples for model training and evaluation.

### Dataset Source
- **Source**: Kaggle Dataset https://www.kaggle.com/competitions/tweet-sentiment-extraction/data?select=train.csv
- **Description**: The dataset contains text entries labeled with sentiment categories such as positive, negative, and neutral. The project applies Natural Language Processing (NLP) techniques to clean and prepare the data for classification tasks.

### Key Features:
- **Text Preprocessing**: Tokenization, stopword removal, lemmatization, and vectorization (e.g., TF-IDF) were applied to prepare text data for model input.
- **Custom Transformers**: Feature engineering involved creating custom transformers to enhance the dataset, such as calculating word frequency and extracting key linguistic features.

### Modeling
Several machine learning algorithms were trained and compared:
1. **Logistic Regression**
2. **Random Forest**
3. **Support Vector Machine (SVM)**
4. **XGBoost**

### Model Performance
Each model's performance was evaluated using precision, recall, and F1 score:
- **Logistic Regression**: Achieved high recall but lower precision in detecting neutral sentiment.
- **Random Forest**: Balanced performance across all classes.
- **XGBoost**: Showcased the highest overall F1 score with improved precision and recall for both positive and negative sentiments.

### Feature Importance
The Random Forest and XGBoost models were analyzed to determine which words most significantly influenced sentiment prediction:
- **Positive Words**: 'love', 'great', 'happy'
- **Negative Words**: 'sad', 'hate', 'bad'
- **Neutral Words**: 'the', 'is', 'and'

### Hyperparameter Tuning
For XGBoost, hyperparameter tuning was performed using **Randomized Search** and **Grid Search** to optimize model performance:
- **Best F1 Score**: 0.92 after tuning.

### Error Analysis
An error analysis was conducted to understand common misclassifications:
- **False Positives**: Texts classified as positive but expressed neutral sentiment.
- **False Negatives**: Texts classified as neutral but contained subtle negative sentiment.

### Conclusion
The XGBoost model performed best, achieving a balance between precision and recall. Future improvements include refining the feature engineering process and experimenting with additional boosting algorithms like CatBoost or LightGBM.

### Future Work
- **Feature Engineering**: Explore advanced NLP techniques such as word embeddings (Word2Vec) and sentence embeddings (BERT).
- **Modeling**: Implement deep learning models like LSTM or Transformer-based models to enhance performance on complex sentiment analysis tasks.
- **Data Augmentation**: Address class imbalance with SMOTE or oversampling techniques.

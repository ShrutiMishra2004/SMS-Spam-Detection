📩 SMS Spam Classifier using NLP and Machine Learning
This project presents a robust end-to-end solution for detecting spam SMS messages using Natural Language Processing (NLP) and Machine Learning (ML). It involves data cleaning, exploratory data analysis, text preprocessing, feature extraction using TF-IDF, and building multiple classification models. The final model achieves high accuracy and precision, making it ideal for spam filtering applications.

📁 Dataset Overview
Source: UCI SMS Spam Collection Dataset

Size: 5,572 messages

Columns:

v1: Label (ham or spam)

v2: Message text

Preprocessing:

Dropped 3 unnamed irrelevant columns

Renamed columns to Target and Text

Converted labels to binary (ham → 0, spam → 1)

🔍 Project Workflow
📌 1. Data Cleaning
Removed nulls and duplicates

Encoded labels

Cleaned and structured the dataset for analysis

📊 2. Exploratory Data Analysis (EDA)
Checked class imbalance (spam ≈ 11.7%, ham ≈ 88.3%)

Analyzed:

Message length

Word count

Sentence count

Visualized:

Distribution plots

Pair plots

Word clouds (for spam and ham)

Most frequent words

🧹 3. Text Preprocessing
Custom function transform_text() performs:

Lowercasing

Tokenization

Removing stopwords & punctuation

Stemming using PorterStemmer

🧠 4. Feature Extraction
Used TfidfVectorizer to convert text into numerical feature vectors

Extracted top 3,000 features

🤖 5. Model Building
Trained and evaluated 3 classifiers:

Classifier	Accuracy	Precision
Multinomial Naive Bayes	97.1%	100%
Bernoulli Naive Bayes	98.3%	99.18%
Gaussian Naive Bayes	86.9%	50.68%

🔥 MultinomialNB outperformed others and was selected as the final model.

💾 6. Model Deployment
Saved the model and vectorizer using pickle

model.pkl

vectorizer.pkl

📈 Visual Insights
📊 Bar charts for most common words

☁️ Word clouds for spam vs. ham

🔥 Heatmap showing feature correlations

📉 Distribution plots for message metrics

🛠️ Technologies & Libraries
Python

Pandas, NumPy

Matplotlib, Seaborn

NLTK

Scikit-learn

WordCloud

Pickle

🚀 How to Run
Install dependencies:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Train & Save the Model:
The notebook saves:

Trained model → model.pkl

TF-IDF vectorizer → vectorizer.pkl

(Optional): Build a front-end using Flask or Streamlit for deployment.

🔮 Possible Improvements
Address class imbalance using SMOTE or class weighting

Hyperparameter tuning with GridSearchCV

Integrate Deep Learning models (LSTM, BERT)

Deploy with a web UI and live SMS input

✅ Conclusion
This project showcases how traditional NLP and machine learning techniques can accurately detect spam messages. With high accuracy and precision, it can be used in real-world SMS spam detection systems.

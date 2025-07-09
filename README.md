📩 SMS Spam Classifier
A machine learning project to detect spam messages using NLP techniques and classification algorithms. This project processes SMS data, performs thorough exploratory data analysis, text preprocessing, vectorization, and finally builds and evaluates multiple models to classify messages as spam or ham (not spam).

📂 Dataset
The dataset contains 5,572 SMS messages.

Labeled as:

ham = not spam

spam = unsolicited message

Original columns: v1 (label), v2 (text), plus 3 unnamed/irrelevant columns (dropped during cleaning).

🧹 Steps Followed
1. Data Cleaning
Removed irrelevant columns.

Renamed columns for clarity: v1 → Target, v2 → Text.

Encoded labels using LabelEncoder: ham → 0, spam → 1.

Removed duplicate entries.

2. Exploratory Data Analysis (EDA)
Visualized class distribution (imbalanced).

Extracted:

Character count

Word count

Sentence count

Generated histograms and pairplots to explore message characteristics by class.

3. Text Preprocessing
Lowercased text

Tokenization

Removed stopwords and punctuation

Applied stemming (PorterStemmer)

Created a transform_text() function to automate preprocessing

4. Feature Engineering
Vectorized text using TF-IDF (TfidfVectorizer) with top 3,000 features.

5. Model Building
Trained three classifiers:

Multinomial Naive Bayes ✅ (Best performing)

Bernoulli Naive Bayes

Gaussian Naive Bayes

Evaluated using:

Accuracy

Confusion Matrix

Precision Score

Model	Accuracy	Precision
MultinomialNB	97.09%	100%
BernoulliNB	98.36%	99.18%
GaussianNB	86.94%	50.68%

6. Model Saving
Final model (MultinomialNB) and vectorizer saved using pickle:

model.pkl

vectorizer.pkl

📊 Visualizations
Word clouds for spam and ham messages

Bar plots of most frequent words

Heatmap of feature correlation

🛠️ Requirements
Install dependencies with:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud
🧠 Future Improvements
Handle class imbalance (e.g., using SMOTE)

Hyperparameter tuning

Try deep learning approaches (LSTM, BERT)

Build a web UI using Flask or Streamlit

✅ Conclusion
This project demonstrates the effectiveness of traditional NLP and machine learning techniques in detecting spam messages with high precision using Multinomial Naive Bayes.


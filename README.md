# Quora Question Similarity – Machine Learning Approach

This project was developed as part of the **NLP & Text Mining** course in my **MSc in Data Science**.  
The goal is to detect whether two questions from Quora ask the **same thing** (duplicate detection) using
classical **NLP + Machine Learning** techniques.

---

## 🧩 Problem Statement

Given pairs of questions:

- **question1**
- **question2**
- **is_duplicate** ∈ {0, 1}

we want to build a model that predicts whether the pair is a **duplicate** (same intent) or **not**.

Example:

- Q1: *"How do you start a bakery?"*  
- Q2: *"How can one start a bakery business?"*  

These should be classified as **duplicates**.
However, the following pair of questions should be classified as **not duplicate**:

- Q1: *"What is the step by step guide to invest in share market in india?"*  
- Q2: *"What is the step by step guide to invest in share market?"*  

---

## 📦 Dataset

The dataset is the well-known **Quora Question Pairs** dataset.  
It contains a CSV file with the following columns:

- `id` – pair ID  
- `qid1`, `qid2` – IDs of the individual questions  
- `question1`, `question2` – the text of each question  
- `is_duplicate` – 1 if the questions are duplicates, 0 otherwise  

> 🔒 **Note:** The dataset is not included in this repository due to size and licensing.  
> You can download it from Kaggle and place it under your own `data/` folder.

---

## 🛠️ Methods & Workflow

The full workflow is implemented in the notebook:

- `quora-machine-learning-approach.ipynb`

The pipeline consists of the following steps:

### 1. Exploratory Data Analysis (EDA)

- Check for:
  - Missing values and duplicate rows  
  - Class distribution of `is_duplicate`  
  - Question frequency: how many times each `qid` appears  
- Analyze typical question patterns:
  - Questions with/without `?`
  - Multi-part questions (multiple `?`)
  - Presence of `[math]` tags
  - Opinion/personal questions containing “you” or “I”

---

### 2. Text Preprocessing

For each question, a custom preprocessing function is applied:

- Lowercasing and stripping whitespace
- Normalizing symbols and currencies:
  - `%` → `percent`, `$` / `€` / `₹` → `dollar` / `euro` / `rupee`, etc.
- Removing/remapping special tags (e.g. `[math]`)
- Normalizing numbers and magnitudes:
  - `1000` → `1k`, `1,000,000` → `1m`, etc.
- Expanding contractions:
  - `don't` → `do not`, `you're` → `you are`, etc.
- Removing emojis
- Tokenization with NLTK
- Stopword removal (English stopwords)
- Lemmatization using `WordNetLemmatizer`

The result is a clean, lemmatized representation of each question, focused on content words.

---

### 3. Handcrafted Feature Engineering

On top of the cleaned questions, several **interpretable features** are constructed:

#### Frequency & length features

- `freq_qid1`, `freq_qid2`: how often each question appears in the dataset
- `q1len`, `q2len`: character length of each question
- `q1_n_words`, `q2_n_words`: number of words in each question
- `word_Common`: count of common unique words between `question1` and `question2`
- `word_Total`: total number of unique words across both questions
- `word_share`: ratio of overlap: `word_Common / (q1_n_words + q2_n_words)`
- `freq_q1+q2`, `freq_q1-q2`: sum and absolute difference of question frequencies

#### Token-level similarity features

Using token sets with and without stopwords, the following features are computed:

- `cwc_min`, `cwc_max`: common **non-stopword** ratios (min and max over both questions)
- `csc_min`, `csc_max`: common **stopword** ratios
- `ctc_min`, `ctc_max`: common **token** ratios overall
- `first_word_eq`: 1 if first tokens are equal, else 0
- `last_word_eq`: 1 if last tokens are equal, else 0
- `abs_len_diff`: absolute difference in token counts
- `mean_len`: mean token length across both questions

These features are visualized and inspected with pairplots and distributions split by `is_duplicate`.

---

### 4. Machine Learning on Handcrafted Features

#### Models used

- **Random Forest**
- **XGBoost**
- **Logistic Regression** (with and without feature scaling)

#### Evaluation

- Dataset split into **train/test** (e.g. 80/20)
- Metrics:
  - **Log-loss**
  - **Accuracy**
  - **Confusion matrix** (visualized with Seaborn heatmaps)
- For XGBoost:
  - Hyperparameter tuning with **GridSearchCV** (learning rate, max depth, n_estimators)
  - Cross-validation using negative log-loss as the score

---

### 5. TF-IDF + n-grams + Chi-square Feature Selection

In addition to handcrafted features, the project also explores classic vector-space models:

1. **TF-IDF Unigrams**
   - Build TF-IDF vectors for all questions (`question1` and `question2`)
   - Concatenate TF-IDF representations of the two questions
   - Apply **chi-square feature selection** to keep only the most important terms

2. **TF-IDF Bigrams and Trigrams**
   - Use `ngram_range=(1, 2)` and `ngram_range=(1, 3)` for more expressive patterns
   - Again apply chi-square selection
   - Inspect the top n-grams contributing to the classification

3. **Models on TF-IDF space**
   - **Random Forest**
   - **Logistic Regression**
   - **XGBoost**

For each configuration, log-loss, accuracy, and confusion matrices are computed and compared.

---

## 📊 Results 

- Best model (**XGBoost on handcrafted features**):
  - Accuracy: **81%**
  - Log-loss: **0.373**

---

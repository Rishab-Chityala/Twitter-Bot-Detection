# Twitter Bot Detection System

A machine learning web application that detects automated bot accounts on Twitter. Upload a CSV of Twitter accounts and instantly receive bot probability scores, visual analytics, and downloadable results.

## Results
- Validation Accuracy: 91.4%
- ROC-AUC: 96.7%
- Bot Precision: 95%
- Processes 500+ accounts in under 3 seconds

## Tech Stack
Python · Scikit-learn · Streamlit · Pandas · TF-IDF · Random Forest

---

## Project Structure

```
twitter-bot-detection/
├── twitter_bot.py                  ← model training script
├── app.py                          ← streamlit web app
├── requirements.txt                ← dependencies
├── training_data_2_csv_UTF.csv     ← training dataset
└── test_data_4_students.csv        ← test dataset
```

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/twitter-bot-detection.git
cd twitter-bot-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
Run this once to generate the model files:
```bash
python twitter_bot.py
```
This creates `rf_model.pkl` and `tfidf.pkl` in your project folder.

### 4. Run the app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## How to Use

1. Run the app using the command above
2. Upload a CSV file containing Twitter account data
3. Click **Analyze for Bots**
4. View results — bot probability scores, summary metrics, and account breakdown
5. Download results as CSV

### Required CSV Columns
| Column | Description |
|---|---|
| `screen_name` | Twitter username |
| `description` | Bio text |
| `followers_count` | Number of followers |
| `friends_count` | Number of following |
| `statuses_count` | Total tweets posted |
| `verified` | Verified status |
| `default_profile_image` | Default profile picture |

---

## Features

- 69 engineered features including follower/following ratio, posting frequency, account age, and username patterns
- TF-IDF vectorization on bio text to extract linguistic bot patterns
- Random Forest classifier trained on 2,797 labeled accounts
- Interactive Streamlit dashboard with filtering by verdict
- Downloadable results as CSV

---

## Model Performance

| Metric | Value |
|---|---|
| Training Accuracy | 98.3% |
| Validation Accuracy | 91.4% |
| Mean ROC-AUC (5-fold) | 96.7% |
| Bot Precision | 95% |
| Bot Recall | 86% |

---

## Dataset

The model is trained on a labeled dataset of 2,797 Twitter accounts:
- 1,476 genuine human accounts
- 1,321 confirmed bot accounts

---

## License
MIT

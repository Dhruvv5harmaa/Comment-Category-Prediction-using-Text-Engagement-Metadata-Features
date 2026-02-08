# 📝 Comment Category Prediction  
### Text, Engagement & Metadata-based Classification

---

## 📌 Problem Overview
Online platforms receive millions of user-generated comments that must be accurately categorized for moderation and internal processing.

The objective of this project is to **predict the final category assigned to a comment** using:
- textual content,
- user engagement signals (upvotes/downvotes),
- and system-generated metadata.

This is a **multi-class classification problem** with **four target categories**.

---

## 📊 Dataset Summary
| Split | Records |
|------|--------|
| Training | ~198,000 |
| Test | ~102,000 |

**Target Variable:** `label` (4 classes)

### Feature Groups
- **Textual:** Comment text  
- **Engagement:** Upvotes, downvotes  
- **Metadata:** Emoticon indicators, internal system flags  
- **Temporal:** Comment creation timestamp  
- **Topic Indicators:** Race, religion, gender, disability references  

---

## 🧹 Data Preparation & Feature Engineering
Key preprocessing and feature engineering steps:

- Handled missing and invalid values using robust imputation strategies  
- Extracted **temporal features**:
  - Hour of day
  - Day of week
  - Month
  - Weekend indicator  
- Created **engagement-based features**:
  - Score = upvotes − downvotes  
  - Total votes  
  - Upvote ratio  
- Converted text into numerical features using **TF-IDF vectorization**  
- Encoded categorical features using **One-Hot Encoding**

All preprocessing was implemented using **scikit-learn Pipelines** and `ColumnTransformer` for reproducibility.

---

## 🤖 Modeling Strategy
- Formulated as a **multi-class classification** task  
- Built an end-to-end ML pipeline combining:
  - TF-IDF text features
  - Numerical features
  - Categorical features  
- Trained a **Logistic Regression** classifier as a strong baseline  
- Used a **stratified train–validation split** to handle class imbalance  

---

## 📈 Model Evaluation
Evaluation metrics used:
- **Accuracy**
- **Macro F1-score** (to ensure balanced performance across all classes)

The model achieved **strong macro F1 performance** on the validation set, indicating reliable classification across categories.

---

## 🧠 Key Insights
- Combining **textual features with engagement signals** significantly improves prediction quality  
- Feature engineering on time-based and vote-based attributes adds meaningful predictive power  
- ML pipelines help maintain clean, scalable, and production-ready workflows  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, scikit-learn  
- **Text Processing:** TF-IDF  
- **Environment:** Jupyter Notebook  

---

## 📁 Repository Structure

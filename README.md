# Comment-Category-Prediction-using-Text-Engagement-Metadata-Features
Comment Category Prediction using Text, Engagement & Metadata
Problem Statement

Online platforms receive a large volume of user-generated comments that must be categorized correctly for moderation and internal handling.
The goal of this project is to predict the final category assigned to a comment using a combination of:

textual content,

user engagement signals,

and system-generated metadata.

This is a multi-class classification problem with four possible output categories.

Dataset Overview

Training data: ~198,000 comments

Test data: ~102,000 comments

Target variable: label (4 classes)

Key Features

Textual: raw comment text

Engagement: upvotes, downvotes

Metadata: emoticon indicators, internal system flags

Temporal: comment creation time

Topic indicators: race, religion, gender, disability references

Data Preprocessing & Feature Engineering

The following steps were applied to prepare the data for modeling:

Handled missing and invalid values using appropriate imputation strategies

Converted timestamps into useful temporal features:

hour of day

day of week

month

weekend indicator

Engineered engagement-based features:

score (upvotes − downvotes)

total votes

upvote ratio

Processed text data using TF-IDF vectorization

Encoded categorical variables using One-Hot Encoding

All preprocessing steps were implemented using scikit-learn pipelines to ensure reproducibility.

Modeling Approach

Framed the task as a multi-class classification problem

Built a unified pipeline using ColumnTransformer to handle:

text features

numerical features

categorical features

Trained a Logistic Regression classifier as a strong baseline model

Used stratified train–validation split to handle class imbalance

Evaluation

Evaluation metrics:

Accuracy

Macro F1-score (to account for class imbalance)

The model achieved strong macro F1 performance on the validation set, demonstrating balanced prediction quality across all categories.

Key Learnings

Combining textual data with structured engagement signals significantly improves classification performance

Feature engineering on time and vote-related attributes adds meaningful predictive power

End-to-end ML pipelines help maintain clean, scalable, and production-ready workflows

Tech Stack

Python

Pandas, NumPy

scikit-learn

TF-IDF Vectorization

Jupyter Notebook

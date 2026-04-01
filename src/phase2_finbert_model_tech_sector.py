# PHASE 2
# Extract rows whose tickers appear in S5INFT.xlsx
import pandas as pd
import os
os.chdir(r"C:\Users\Khanh Hoang\Desktop\Pe_alice")

# 1️⃣ Load the NASDAQ dataset
df_nasdaq = pd.read_csv("2018_nasdaq_cleaned.csv", parse_dates=["Date"])

# 2️⃣ Load the “technology” sheet from S5INFT.xlsx
df_tech = pd.read_excel("S5INFT.xlsx", sheet_name="technology")

# 3️⃣ Build a set of valid tickers
valid_tickers = set(df_tech["Ticker"].astype(str))

# 4️⃣ Filter rows whose Stock_symbol is in the valid_tickers set
df_phase2 = df_nasdaq[df_nasdaq["Stock_symbol"].isin(valid_tickers)].copy()

# 5️⃣ Save the filtered DataFrame
output_path = "2018_nasdaq_cleaned_phase2.csv"
df_phase2.to_csv(output_path, index=False)

# 6️⃣ Print the number of rows
print(f"Filtered dataset saved to {output_path}")
print(f"Number of rows after filtering: {len(df_phase2)}")

# --------------------------------------------------
# Calculate price change for this dataset
import pandas as pd

# 1️⃣ Read data and parse Date
df = pd.read_excel("S5INFT.xlsx", sheet_name="index price", parse_dates=["date"])
print(df.head())

# 2️⃣ Remove rows without price information
df = df[df["last_price"].notna()].copy()

# 3️⃣ Sort by date in ascending order
df = df.sort_values("date").reset_index(drop=True)

# 4️⃣ Shift to get the previous trading day's price
df["prev_price"] = df["last_price"].shift(1)

# 5️⃣ Compute price_change (percentage change compared to previous price)
df["price_change"] = (df["last_price"] - df["prev_price"]) / df["prev_price"]

# 6️⃣ Drop first row (no prev_price)
df = df.dropna(subset=["price_change"]).reset_index(drop=True)

# 7️⃣ Keep three columns and export file
df_phase2 = df[["date", "last_price", "price_change"]]
df_phase2.to_csv("index_price_phrase2.csv", index=False)

# --------------------------------------------------
# Merge the two datasets for the technology industry
import os
os.chdir(r"C:\Users\Khanh Hoang\Desktop\Pe_alice")
import pandas as pd

# Load datasets
df_articles = pd.read_csv('2018_nasdaq_cleaned_phase2.csv', parse_dates=['Date'])
df_index_price = pd.read_csv('index_price_phrase2.csv', parse_dates=['date'])

df_index_price.rename(columns={"date": "Date", "price_change": "Change"}, inplace=True)

# Merge datasets based on 'Date'
df_merged = pd.merge(df_articles, df_index_price[['Date', 'Change']], on='Date', how='left')

# Replace missing 'Change' values with 0
df_merged['Change'].fillna(0, inplace=True)

# Save merged dataset
df_merged.to_csv('tech_merged_phrase2.csv', index=False)

# Display first rows and number of rows
print(df_merged.head())
print(f"Total rows in merged dataset: {len(df_merged)}")

# --------------------------------------------------
# CALCULATE SENTIMENT SCORE FOR TECH INDUSTRY
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Force model to use CPU if GPU is unavailable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FinBERT model and tokenizer
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()  # Set model to evaluation mode

# Function to get sentiment and compute sentiment score
def get_sentiment(text):
    """Returns a sentiment label and confidence score for a given text."""
    if not isinstance(text, str) or text.strip() == "":
        # Default to neutral if text is missing or empty
        return pd.Series(["Neutral", 0.0])

    # Tokenization and model prediction
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)

    # Compute softmax probabilities
    probs = softmax(outputs.logits, dim=1)

    # Get sentiment label
    labels = ["Negative", "Neutral", "Positive"]
    sentiment_index = torch.argmax(probs).item()

    sentiment = labels[sentiment_index]
    confidence_score = probs[0][sentiment_index].item()

    return pd.Series([sentiment, confidence_score])

# Define file paths
new_csv_output_path_2018 = 'tech_merged_phrase2.csv'
sentiment_output_path = r"C:\Users\Khanh Hoang\Desktop\Pe_alice\technology_sentiment_analysis_2018.csv"

# Read CSV in chunks
chunk_iter_2018 = pd.read_csv(new_csv_output_path_2018, chunksize=10000)
count_2018 = 0

# Open CSV for writing processed chunks
for chunk_2018 in chunk_iter_2018:
    print(f"Processing chunk {count_2018 + 1}: {chunk_2018.shape}")

    # Apply sentiment analysis and store sentiment label and confidence score
    chunk_2018[["Sentiment", "Confidence_Score"]] = chunk_2018["Article_title"].apply(get_sentiment)

    # Save results to CSV efficiently
    mode = "w" if count_2018 == 0 else "a"
    header = count_2018 == 0  # Write header only for the first chunk

    chunk_2018.to_csv(sentiment_output_path, index=False, mode=mode, header=header)

    count_2018 += 1

print(f"Sentiment analysis completed. Results saved to {sentiment_output_path}")
print(f"Total chunks processed: {count_2018}")

import os
print("Current Working Directory:", os.getcwd())

# --------------------------------------------------
# Test the magnitude of error using an LSTM model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 0) (Optional) Check GPU availability and create distribution strategy
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
strategy = tf.distribute.MirroredStrategy()

# 1) Load & preprocess
a1_file_path = "technology_sentiment_analysis_2018.csv"
df = pd.read_csv(
    a1_file_path,
    encoding='ISO-8859-1',
    parse_dates=['Date'],            # Parse Date column when reading
    usecols=['Date','Article_title','Sentiment','Confidence_Score','Change']
)

df = df.sort_values('Date').reset_index(drop=True)

# Map sentiment to numeric values
df['Sentiment_Num'] = df['Sentiment'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0})

# 2) Prepare input X and target y
# — text → sequences → padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Article_title'])
seqs = tokenizer.texts_to_sequences(df['Article_title'])
text_pad = pad_sequences(seqs, maxlen=30, padding='post', truncating='post')

# — numeric features
sentiment_array  = df['Sentiment_Num'].to_numpy().reshape(-1,1)
confidence_array = df['Confidence_Score'].to_numpy().reshape(-1,1)

# — concatenate features
X = np.hstack((text_pad, sentiment_array, confidence_array))
y = df['Change'].to_numpy()

# 3) Train-test split (no shuffle to preserve temporal order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    shuffle=False
)

# 4) Build & train LSTM under GPU strategy
with strategy.scope():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64, input_length=30),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,             # GPU can typically handle larger batches
    validation_data=(X_test, y_test),
    verbose=1
)

# 5) Predict
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

# — Compute evaluation metrics
train_mae  = mean_absolute_error(y_train, y_train_pred)
train_mse  = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

test_mae   = mean_absolute_error(y_test, y_test_pred)
test_mse   = mean_squared_error(y_test, y_test_pred)
test_rmse  = np.sqrt(test_mse)

# — Print out the metrics
print("── Train Set Evaluation ──")
print(f"Train MAE : {train_mae:.6f}")
print(f"Train MSE : {train_mse:.6f}")
print(f"Train RMSE: {train_rmse:.6f}\n")

print("── Test Set Evaluation ──")
print(f"Test MAE  : {test_mae:.6f}")
print(f"Test MSE  : {test_mse:.6f}")
print(f"Test RMSE : {test_rmse:.6f}")

# 6) Combine into DataFrame indexed by Date
dates       = df['Date']
train_dates = dates.iloc[:len(y_train)]
test_dates  = dates.iloc[len(y_train):len(y_train) + len(y_test)]

t_train = pd.DataFrame({
    'actual':    y_train.flatten(),
    'predicted': y_train_pred.flatten()
}, index=train_dates)

t_test  = pd.DataFrame({
    'actual':    y_test.flatten(),
    'predicted': y_test_pred.flatten()
}, index=test_dates)

# 7) Aggregate by day (1 point per day)
t_train_daily = t_train.resample('D').mean().dropna()
t_test_daily  = t_test.resample('D').mean().dropna()

# 8) Plot chart
plt.figure(figsize=(15,6))
plt.plot(t_train_daily.index, t_train_daily['actual'],    label='Actual Train',    linewidth=1)
plt.plot(t_train_daily.index, t_train_daily['predicted'], label='Predicted Train', linewidth=1)
plt.plot(t_test_daily.index,  t_test_daily['actual'],     label='Actual Test',     linewidth=1)
plt.plot(t_test_daily.index,  t_test_daily['predicted'],  label='Predicted Test',  linewidth=1)

plt.xlabel('Date')
plt.ylabel('Price Change')
plt.title('LSTM: Actual vs Predicted (Aggregated Daily)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

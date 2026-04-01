import os
os.chdir(r"C:\Users\Khanh Hoang\Desktop\Pe_alice\phrase1_%PriceChange")
import pandas as pd

# Path to the original CSV file
input_csv_file = "2018_nasdaq_external.csv"
output_csv_file = "2018_nasdaq_cleaned.csv"

# Read data from the CSV file
df = pd.read_csv(input_csv_file)

# ✅ Remove rows where the "Stock_symbol" column is null (NaN)
df_cleaned = df[df["Stock_symbol"].notnull()]

# Save the cleaned data to a new CSV file
df_cleaned.to_csv(output_csv_file, index=False)

# Report the result
print(f"✅ Cleaned dataset saved to: {output_csv_file}")
print(f"Rows before cleaning: {len(df)}, Rows after cleaning: {len(df_cleaned)}")

# --------------------- Main code to compute sentiment score
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Load Hugging Face model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# ✅ Function to calculate sentiment score
def calculate_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    sentiment_score = torch.arange(-2, 3).float().to(device) @ scores.T
    return sentiment_score.item()

# ✅ Load the cleaned dataset
file_path = "2018_nasdaq_cleaned.csv"
df = pd.read_csv(file_path)

# ✅ Convert Date to date-only format
df["Date"] = pd.to_datetime(df["Date"]).dt.date

# ✅ Initialize an empty list to store processed data
processed_data = []

# ✅ Group by Date and process each group
for date, group in df.groupby("Date"):
    print(f"📅 Processing date: {date} ({len(group)} rows)...")
    
    # Calculate sentiment score for each Article_title on this date
    group["sentiment_score"] = group["Article_title"].apply(calculate_sentiment_score)
    
    # Append the processed group to the list
    processed_data.append(group)
    
    print(f"✅ Finished processing {date}.")

# ✅ Concatenate all processed groups into a single DataFrame
df_processed = pd.concat(processed_data)

# ✅ Group by Date and Stock_symbol to calculate the average sentiment score
grouped_df = df_processed.groupby(["Date", "Stock_symbol"])["sentiment_score"].mean().reset_index()

# ✅ Pivot the table: Rows = Date, Columns = Stock_symbol, Values = Average Sentiment Score
pivot_df = grouped_df.pivot(index="Date", columns="Stock_symbol", values="sentiment_score")

# ✅ Fill missing values with 0 (or another value if necessary)
pivot_df = pivot_df.fillna(0)

# ✅ Save the resulting table to a new CSV file
pivot_df.to_csv("daily_sentiment_by_stock.csv")

print("🎉 Sentiment analysis table created successfully! Check 'daily_sentiment_by_stock.csv'.")

# -------------- View summary of the dataset with average sentiment scores
import pandas as pd

# Define the path to your CSV file
file_path = "daily_sentiment_by_stock.csv"

# Read the CSV file
df = pd.read_csv('daily_sentiment_by_stock.csv')

# Summary information
print("✅ File Summary:")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Show the first 5 rows for inspection
print("\n✅ First 5 rows of the dataset:")
print(df.head())

# Summary statistics for numeric columns (e.g., sentiment scores)
print("\n✅ Summary Statistics:")
print(df.describe())
num_columns = len(df.columns)
print(f"✅ Total number of columns: {num_columns}")

# CHI CAN CHAY LAI TU BUOC AY CHO %PRICE CHANGE
# -----------------------------------------
# ----------------------
# Extract only the 2018 data
import pandas as pd

# Load the dataset
file_path = "nasdaq_historical_data.csv"  # Adjust the path if needed
df = pd.read_csv(file_path)

# Summary information
print("✅ Dataset Summary:")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}\n")

# Show the first 5 rows for inspection
print("✅ First 5 rows of the dataset:")
print(df.head())

# Show data types of each column
print("\n✅ Column Data Types:")
print(df.dtypes)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%y')

# Filter rows where the year is 2018
df_2018 = df[df['Date'].dt.year == 2018]

# Display the filtered rows
print("✅ Rows from 2018:")
print(df_2018)

# Optional: Save filtered rows to a new CSV file
df_2018.to_csv("nasdaq_2018_price_change.csv", index=False)

# -------------------- Create 'Price_Change' column in the initial dataset
import pandas as pd

# Load the existing dataset
file_path = "nasdaq_2018_price_change.csv"  # Make sure this is the correct file
df = pd.read_csv(file_path)

# Create or update the 'Price_Change' column as (Close/Last - Open) / Open
df['Price_Change'] = (df['Close/Last'] - df['Open']) / df['Open']

# Display the updated DataFrame
print("✅ Updated Dataset with 'Price_Change' column:")
print(df.head())

# Save the changes back to the same file (overwrite it)
df.to_csv(file_path, index=False)

print(f"✅ 'Price_Change' column added and saved in {file_path}")

# -------------------- Merge the two datasets together
import pandas as pd

# Load both datasets
sentiment_file = "daily_sentiment_by_stock.csv"
price_change_file = "nasdaq_2018_price_change.csv"

df_sentiment = pd.read_csv(sentiment_file)
df_price_change = pd.read_csv(price_change_file)

# Convert 'Date' columns to datetime for both datasets to ensure proper merging
df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])
df_price_change['Date'] = pd.to_datetime(df_price_change['Date'])

# Merge the datasets on 'Date' column, keeping all rows from df_sentiment
df_merged = pd.merge(df_sentiment, df_price_change[['Date', 'Price_Change']], on='Date', how='left')

# Save the combined dataset
output_file = "2018_full.csv"
df_merged.to_csv(output_file, index=False)

# Display the first 5 rows of the combined dataset
print("✅ Combined Dataset (First 5 Rows):")
print(df_merged.head(7))

# TEST ---------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the dataset
file_path = "daily_sentiment_by_stock.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

# ✅ Display basic information
print(df.head())
print(f"Number of unique dates: {df['Date'].nunique()}")
print(f"Number of unique stock symbols: {len(df.columns) - 1}")

# ✅ Create distribution plot of sentiment scores across all days and stocks
plt.figure(figsize=(15, 6))
sns.boxplot(data=df.drop(columns=["Date"]), orient="h")
plt.title("Distribution of Sentiment Scores Across All Stock Symbols")
plt.xlabel("Sentiment Score")
plt.ylabel("Stock Symbols")
plt.show()

# ✅ Create distribution plot of sentiment scores for each stock
plt.hist(pivot_df.values.flatten(), bins=50, edgecolor='black')
plt.title("Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------
# Check which days have missing price change
import pandas as pd

# Load the dataset
file_path = "2018_full.csv"
df = pd.read_csv(file_path)

# Find dates where 'Price_Change' is missing
missing_price_change_dates = df[df['Price_Change'].isna()]['Date'].unique()

# Print the results
print(f"Total number of days with missing 'Price_Change': {len(missing_price_change_dates)}")
print("Dates with missing 'Price_Change':")
for date in missing_price_change_dates:
    print(date)

# ---------------- Process data for days without price change
import pandas as pd

# ✅ Load dataset
file_path = "2018_full.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ✅ Identify sentiment columns (all except Date & Price_Change)
sentiment_cols = df.columns[1:-1]

# ✅ Initialize a buffer to store sentiment scores of non-trading days
buffer = {col: [] for col in sentiment_cols}
processed_rows = []

# ✅ Iterate through the dataset
for idx, row in df.iterrows():
    if pd.isna(row["Price_Change"]):  
        # Store sentiment scores for non-trading days
        for col in sentiment_cols:
            buffer[col].append(row[col])
    else:
        # When reaching a trading day, average the buffered sentiment scores with the current day
        averaged_sentiment = {}
        for col in sentiment_cols:
            if buffer[col]:  
                buffer[col].append(row[col])  # Include the current trading day’s sentiment
                averaged_sentiment[col] = sum(buffer[col]) / len(buffer[col])  # Compute the average
            else:
                averaged_sentiment[col] = row[col]  # If no buffer, keep original score
        
        # Store the processed row
        processed_rows.append([row["Date"], *averaged_sentiment.values(), row["Price_Change"]])

        # Clear buffer after assigning sentiment to a valid trading day
        buffer = {col: [] for col in sentiment_cols}

# ✅ Create a new DataFrame with processed data
df_cleaned = pd.DataFrame(processed_rows, columns=["Date", *sentiment_cols, "Price_Change"])

# ✅ Save the cleaned dataset
df_cleaned.to_csv("2018_full_processed.csv", index=False)

print("🎉 Process completed! Sentiment scores for non-trading days have been averaged into the next available trading day.")

# ---------- Verify whether any data is missing in 2018_full_processed
import pandas as pd

# Read the processed file
file_path = "2018_full_processed.csv"
df = pd.read_csv(file_path)

# Check rows with null Price_Change
null_rows = df[df["Price_Change"].isnull()]

# Display the result
print(f"Total rows with missing Price_Change: {len(null_rows)}")
print(null_rows)

# ───────────────────────────────────────────
# 1) Load & standardize data
# ───────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import regularizers

file_path = "2018_full_processed.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

df.set_index("Date", inplace=True)
X = df.drop(columns=["Price_Change"]).values          # Sentiment scores
y = df["Price_Change"].values.reshape(-1, 1)          # Target

scalerX = StandardScaler()
scalery = StandardScaler()

X_scaled = scalerX.fit_transform(X)
y_scaled = scalery.fit_transform(y)

# ───────────────────────────────────────────
# 2) Create LSTM sequences (look_back = 5)
# ───────────────────────────────────────────
def create_dataset(dataX, dataY, look_back=5):
    xs, ys = [], []
    for i in range(len(dataX) - look_back):
        xs.append(dataX[i : i + look_back, :])
        ys.append(dataY[i + look_back])
    return np.array(xs), np.array(ys)

look_back = 5
X_lstm, y_lstm = create_dataset(X_scaled, y_scaled, look_back)

# Reshape (samples, time_steps, features)
X_lstm = X_lstm.reshape((X_lstm.shape[0], look_back, X_lstm.shape[2]))

# ───────────────────────────────────────────
# 3) Train–test split 80–20 (maintain chronological order)
# ───────────────────────────────────────────
train_size = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

# Using the same dates after removing look_back offset
all_dates = df.index.values[look_back:]
train_dates = all_dates[:train_size]
test_dates  = all_dates[train_size:]

# ───────────────────────────────────────────
# 4) Build & train the model
# ───────────────────────────────────────────
model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh',
         input_shape=(look_back, X.shape[1]),
         kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    LSTM(64, return_sequences=True, activation='tanh'),
    Dropout(0.3),
    LSTM(64, activation='tanh'),
    Dropout(0.3),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=16,
          validation_split=0.2, verbose=1)

# ───────────────────────────────────────────
# 5) Predict & inverse transform
# ───────────────────────────────────────────
train_pred_scaled = model.predict(X_train)
test_pred_scaled  = model.predict(X_test)

train_pred = scalery.inverse_transform(train_pred_scaled)
train_true = scalery.inverse_transform(y_train.reshape(-1, 1))

test_pred  = scalery.inverse_transform(test_pred_scaled)
test_true  = scalery.inverse_transform(y_test.reshape(-1, 1))

# ───────────────────────────────────────────
# 6) Compute MAE, MSE, RMSE for train & test
# ───────────────────────────────────────────
def metric_scores(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

train_mae, train_mse, train_rmse = metric_scores(train_true, train_pred)
test_mae , test_mse , test_rmse  = metric_scores(test_true,  test_pred)

print("\n── Evaluation ──")
print(f"Train MAE : {train_mae :.5f}")
print(f"Train MSE : {train_mse :.5f}")
print(f"Train RMSE: {train_rmse:.5f}")
print(f"Test  MAE : {test_mae  :.5f}")
print(f"Test  MSE : {test_mse  :.5f}")
print(f"Test  RMSE: {test_rmse :.5f}")

# ───────────────────────────────────────────
# 7) Plot Actual vs Predicted
# ───────────────────────────────────────────
plt.figure(figsize=(15, 6))

# Train
plt.plot(train_dates, train_true, label='Actual Train', color='blue')
plt.plot(train_dates, train_pred, label='Predicted Train', color='orange')

# Test
plt.plot(test_dates, test_true, label='Actual Test', color='green')
plt.plot(test_dates, test_pred, label='Predicted Test', color='red')

plt.xlabel("Date")
plt.ylabel("Price Change")
plt.title("LSTM: Actual vs Predicted (Train & Test)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------- 
# Compute accuracy metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ───────────────────────────────────────────
# 7) Compute Accuracy, Precision, Recall, F1, Confusion Matrix
# ───────────────────────────────────────────

# Binarize: >0 as 1 (Positive), <=0 as 0 (Negative)
train_pred_label = (train_pred.flatten() > 0).astype(int)
train_true_label = (train_true.flatten() > 0).astype(int)

test_pred_label = (test_pred.flatten() > 0).astype(int)
test_true_label = (test_true.flatten() > 0).astype(int)

# Train metrics
train_accuracy = accuracy_score(train_true_label, train_pred_label)
train_precision = precision_score(train_true_label, train_pred_label)
train_recall = recall_score(train_true_label, train_pred_label)
train_f1 = f1_score(train_true_label, train_pred_label)
train_cm = confusion_matrix(train_true_label, train_pred_label)

# Test metrics
test_accuracy = accuracy_score(test_true_label, test_pred_label)
test_precision = precision_score(test_true_label, test_pred_label)
test_recall = recall_score(test_true_label, test_pred_label)
test_f1 = f1_score(test_true_label, test_pred_label)
test_cm = confusion_matrix(test_true_label, test_pred_label)

# Print results
print("\n── Classification Evaluation (Train) ──")
print(f"Train Accuracy : {train_accuracy:.5f}")
print(f"Train Precision: {train_precision:.5f}")
print(f"Train Recall   : {train_recall:.5f}")
print(f"Train F1 Score : {train_f1:.5f}")
print(f"Train Confusion Matrix:\n{train_cm}")

print("\n── Classification Evaluation (Test) ──")
print(f"Test Accuracy : {test_accuracy:.5f}")
print(f"Test Precision: {test_precision:.5f}")
print(f"Test Recall   : {test_recall:.5f}")
print(f"Test F1 Score : {test_f1:.5f}")
print(f"Test Confusion Matrix:\n{test_cm}")

# -----
# Combine train and test
all_true_label = np.concatenate([train_true_label, test_true_label])
all_pred_label = np.concatenate([train_pred_label, test_pred_label])

# Compute overall metrics
overall_accuracy = accuracy_score(all_true_label, all_pred_label)
overall_precision = precision_score(all_true_label, all_pred_label)
overall_recall = recall_score(all_true_label, all_pred_label)
overall_f1 = f1_score(all_true_label, all_pred_label)
overall_cm = confusion_matrix(all_true_label, all_pred_label)

# Print overall result
print("\n── Overall Classification Evaluation ──")
print(f"Overall Accuracy : {overall_accuracy:.5f}")
print(f"Overall Precision: {overall_precision:.5f}")
print(f"Overall Recall   : {overall_recall:.5f}")
print(f"Overall F1 Score : {overall_f1:.5f}")
print(f"Overall Confusion Matrix:\n{overall_cm}")

# Plot overall confusion matrix
plot_confusion_matrix(overall_cm, "Overall Confusion Matrix")

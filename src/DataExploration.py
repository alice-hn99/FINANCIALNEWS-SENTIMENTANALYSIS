# PART 1 - DATA CLEANING ORIGINAL NASDAQ DATA
import pandas as pd

# Load data
file_path = "C:/Users/vuthu/Downloads/nasdaq_exteral_data.csv"

# Cleaning data in chunks of 10,000 rows each time to avoid memory issues
# chunk_iter = pd.read_csv(file_path, chunksize=10000)
# count = 0
# for chunk in chunk_iter:
#   print(chunk.info())
#   count = count + 1
# print(count)

# Keep only Date,Article_title, Stock_symbol columns since there are missing data in the index column.
chunk_iter = pd.read_csv(file_path, chunksize=10000)

count = 0
total_missing_values = None  # Initialize missing values accumulator
newsheadline_2018_2019  = [] # empty list to store filtered chunks

for chunk in chunk_iter:
  chunk = chunk.iloc[:, 1:4]
  # print(chunk.info())

  # Convert 'Date' column to datetime format

  chunk["Date"] = pd.to_datetime(chunk["Date"], errors='coerce')

# Fill missing values in the 'Stock_symbol' column with "Missed symbol"
  # if 'Stock_symbol' in chunk.columns:
  #    chunk['Stock_symbol'] = chunk['Stock_symbol'].fillna("Missed symbol")

# Drop rows where 'Article_title' is missing
  chunk = chunk.dropna(subset=['Article_title'])

##Filter rows between Jan 1, 2018 - Dec 31, 2018
  filtered_2018_2019 = chunk[(chunk["Date"] >= "2018-01-01") & (chunk["Date"] <= "2018-12-31")]
 # Append filtered data to the list
  newsheadline_2018_2019 .append(filtered_2018_2019)


# Sum missing values for this chunk
  # missing_values = chunk.isnull().sum()

 # Accumulate missing values across chunks
  # if total_missing_values is None:
  #   total_missing_values = missing_values
  # else: 
  #   total_missing_values += missing_values


  count = count + 1

# Printing results
print(f"Total missing value of all chunks: {total_missing_values}")
print(f"Total chunks processed: {count}")
# Concatenate all filtered chunks into a single DataFrame
newsheadline_2018_2019_df = pd.concat(newsheadline_2018_2019, ignore_index=True)

# Aggregate news titles for each date and symbol
#grouped_df = newsheadline_2018_2019_df.groupby(["Date", "Stock_symbol"])["Article_title"].apply(lambda x: " | ".join(x)).reset_index()


# Get the shape (rows, columns) of the DataFrame
print(f"newsheadline shape: {newsheadline_2018_2019_df.shape}")



# Part 2 - Text Preprocessing and Lemmatization for NLP
import string
import sys
import nltk
nltk.download('averaged_perceptron_tagger', download_dir='C:/Users/vuthu/nltk_data')
nltk.download('wordnet')
nltk.download('stopwords', download_dir='C:/Users/vuthu/nltk_data')
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
nltk.data.path.append('C:/Users/vuthu/nltk_data')

import nltk
nltk.download('punkt')

from nltk import pos_tag, word_tokenize

text = "This is a test sentence."
tokens = word_tokenize(text)
print(pos_tag(tokens))


# Load custom stopwords from file
stopwords_path = r"C:\Users\vuthu\OneDrive\Desktop\Master program\CAPSTONE PROJECT\gist_stopwords.txt"

# Read the stopwords file
with open(stopwords_path, "r", encoding="utf-8") as f:
    stop_words = set(f.read().splitlines())

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # remove \t
    text = text.replace('\t', '')
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    text = [x for x in text if x not in stop_words]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = ' '.join(text)
    return(text)

# Apply text preprocessing to "Article_title"
newsheadline_2018_2019_df["Cleaned_Article_Title"] = newsheadline_2018_2019_df["Article_title"].apply(clean_text)


# PART 3 - using python to modify and calculate the change of stock price
import pandas as pd

# File paths
file_2018 = "C:/Users/vuthu/Downloads/Download Data - INDEX_US_S&P US_SPX 2018.csv"
file_2019 = "C:/Users/vuthu/Downloads/Download Data - INDEX_US_S&P US_SPX 2019.csv"

# Load the CSV files
df_2018 = pd.read_csv(file_2018)
df_2019 = pd.read_csv(file_2019)

# Combine both DataFrames
df_combined = pd.concat([df_2018, df_2019], ignore_index=True)
print(df_combined.head(-20))
print(df_combined.head())

# Convert 'Date' column to datetime format
df_combined["Date"] = pd.to_datetime(df_combined["Date"])

# Remove commas & convert 'Open', 'Close', 'High', 'Low' to float
columns_to_convert = ["Open", "Close", "High", "Low"]
for col in columns_to_convert:
    df_combined[col] = df_combined[col].astype(str).str.replace(",", "").astype(float)
print(df_combined.dtypes)
print(df_combined.head())

# Calculate Change = (Close - Open) / Open
df_combined["Change"] = (df_combined["Close"] - df_combined["Open"]) / df_combined["Open"]
print(df_combined.head())

# PART 4 - Generate Sentiment Score for  newsheadline_2018_2019_df
# FINANCIALNEWS-SENTIMENTANALYSIS
Our mission is to help  investors make smarter, faster decisions by using AI-driven sentiment analysis to turn financial news into clear,  actionable insights.

📊 Project Overview

This project investigates the effectiveness of various Natural Language Processing (NLP) models in forecasting stock price movements based on financial news headlines. The primary goal was to determine whether a general-purpose financial sentiment model (FinBERT) can reliably predict price changes across different industries or if sector-specific fine-tuning is required for better accuracy.

🛠️ Tech StackLanguage: 

PythonSentiment Models: TextBlob (Lexicon-based), BERT (General Transformer), FinBERT (Domain-specific Transformer).
Time-Series Forecasting: Long Short-Term Memory (LSTM) Networks.
Data Source: Bloomberg Terminal news headlines and NASDAQ historical trading data (2018).

📈 Data PipelinePreprocessing: 

Cleaned a subset of 696,948 financial records from 2018.Sentiment 
Scoring: Generated daily average sentiment scores per ticker using TextBlob, BERT, and FinBERT.
Feature Engineering: Calculated daily stock price changes ($Price\_Change = \frac{Close - Open}{Open}$) and merged them with sentiment features.
Modeling: Fed the combined sentiment and historical price data into an LSTM network to capture temporal market patterns.

🔬 Key Findings

1. Model Selection (Phase 1)Among the tested architectures, FinBERT-LSTM emerged as the most robust model, significantly outperforming TextBlob and general BERT in both classification and regression tasks.

2. Cross-Sector Generalization (Phase 2)The study found that the general FinBERT model delivered high predictive accuracy across Technology, Real Estate, Healthcare, Energy, and Finance.
   
Macro Trends: These sectors are heavily influenced by broad economic indicators already encoded in FinBERT.

The Entertainment Exception: This was the only sector where a fine-tuned model outperformed the general version, due to the industry's unique use of nuanced and media-specific language.

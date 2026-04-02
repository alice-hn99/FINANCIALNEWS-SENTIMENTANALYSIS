# FINANCIALNEWS-SENTIMENTANALYSIS
Our mission is to help  investors make smarter, faster decisions by using AI-driven sentiment analysis to turn financial news into clear,  actionable insights.

📖 Introduction

This study investigates the effectiveness of various sentiment analysis models in forecasting stock price movements based on financial news headlines. Market sentiment plays a critical role in financial decisions, but the increasing volume and speed of information have made traditional manual review impractical. This project evaluates whether sector-level customization improves prediction accuracy enough to justify the extra cost and complexity compared to general models.

🛠️ Methodology

The study followed a two-phase research framework:

Phase 1: General Model Development: Tested various sentiment analysis models—TextBlob, BERT, and FinBERT—to determine which most accurately interprets financial news sentiment.

Phase 2: Industry-Specific Development: Fine-tuned the best-performing model (FinBERT) for six target industries: Technology, Energy, Healthcare, Real Estate, Finance, and Entertainment.Modeling Approach: Integrated sentiment scores with historical stock price data using Long Short-Term Memory (LSTM) networks to capture temporal market patterns.

🖼️ Visualizations & Results1. 

Model Comparison (Phase 1)
[IMAGE_NOTE: Insert Figure 11 - Classification Metrics Comparison showing FinBERT outperforming others]

FinBERT emerged as the most robust model, outperforming alternatives in accuracy, precision, recall, and F1 score. It achieved a test accuracy of 70.44% and a test F1 score of 0.71.

Sector-Specific Performance
[IMAGE_NOTE: Insert Figure 13 (Tech), Figure 14 (Real Estate), or Figure 16 (Energy) to show Actual vs. Predicted Price Change]

The general FinBERT model outperformed industry-specific versions in five out of six sectors due to the strong influence of macroeconomic trends. Entertainment was the only sector where a fine-tuned model performed better, likely due to the unique cultural language and subjective tone of media news.

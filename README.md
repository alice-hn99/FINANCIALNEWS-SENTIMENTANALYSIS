# FINANCIALNEWS-SENTIMENTANALYSIS
Our mission is to help  investors make smarter, faster decisions by using AI-driven sentiment analysis to turn financial news into clear,  actionable insights.

📖 Introduction

This study investigates the effectiveness of various sentiment analysis models in forecasting stock price movements based on financial news headlines. Market sentiment plays a critical role in financial decisions, but the increasing volume and speed of information have made traditional manual review impractical. This project evaluates whether sector-level customization improves prediction accuracy enough to justify the extra cost and complexity compared to general models.

🛠️ Methodology

The study followed a two-phase research framework:

Phase 1: General Model Development: Tested various sentiment analysis models—TextBlob, BERT, and FinBERT—to determine which most accurately interprets financial news sentiment.

Phase 2: Industry-Specific Development: Fine-tuned the best-performing model (FinBERT) for six target industries: Technology, Energy, Healthcare, Real Estate, Finance, and Entertainment.

Modeling Approach: Integrated sentiment scores with historical stock price data using Long Short-Term Memory (LSTM) networks to capture temporal market patterns.


🖼️ Visualizations & Results. 

Model Comparison (Phase 1)
[IMAGE_NOTE: Insert Figure 11 - Classification Metrics Comparison showing FinBERT outperforming others]

FinBERT emerged as the most robust model, outperforming alternatives in accuracy, precision, recall, and F1 score. 

It achieved a test accuracy of 70.44% and a test F1 score of 0.71.

Sector-Specific Performance
[IMAGE_NOTE: Insert Figure 13 (Tech), Figure 14 (Real Estate), or Figure 16 (Energy) to show Actual vs. Predicted Price Change]

The general FinBERT model outperformed industry-specific versions in five out of six sectors due to the strong influence of macroeconomic trends. 

Entertainment was the only sector where a fine-tuned model performed better, likely due to the unique cultural language and subjective tone of media news.

Gemini đã nói
Based on the provided project report, here is the professionally structured content for your Financial News Sentiment Analysis GitHub README:

📈 Financial News Sentiment Analysis using AI
📖 Introduction
This study investigates the effectiveness of various sentiment analysis models in forecasting stock price movements based on financial news headlines. Market sentiment plays a critical role in financial decisions, but the increasing volume and speed of information have made traditional manual review impractical. This project evaluates whether sector-level customization improves prediction accuracy enough to justify the extra cost and complexity compared to general models.
+2

🛠️ Methodology
The study followed a two-phase research framework:


Phase 1: General Model Development: Tested various sentiment analysis models—TextBlob, BERT, and FinBERT—to determine which most accurately interprets financial news sentiment.


Phase 2: Industry-Specific Development: Fine-tuned the best-performing model (FinBERT) for six target industries: Technology, Energy, Healthcare, Real Estate, Finance, and Entertainment.
+2


Modeling Approach: Integrated sentiment scores with historical stock price data using Long Short-Term Memory (LSTM) networks to capture temporal market patterns.
+1

🖼️ Visualizations & Results
1. Model Comparison (Phase 1)
<img width="1646" height="796" alt="image" src="https://github.com/user-attachments/assets/b2489738-054a-4166-b792-aaed0d64a0cf" />

FinBERT emerged as the most robust model, outperforming alternatives in accuracy, precision, recall, and F1 score.

It achieved a test accuracy of 70.44% and a test F1 score of 0.71.

2. Sector-Specific Performance

<img width="1776" height="812" alt="image" src="https://github.com/user-attachments/assets/c48d2043-5484-468b-a506-063bd3f935aa" />
Actual vs Predicted Daily Price Change (Train vs Test) for technology sector


<img width="1536" height="750" alt="image" src="https://github.com/user-attachments/assets/18e2e444-57a6-45a2-acf9-8ce7595855be" />
Actual vs Predicted Real Estate Sector– Last 50 Train, first 50 Test Observations



<img width="1624" height="812" alt="image" src="https://github.com/user-attachments/assets/b8aefadf-e98e-44c5-8247-1ee77b3f04e2" />
ctual vs Predicted Percentage Price Change (Train vs Test) for Energy Sector


The general FinBERT model outperformed industry-specific versions in five out of six sectors due to the strong influence of macroeconomic trends.

Entertainment was the only sector where a fine-tuned model performed better, likely due to the unique cultural language and subjective tone of media news.

🚀 Key Findings

FinBERT is the optimal baseline: It is highly aligned with financial language, enabling strong cross-sector generalizability.

Macro trends dominate: Broad economic indicators (interest rates, inflation) often affect multiple sectors simultaneously, making general models sufficient for most industries.

Domain adaptation: Targeted refinement is only necessary for sectors with specialized communication patterns, such as Entertainment.


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

<img width="1600" height="764" alt="image" src="https://github.com/user-attachments/assets/b30f5b8b-b9c7-4e00-8ad8-2bab93c03233" />
Classification Metrics Comparison

FinBERT emerged as the most robust model, outperforming alternatives in accuracy, precision, recall, and F1 score. 

It achieved a test accuracy of 70.44% and a test F1 score of 0.71.

Sector-Specific Performance

The general FinBERT model outperformed industry-specific versions in five out of six sectors due to the strong influence of macroeconomic trends. 

Entertainment was the only sector where a fine-tuned model performed better, likely due to the unique cultural language and subjective tone of media news.

<img width="1606" height="764" alt="image" src="https://github.com/user-attachments/assets/b07cb8c3-c040-4bae-be59-bb72fef7dc04" />
Actual vs Predicted Daily Price Change (Train vs Test) for technology sector


<img width="1480" height="742" alt="image" src="https://github.com/user-attachments/assets/a64e8cf2-dabf-4db4-91a9-3f0f20f2f51a" />
Actual vs Predicted Real Estate Sector– Last 50 Train, first 50 Test Observations


<img width="1602" height="754" alt="image" src="https://github.com/user-attachments/assets/250e73be-6c80-4a00-9316-d4873f3da1ed" />
Actual vs Predicted Percentage Price Change (Train vs Test) for Energy Sector

🚀 Key Findings

FinBERT is the optimal baseline: It is highly aligned with financial language, enabling strong cross-sector generalizability.

Macro trends dominate: Broad economic indicators (interest rates, inflation) often affect multiple sectors simultaneously, making general models sufficient for most industries.

Domain adaptation: Targeted refinement is only necessary for sectors with specialized communication patterns, such as Entertainment.


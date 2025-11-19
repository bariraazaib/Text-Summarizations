# Encoder–Decoder Text Summarization with T5 / BART

This project implements **abstractive text summarization** using an Encoder–Decoder architecture (T5 / BART) on the **CNN / DailyMail news dataset**. The model generates concise summaries of input articles.

## 📂 Dataset
- Source: [CNN/DailyMail Summarization Dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
- Contains news articles paired with human-written summaries.

## 🛠 Project Components

1. **Dataset Preprocessing**
   - Extract article-summary pairs.
   - Tokenization and padding for Encoder–Decoder input.

2. **Model Fine-Tuning**
   - Models: `t5-base` or `facebook/bart-base`
   - Framework: Hugging Face Transformers
   - Training pipeline includes optimizer, scheduler, and evaluation loop.

3. **Evaluation**
   - Metrics: ROUGE-1, ROUGE-2, ROUGE-L
   - Qualitative comparison between original, reference, and generated summaries.
   
   Example:
   ROUGE1_F1 : 0.369 ± 0.047
   ROUGE2_F1 : 0.115 ± 0.088
   ROUGEL_F1 : 0.274 ± 0.016

4. **Example Outputs**
- Input Article: "The company reported strong earnings this quarter with profits increasing by 20%..."
- Reference Summary: "Company profits grew 20% due to new products and market expansion."
- Generated Summary: "The company reported strong earnings this quarter with profits increasing by 20%. CEO expressed optimism about future growth prospects. The company's profits were driven by successful product launches and expanding market share."

## 🚀 Live Demo
Try the model live using **Streamlit / Gradio**:  
[[Your Streamlit/Gradio link here](https://paragraph-summarizer-hwzhjaixyseivqrnsw3bn8.streamlit.app/)]

## 💻 Code Repository
[GitHub Repository](https://github.com/YourUsername/Text-Summarization-T5-BART)

## 🔧 Dependencies
- Python >= 3.8
- transformers
- torch
- datasets
- rouge-score
- streamlit / gradio

## ⚡ Notes
- Supports **dynamic max_length** for summaries.
- Works for news articles, reports, and other structured text.
- Can be extended for multilingual summarization or domain-specific summarization.



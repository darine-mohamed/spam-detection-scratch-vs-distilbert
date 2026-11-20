# Transformer from Scratch vs Pretrained DistilBERT – SMS Spam Detection

Course project: one NLP task (**spam / ham classification**), two Transformer models trained/evaluated on the
same split, compared in a Streamlit app.

| | Model A | Model B |
|---|---|---|
| Name | Scratch Transformer encoder | DistilBERT (fine-tuned) |
| Weights | random init, trained here | `distilbert-base-uncased` (Hugging Face) + fine-tuning |
| Code | `scratch_model.py` (own attention, positional emb., FFN, residual, LayerNorm, masks) | `transformers` |

## Dataset
**SMS Spam Collection** – Almeida, Gómez Hidalgo, Yamakami, *Contributions to the Study of SMS Spam Filtering*, ACM DocEng 2011.
UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/228/sms+spam+collection (CC BY 4.0 – verify on the page).
The copy used in this project was downloaded from **Kaggle** (file `spam_sms.csv`; Kaggle page URL: ___ add the link you used). The original source is the UCI repository above.
5,574 messages → 5,168 after removing duplicates and empty texts → stratified 80/10/10 split (seed 42):
**4,134 train / 517 validation / 517 test**. Both models use identical rows; test data is never used for tuning.

## Project structure
```
app.py                      Streamlit app
scratch_model.py            Model A + preprocessing used for training
Task3_Full_Project.ipynb    Full notebook: Part 1 scratch model, Part 2 DistilBERT + comparison
requirements.txt
artifacts/                  scratch_model.pt, vocab.json, scratch_results.json, scratch_test_predictions.csv
artifacts_distilbert/       distilbert_model/, distilbert_results.json, distilbert_test_predictions.csv, charts
```

## Installation
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the experiment
1. Open `Task3_Full_Project.ipynb` in Google Colab, choose a **T4 GPU**, *Run all*, upload `spam_sms.csv` when asked.
2. Download `project_artifacts.zip` at the end and unzip it into this folder so that `artifacts/` and
   `artifacts_distilbert/` sit next to `app.py`.

## Run the app
```bash
streamlit run app.py
```
Tabs: **Try it** (input, examples, both models side by side with measured latency), **Comparison dashboard**
(metrics table, charts, confusion matrices), **Test predictions** (per-message results, disagreements), **About**.
If a model folder is missing the app shows a clear message instead of crashing.

## Settings
| | Scratch | DistilBERT |
|---|---|---|
| Input text | `clean_message` (lowercased, letters only) | original `message` |
| Tokenizer | regex word tokenizer, vocab 6,843 | WordPiece, max 128 tokens |
| Architecture | 2 layers, d_model 128, 4 heads, FF 256, dropout 0.15 | DistilBERT base (6 layers, 768 dim) |
| Loss | class-weighted cross-entropy (0.57 / 3.96) | same weights |
| Optimizer | AdamW lr 2e-4, wd 1e-4, 8 epochs, batch 32 | AdamW lr 2e-5, wd 0.01, 3 epochs, batch 32, linear warm-up 10% |
| Model selection | best validation loss | best validation loss |
| Seed | 42 | 42 |

Latency in the app = preprocessing/tokenization + forward pass for one message, on the machine running the app.
Latency in the results table = forward pass per message with batch size 32 (same method for both models).

## Notes
- `artifacts_distilbert/distilbert_model` is ~250 MB. If it is not included, re-run Part 2 of the notebook
  (the model is downloaded automatically from Hugging Face: `distilbert-base-uncased`).

# Short Report – Scratch Transformer vs Pretrained DistilBERT (SMS Spam Detection)

## 1. Task and dataset
Binary classification of SMS messages as ham or spam. Dataset: SMS Spam Collection (Almeida et al. 2011; original source UCI, the copy we used was downloaded from Kaggle).
5,574 messages; 403 duplicates and 3 empty texts removed → 5,168 (87.4% ham, 12.6% spam). The imbalance is handled with
class-weighted cross-entropy. Split (stratified, seed 42): 4,134 train / 517 validation / 517 test (452 ham, 65 spam).

## 2. Models
**Model A – scratch Transformer encoder.** Token + learned positional embeddings → 2 encoder blocks (own multi-head
self-attention with scaling and padding mask, feed-forward, residual connections, LayerNorm) → masked mean pooling →
linear classifier. 1,153,922 parameters, random initialisation, no pretrained weights or built-in attention layers.

**Model B – DistilBERT.** `distilbert-base-uncased` with a classification head, fine-tuned on the same training set
(settings in README). ~66M parameters.

## 3. Results on the held-out test set (517 messages)
Measured values from `artifacts/scratch_results.json` and `artifacts_distilbert/distilbert_results.json`. Both models were trained and
evaluated in the same Colab session on a **Tesla T4 GPU**, same seed (42), same split, same 517 test messages.

| Metric | Scratch Transformer | DistilBERT (fine-tuned) |
|---|---|---|
| Accuracy | 0.9710 | 0.9923 |
| Macro F1 | 0.9377 | 0.9824 |
| Spam precision | 0.8378 | 0.9692 |
| Spam recall | 0.9538 | 0.9692 |
| Spam F1 | 0.8921 | 0.9692 |
| Parameters | 1,153,922 | 66,955,010 |
| Training / fine-tuning time (s) | 11.3 (8 epochs) | 78.3 (3 epochs) |
| Latency / message, batch 32 (ms) | 0.230 | 2.134 |
| Single-message latency incl. tokenization (ms) | see app (measured live) | 18.67 |
| Hardware | Tesla T4 GPU (Colab) | Tesla T4 GPU (Colab) |

Confusion matrices (rows = true, columns = predicted; Ham, Spam):
- Scratch: [[440, 12], [3, 62]] – 15 errors (12 normal messages flagged as spam, 3 spam missed).
- DistilBERT: [[450, 2], [2, 63]] – 4 errors (2 normal messages flagged as spam, 2 spam missed).

Note: the single-message latency shown in the Streamlit app is measured live for both models on the machine running the app
(typically a CPU laptop), so those values differ from the GPU numbers in this table.

## 4. Examples (from the 517 test messages)
The models disagree on 15 messages and are both wrong on 2.

| # | Message (shortened) | True | Scratch | DistilBERT |
|---|---|---|---|---|
| 1 | "Someone has contacted our dating service and entered your phone becausethey fancy you! … call from a landline 090580…" | Spam | Spam | Spam |
| 2 | "Urgent Urgent! We have 800 FREE flights to Europe to give away, call B4 10th Sept & take a friend 4 FREE…" | Spam | Spam | Spam |
| 3 | ""CHEERS U TEX MECAUSE U WEREBORED! YEAH OKDEN HUNNY R UIN WK SAT?…"" | Ham | **Spam** | Ham |
| 4 | ""Hi missed your Call and my mumHas beendropping red wine all over theplace! what is your adress?"" | Ham | **Spam** | Ham |
| 5 | "ringtoneking 84484" | Spam | **Ham** | Spam |
| 6 | "Latest News! Police station toilet stolen, cops have nothing to go on!" | Spam | Ham | Ham |
| 7 | "staff.science.nus.edu.sg/~phyhcmk/teaching/pc1323" | Ham | Ham | **Spam** |
| 8 | "Xmas & New Years Eve tickets are now on sale from the club, during the day from 10am till 8pm…" | Spam | Spam | **Ham** |

Observations: (1–2) both models handle typical spam with phone numbers and urgent offers. (3–4) the scratch model flags several
informal, badly spelled or all-caps normal messages as spam; DistilBERT handles them correctly (our interpretation, not tested).
(5) a very short spam message is caught only by DistilBERT. (6) both miss a spam message that reads like a harmless news joke.
(7–8) two of DistilBERT's 4 errors: it flags a bare URL as spam and misses an event-ticket spam message that the scratch model
catches, so the scratch model is not worse on every message.

## 5. Discussion
- **Performance difference:** DistilBERT is better on every metric: accuracy 0.9923 vs 0.9710, macro-F1 0.9824 vs 0.9377, spam precision
  0.9692 vs 0.8378. Recall is close (0.9692 vs 0.9538). The main difference is false alarms: the scratch model wrongly flags 12 normal
  messages as spam, DistilBERT only 2. In total 15 errors vs 4 out of 517.
- **Advantage of pretraining:** DistilBERT already knows word meaning and grammar from large corpora, so fine-tuning only adapts it to
  spam. The scratch model must learn everything from ~4,100 messages; its validation loss was lowest at epoch 6 (0.308) and then rose to
  0.370 at epoch 8, a sign of overfitting. The scratch model is nevertheless strong for a model trained from zero.
- **Cost trade-off:** DistilBERT has ~58× more parameters (66.96M vs 1.15M), takes ~7× longer to train (78.3 s vs 11.3 s) and is ~9× slower
  per message in the batch-32 measurement (2.134 ms vs 0.230 ms), all on the same Tesla T4 GPU. The accuracy gain (+2.1 points) therefore
  has a clear cost.
- **Limitations:** small dataset and a single seed (no confidence intervals; with only 65 spam test messages, 1–2 errors change the
  scores noticeably; re-running the scratch training gave different numbers than an earlier CPU run, showing run-to-run variation);
  strong class imbalance; English-only data from one source; the scratch model has a fixed 6,843-word vocabulary, so unseen words
  become `[UNK]`.

## 6. Conclusion
On this task the pretrained DistilBERT outperforms the Transformer built from scratch (accuracy 99.2% vs 97.1%, 4 vs 15 errors),
mainly because pretraining gives it language knowledge that ~4,100 messages cannot teach a randomly initialised model. The scratch model
is ~58× smaller and ~9× faster per message and still catches most spam, but it produces more false alarms. If accuracy matters most,
DistilBERT is the better choice; if size and speed matter most, the scratch model is a reasonable lightweight option.

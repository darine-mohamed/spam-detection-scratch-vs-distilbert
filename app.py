"""Streamlit app - Scratch Transformer vs fine-tuned DistilBERT (Spam / Ham SMS detection).
Run:  streamlit run app.py"""
import json, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from scratch_model import clean_text, encode_text, load_scratch

BASE = Path(__file__).parent
SCRATCH_DIR = BASE / "artifacts"
DIST_DIR = BASE / "artifacts_distilbert"
DIST_MODEL = DIST_DIR / "distilbert_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "Ham", 1: "Spam"}

st.set_page_config(page_title="Scratch vs Pretrained Transformer", page_icon="📨", layout="wide")

EXAMPLES = {
    "— choose an example —": "",
    "Spam: prize": "Congratulations! You won a free prize. Call now!",
    "Spam: urgent cash": "URGENT! You have won a £1000 cash reward! Text WIN to 80086 now",
    "Spam: free entry": "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry",
    "Ham: meeting": "Hey, are we still meeting today?",
    "Ham: homework": "Can you send me the homework please?",
    "Ham: casual": "Ok lar... Joking wif u oni...",
}


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading scratch Transformer...")
def get_scratch():
    model, vocab = load_scratch(SCRATCH_DIR / "scratch_model.pt", SCRATCH_DIR / "vocab.json", DEVICE)
    predict_scratch("warm up", model, vocab)  # warm-up so the first timing is fair
    return model, vocab


@st.cache_resource(show_spinner="Loading DistilBERT...")
def get_distilbert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(DIST_MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(str(DIST_MODEL)).to(DEVICE).eval()
    predict_distilbert("warm up", model, tok)
    return model, tok


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def predict_scratch(text, model, vocab):
    sync(); t0 = time.perf_counter()
    ids = torch.tensor([encode_text(clean_text(text), vocab)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(ids), dim=1)[0]
    sync(); lat = (time.perf_counter() - t0) * 1000
    c = int(probs.argmax())
    return LABELS[c], float(probs[c]), float(probs[1]), lat


def predict_distilbert(text, model, tok):
    sync(); t0 = time.perf_counter()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1)[0]
    sync(); lat = (time.perf_counter() - t0) * 1000
    c = int(probs.argmax())
    return LABELS[c], float(probs[c]), float(probs[1]), lat


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


scratch_res = read_json(SCRATCH_DIR / "scratch_results.json")
dist_res = read_json(DIST_DIR / "distilbert_results.json")
dist_available = DIST_MODEL.exists()
scratch_available = (SCRATCH_DIR / "scratch_model.pt").exists()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Project")
    st.markdown("**Task:** Spam / Ham SMS detection\n\n**Model A:** Transformer from scratch\n\n"
                "**Model B:** DistilBERT (fine-tuned)")
    st.caption(f"Running on: **{DEVICE.type.upper()}**")
    st.markdown("**Status**")
    st.write(("✅" if scratch_available else "❌") + " Scratch model files")
    st.write(("✅" if dist_available else "❌") + " DistilBERT model files")
    st.write(("✅" if scratch_res else "❌") + " Scratch results JSON")
    st.write(("✅" if dist_res else "❌") + " DistilBERT results JSON")

st.title("📨 Scratch Transformer vs Pretrained DistilBERT")
st.write("Both models classify the same SMS message as **Ham** (normal) or **Spam**. "
         "Type a message, run both, and compare their answers and measured speed.")

tab_try, tab_dash, tab_pred, tab_about = st.tabs(
    ["🔍 Try it", "📊 Comparison dashboard", "🧾 Test predictions", "ℹ️ About"])

# ------------------------------------------------------------------ tab 1
with tab_try:
    example = st.selectbox("Example inputs", list(EXAMPLES.keys()))
    text = st.text_area("SMS message", value=EXAMPLES[example], height=120,
                        placeholder="Type or paste an SMS message here...")
    run = st.button("Run both models", type="primary")

    if run:
        if not text.strip():
            st.warning("Please type a message (or choose an example) first.")
        else:
            c1, c2 = st.columns(2)
            for col, name, key in [(c1, "Model A – Scratch Transformer", "a"),
                                   (c2, "Model B – DistilBERT (fine-tuned)", "b")]:
                with col:
                    st.subheader(name)
                    try:
                        if key == "a":
                            if not scratch_available:
                                raise FileNotFoundError("artifacts/scratch_model.pt not found")
                            m, v = get_scratch()
                            label, conf, p_spam, lat = predict_scratch(text, m, v)
                        else:
                            if not dist_available:
                                raise FileNotFoundError(
                                    "artifacts_distilbert/distilbert_model not found - "
                                    "run Part 2 of the notebook and copy its output folder here")
                            m, t = get_distilbert()
                            label, conf, p_spam, lat = predict_distilbert(text, m, t)
                        (st.error if label == "Spam" else st.success)(f"Prediction: **{label}**")
                        st.metric("Confidence", f"{conf*100:.2f}%")
                        st.metric("Spam probability", f"{p_spam*100:.2f}%")
                        st.metric("Prediction latency", f"{lat:.2f} ms")
                        st.caption("Latency = preprocessing/tokenization + forward pass, one message.")
                    except Exception as e:
                        st.error(f"Could not run this model: {e}")

# ------------------------------------------------------------------ tab 2
def cm_fig(cm, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Ham", "Spam"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Ham", "Spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


with tab_dash:
    if not scratch_res:
        st.error("artifacts/scratch_results.json not found.")
    else:
        results = {"Scratch Transformer": scratch_res}
        if dist_res:
            results["DistilBERT (fine-tuned)"] = dist_res
        else:
            st.warning("DistilBERT results not found (artifacts_distilbert/distilbert_results.json). "
                       "Showing the scratch model only.")

        rows = [("Accuracy", "accuracy", "{:.4f}"), ("Macro F1", "macro_f1", "{:.4f}"),
                ("Spam precision", "spam_precision", "{:.4f}"), ("Spam recall", "spam_recall", "{:.4f}"),
                ("Spam F1", "spam_f1", "{:.4f}"), ("Parameters", "parameters", "{:,.0f}"),
                ("Training / fine-tuning time (s)", "training_time_seconds", "{:.1f}"),
                ("Avg latency per message, batch=32 (ms)", "average_latency_ms", "{:.3f}")]
        table = pd.DataFrame({"Metric": [r[0] for r in rows]})
        for name, r in results.items():
            table[name] = [fmt.format(r[k]) if k in r else "n/a" for _, k, fmt in rows]
        extra = [("Single-message latency incl. tokenization (ms)", "single_message_latency_ms")]
        for label, k in extra:
            table.loc[len(table)] = [label] + [f"{r[k]:.2f}" if k in r else "n/a" for r in results.values()]
        st.subheader("Side-by-side test results")
        st.caption(f"Same held-out test set for both models: {scratch_res['test_size']} messages "
                   f"(seed {scratch_res['seed']}).")
        st.dataframe(table, hide_index=True)

        st.subheader("Quality metrics")
        keys = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("spam_precision", "Spam P"),
                ("spam_recall", "Spam R"), ("spam_f1", "Spam F1")]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        n_models = len(results)
        w = 0.8 / n_models
        x = np.arange(len(keys))
        for i, (name, r) in enumerate(results.items()):
            vals = [r[k] for k, _ in keys]
            bars = ax.bar(x + (i - (n_models - 1) / 2) * w, vals, w, label=name)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys])
        ax.set_ylim(0, 1.1); ax.set_ylabel("Score"); ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Cost: parameters, training time, latency")
        cc = st.columns(3)
        for col, (k, title) in zip(cc, [("parameters", "Parameters"),
                                        ("training_time_seconds", "Training time (s)"),
                                        ("average_latency_ms", "Latency / message (ms)")]):
            with col:
                st.caption(title)
                st.bar_chart(pd.DataFrame({"value": [r[k] for r in results.values()]},
                                          index=list(results.keys())))

        st.subheader("Confusion matrices (test set)")
        cols = st.columns(len(results))
        for col, (name, r) in zip(cols, results.items()):
            with col:
                st.pyplot(cm_fig(r["confusion_matrix"], name))

        if dist_res is None:
            st.info("Run Part 2 of the notebook to generate the DistilBERT results.")

# ------------------------------------------------------------------ tab 3
with tab_pred:
    sp = SCRATCH_DIR / "scratch_test_predictions.csv"
    dp = DIST_DIR / "distilbert_test_predictions.csv"
    if not sp.exists():
        st.error("Scratch test predictions not found.")
    else:
        df = pd.read_csv(sp)[["message", "label", "predicted_label"]].rename(
            columns={"label": "true", "predicted_label": "scratch"})
        if dp.exists():
            d2 = pd.read_csv(dp)
            if len(d2) == len(df) and (d2["message"].values == pd.read_csv(sp)["message"].values).all():
                df["distilbert"] = d2["predicted_label"].values
            else:
                st.warning("Test sets of the two models do not match.")
        else:
            st.info("DistilBERT test predictions not found yet.")
        only_diff = False
        if "distilbert" in df:
            only_diff = st.checkbox("Show only messages where the models disagree")
            if only_diff:
                df = df[df["scratch"] != df["distilbert"]]
        only_err = st.checkbox("Show only messages where at least one model is wrong")
        if only_err:
            wrong = df["scratch"].str.lower() != df["true"]
            if "distilbert" in df:
                wrong = wrong | (df["distilbert"].str.lower() != df["true"])
            df = df[wrong]
        st.caption(f"{len(df)} messages")
        st.dataframe(df, hide_index=True)

# ------------------------------------------------------------------ tab 4
with tab_about:
    st.markdown("""
### Task
**Spam / ham detection** for SMS messages. Input: one message. Output: `Ham` or `Spam`.

### Dataset
SMS Spam Collection (Almeida, Gómez Hidalgo & Yamakami, 2011) – UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
5,574 messages → 5,168 after removing duplicates/empty. Split 80/10/10 (stratified, seed 42):
4,134 train / 517 validation / 517 test. Both models use exactly the same rows.

### Model A – Transformer from scratch
Own implementation of multi-head self-attention, positional embeddings, feed-forward layers, residual
connections, LayerNorm and padding masks. 2 encoder layers, d_model 128, 4 heads, FF 256, ~1.15M
parameters, random initialisation, masked mean pooling + linear classifier.

### Model B – Pretrained Transformer
`distilbert-base-uncased` (Hugging Face), fine-tuned on the same training set with class-weighted loss.

### Notes
- Latency numbers are measured on the machine running this app and change with hardware.
- Test-set metrics come from the saved results files produced by the notebook (never tuned on test data).
""")

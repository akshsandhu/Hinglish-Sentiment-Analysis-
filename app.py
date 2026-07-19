"""
app.py
──────
Streamlit web app for Hinglish Sentiment Analysis.
Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

sys.path.append("src")
from predict import load_model, predict_sentiment, ID2LABEL
from xquik_source import XquikSourceError, fetch_tweet_texts

st.set_page_config(
    page_title = "Hinglish Sentiment Analyzer",
    page_icon  = "🇮🇳",
    layout     = "centered",
)

@st.cache_resource
def get_model():
    try:
        return load_model("models/indicbert-hinglish")
    except FileNotFoundError:
        return None, None

tokenizer, model = get_model()
model_loaded     = model is not None

st.title("🇮🇳 Hinglish Sentiment Analyzer")
st.caption("Fine-tuned IndicBERT for Hindi, English & Hinglish text")

if not model_loaded:
    st.warning(
        "⚠️ Model not found. Train it first in Google Colab (src/train_colab.py), "
        "then place the folder at: models/indicbert-hinglish/"
    )
    st.info("Running in DEMO mode - UI is fully functional, predictions are random.")

st.divider()

EXAMPLES = [
    "Bhut accha product hai! Delivery bhi fast thi. Ekdum sahi.",
    "Bekar quality hai, bilkul waste of money. Mat kharido.",
    "Theek thaak hai. Average product, not bad not great.",
    "Zabardast! Best phone in this price range. Highly recommend.",
    "Bahut kharab nikla. Photo aur actual mein bahut fark tha.",
    "Acha hai par thoda mehnga lagta hai price ke hisaab se.",
]

st.write("**Try an example:**")
col1, col2 = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    btn_col = col1 if i % 2 == 0 else col2
    if btn_col.button(ex[:45] + "...", key=f"ex_{i}"):
        st.session_state["input_text"] = ex

st.divider()
st.write("**Or load recent X text with Xquik:**")

xquik_query = st.text_input(
    label       = "X search query",
    placeholder = 'product review lang:hi OR "hinglish"',
)

if st.button("Load X Results", use_container_width=True, disabled=(len(xquik_query.strip()) < 2)):
    try:
        xquik_results = fetch_tweet_texts(xquik_query, limit=5)
        st.session_state["xquik_results"] = xquik_results
        st.session_state["input_text"] = xquik_results[0]
        st.success(f"Loaded {len(xquik_results)} X results.")
    except XquikSourceError as error:
        st.session_state.pop("xquik_results", None)
        st.warning(str(error))

if st.session_state.get("xquik_results"):
    selected_xquik_text = st.selectbox(
        "Loaded X text",
        options=st.session_state["xquik_results"],
        format_func=lambda value: value[:90] + ("..." if len(value) > 90 else ""),
    )
    if st.button("Use Selected X Text", use_container_width=True):
        st.session_state["input_text"] = selected_xquik_text

st.divider()
st.write("**Or enter your own Hinglish/Hindi text:**")

text = st.text_area(
    label            = "Text input",
    value            = st.session_state.get("input_text", ""),
    height           = 120,
    placeholder      = "Yeh product bahut accha hai, delivery fast thi...",
    label_visibility = "collapsed",
)

st.caption(f"{len(text)} characters")

analyze_btn = st.button(
    "🔍 Analyze Sentiment",
    type                = "primary",
    use_container_width = True,
    disabled            = (len(text) < 3),
)

if analyze_btn and text.strip():
    st.divider()
    with st.spinner("Analyzing..."):
        if model_loaded:
            result = predict_sentiment(text, tokenizer, model)
        else:
            import random
            label_id = random.randint(0, 2)
            result = {
                "label"      : ID2LABEL[label_id],
                "label_id"   : label_id,
                "confidence" : round(random.uniform(0.70, 0.96), 4),
                "probs"      : [0.33, 0.33, 0.34],
                "emoji"      : ["😊", "😞", "😐"][label_id],
                "clean_text" : text,
            }

    label = result["label"]
    conf  = result["confidence"]
    emoji = result["emoji"]
    probs = result["probs"]

    COLOR_MAP = {
        "Positive": ("#EAF3DE", "#27500A", "#639922"),
        "Negative": ("#FCEBEB", "#791F1F", "#E24B4A"),
        "Neutral" : ("#FAEEDA", "#633806", "#EF9F27"),
    }
    bg, fg, border = COLOR_MAP[label]

    st.markdown(f"""
        <div style="background:{bg};border:1.5px solid {border};border-radius:12px;
                    padding:1.2rem 1.5rem;text-align:center;margin-bottom:1rem">
            <div style="font-size:48px;margin-bottom:8px">{emoji}</div>
            <div style="font-size:24px;font-weight:600;color:{fg}">{label}</div>
            <div style="font-size:14px;color:{fg};opacity:0.8;margin-top:4px">
                Confidence: {conf*100:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Sentiment",  label)
    m2.metric("Confidence", f"{conf*100:.1f}%")
    m3.metric("Model",      "IndicBERT")

    st.write("")
    st.write("**Score breakdown:**")
    for lbl, prob, color in zip(
        ["Positive", "Negative", "Neutral"],
        probs,
        ["#639922", "#E24B4A", "#EF9F27"]
    ):
        c1, c2 = st.columns([1, 3])
        c1.write(lbl)
        c2.progress(float(prob), text=f"{prob*100:.1f}%")

    with st.expander("See cleaned text (after preprocessing)"):
        st.code(result["clean_text"])

    st.divider()
    st.write("**Aspects detected:**")
    keywords = {
        "Quality"  : ["quality", "material", "build", "sahi", "kharab", "badhiya"],
        "Delivery" : ["delivery", "shipping", "fast", "jaldi", "late"],
        "Price"    : ["price", "paisa", "mehnga", "sasta", "value"],
        "Service"  : ["service", "support", "return", "refund"],
    }
    detected = [a for a, words in keywords.items() if any(w in text.lower() for w in words)]
    if detected:
        cols = st.columns(len(detected))
        for i, aspect in enumerate(detected):
            cols[i].success(f"✓ {aspect}")
    else:
        st.info("No specific aspect keywords detected.")

st.divider()
st.caption("IndicBERT fine-tuned on SemEval 2020 Task 9 · Hinglish Sentiment Analysis")
c1, c2, c3 = st.columns(3)
c1.metric("Model",     "IndicBERT")
c2.metric("Accuracy",  "88%")
c3.metric("Languages", "Hindi + Hinglish")

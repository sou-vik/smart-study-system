import streamlit as st
import nltk
import random

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Tutor AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK DATA
# ─────────────────────────────────────────────────────────────────────────────

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — White + Maroon Premium Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Variables ── */
:root {
    --maroon:         #006400;               /* Dark Green */
    --maroon-mid:     #228B22;               /* Forest Green */
    --maroon-pale:    rgba(0, 100, 0, 0.06);
    --maroon-glass:   rgba(0, 100, 0, 0.11);
    --maroon-hover:   rgba(0, 100, 0, 0.88);
    --white:         #ffffff;
    --off-white:     #fafafa;
    --surface:       #f5f3f3;
    --text-dark:     #1a1a1a;
    --text-mid:      #444444;
    --text-soft:     #888888;
    --border:        rgba(128, 0, 0, 0.18);
    --border-strong: rgba(128, 0, 0, 0.38);
    --shadow-sm:     0 2px 12px rgba(128, 0, 0, 0.06);
    --shadow-md:     0 6px 28px rgba(128, 0, 0, 0.11);
    --shadow-lg:     0 14px 44px rgba(128, 0, 0, 0.15);
    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     22px;
    --ease:          cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--off-white: #fff7f7;) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-dark) !important;
}

/* thin maroon top bar */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0 0 auto 0;
    height: 3px;
    background: #006400;
    z-index: 9999;
}

.main .block-container {
    padding: 2.8rem 3.2rem 3rem !important;
    max-width: 1140px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 18px rgba(128, 0, 0, 0.05) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.3rem !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--maroon) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.55rem !important;
    margin-bottom: 1.1rem !important;
}

/* Sidebar info / success boxes → clean cards */
[data-testid="stSidebar"] .stAlert {
    background: var(--off-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease),
                background 0.25s var(--ease), border-color 0.25s var(--ease) !important;
    backdrop-filter: none !important;
}

[data-testid="stSidebar"] .stAlert:hover {
    transform: translateX(4px) !important;
    background: var(--maroon-pale) !important;
    border-color: var(--border-strong) !important;
    box-shadow: var(--shadow-md) !important;
}

[data-testid="stSidebar"] .stAlert p,
[data-testid="stSidebar"] .stAlert div {
    color: var(--text-mid) !important;
    font-size: 0.87rem !important;
    line-height: 1.75 !important;
}

/* ── Header ── */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: var(--maroon) !important;
    text-align: center !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    margin-bottom: 0.15rem !important;
}

h2 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    color: var(--maroon) !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 1rem !important;
}

h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: var(--maroon) !important;
}

h4 {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-soft) !important;
    font-weight: 300 !important;
    font-size: 0.97rem !important;
    text-align: center !important;
    letter-spacing: 0.02em !important;
}

/* caption / description text */
[data-testid="stCaptionContainer"] p {
    text-align: center !important;
    color: var(--text-soft) !important;
    font-size: 0.84rem !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.6rem 0 !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"],
[data-testid="stMetric"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.4rem 1.6rem !important;
    box-shadow: var(--shadow-sm) !important;
    text-align: center !important;
    transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease),
                border-color 0.25s var(--ease) !important;
}

[data-testid="metric-container"]:hover,
[data-testid="stMetric"]:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 32px rgba(128, 0, 0, 0.13) !important;
    border-color: var(--border-strong) !important;
}

[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: var(--text-soft) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: var(--maroon) !important;
}

/* ── Text Area ── */
.stTextArea textarea,
[data-testid="stTextArea"] textarea {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-dark) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 1rem 1.1rem !important;
    transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease) !important;
    box-shadow: var(--shadow-sm) !important;
}

.stTextArea textarea:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--maroon) !important;
    box-shadow: 0 0 0 3px rgba(128, 0, 0, 0.09) !important;
    outline: none !important;
}

.stTextArea label,
[data-testid="stTextArea"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text-soft) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1.5px solid var(--maroon) !important;
    color: var(--maroon) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    border-radius: var(--radius-sm) !important;
    width: 100% !important;
    height: 52px !important;
    transition: background 0.22s var(--ease), color 0.22s var(--ease),
                transform 0.22s var(--ease), box-shadow 0.22s var(--ease) !important;
    cursor: pointer !important;
    box-shadow: none !important;
}

.stButton > button:hover {
    background: var(--maroon-hover) !important;
    color: var(--white) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

.stButton > button:active {
    background: var(--maroon) !important;
    color: var(--white) !important;
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1.5px solid var(--border) !important;
    gap: 0 !important;
    padding-bottom: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-soft) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    padding: 0.65rem 1.4rem !important;
    margin-bottom: -1.5px !important;
    border-radius: 0 !important;
    transition: color 0.22s var(--ease), background 0.22s var(--ease) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--maroon) !important;
    background: var(--maroon-pale) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--maroon) !important;
    border-bottom: 2px solid var(--maroon) !important;
    font-weight: 700 !important;
    background: transparent !important;
}

/* ── Success / Warning boxes ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 0.88rem !important;
}

div[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(128, 0, 0, 0.05) !important;
    border: 1px solid var(--border) !important;
    color: var(--maroon) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--maroon); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Fade-in stagger ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

.main .block-container > * { animation: fadeInUp 0.45s ease both; }
.main .block-container > *:nth-child(1) { animation-delay: 0.00s; }
.main .block-container > *:nth-child(2) { animation-delay: 0.07s; }
.main .block-container > *:nth-child(3) { animation-delay: 0.14s; }
.main .block-container > *:nth-child(4) { animation-delay: 0.21s; }
.main .block-container > *:nth-child(5) { animation-delay: 0.28s; }
.main .block-container > *:nth-child(6) { animation-delay: 0.35s; }
.main .block-container > *:nth-child(7) { animation-delay: 0.42s; }
            
/* Top Streamlit Header */
[data-testid="stHeader"] {
    background: #006400 !important;
}

/* Main toolbar area */
[data-testid="stToolbar"] {
    background: #800000 !important;
}

/* Optional: Remove transparency */
header {
    background: #800000 !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:

    st.title("About Project")

    st.info(
        "Smart Tutor AI\n\n"
        "Features\n"
        "  Note Summarization\n"
        "  Keyword Extraction\n"
        "  Question Generation\n"
        "  MCQ Generation\n"
        "  Offline Processing"
    )

    st.success("SDG 4 — Quality Education")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1>Smart Tutor AI</h1>"
    "<hr style='width:72px;height:2px;background:#800000;"
    "border:none;border-radius:2px;margin:0.5rem auto 0.7rem;'>"
    "<h4>AI-Inspired Learning Assistant for SDG 4 &ndash; Quality Education</h4>",
    unsafe_allow_html=True,
)

st.caption(
    "An NLP-based educational assistant that helps students summarize notes, "
    "extract keywords, generate questions, and create MCQs."
)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
col1.metric("Features", "4")
col2.metric("Mode", "Offline")
col3.metric("SDG", "4")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────

notes = st.text_area(
    "Paste your study notes here",
    height=300,
    placeholder="Paste notes here...",
)

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — MCQ GENERATION  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def generate_mcqs(notes):
    sentences  = sent_tokenize(notes)
    mcqs       = []

    for sentence in sentences:
        words          = word_tokenize(sentence)
        important_words = [w for w in words if len(w) > 5 and w.isalpha()]

        if not important_words:
            continue

        answer   = important_words[0]
        question = sentence.replace(answer, "______", 1)

        distractors = [
            "Database", "Network", "Compiler",
            "Operating System", "Algorithm", "Software",
        ]
        options = random.sample(distractors, 3)
        options.append(answer)
        random.shuffle(options)

        mcqs.append((question, options, answer))

        if len(mcqs) == 5:
            break

    return mcqs

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────────────────────────────────────

if st.button("Generate Study Material"):

    if not notes.strip():
        st.warning("Please enter some study notes.")

    else:
        # ── BACKEND: keyword extraction (unchanged) ─────────────────────
        words         = word_tokenize(notes.lower())
        stop_words    = set(stopwords.words("english"))
        filtered      = [w for w in words if w.isalnum() and w not in stop_words]
        freq          = Counter(filtered)
        keywords      = [word for word, _ in freq.most_common(10)]

        # ── TABS ────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Summary", "Keywords", "Questions", "MCQs"]
        )

        # ── SUMMARY ─────────────────────────────────────────────────────
        with tab1:
            st.subheader("Summary")

            # BACKEND: sumy LSA summarizer (unchanged)
            parser     = PlaintextParser.from_string(notes, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary    = summarizer(parser.document, 3)

            for sentence in summary:
                st.markdown(
                    f"<div style='"
                    f"background:#fff;border:1px solid rgba(128,0,0,0.18);"
                    f"border-radius:12px;padding:1rem 1.3rem;margin-bottom:0.7rem;"
                    f"box-shadow:0 2px 10px rgba(128,0,0,0.06);"
                    f"font-size:0.92rem;color:#333;line-height:1.7;"
                    f"transition:box-shadow 0.25s ease,border-color 0.25s ease;'>"
                    f"{sentence}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── KEYWORDS ────────────────────────────────────────────────────
        with tab2:
            st.subheader("Important Keywords")

            badges = " ".join(
                f"<span style='"
                f"display:inline-block;background:transparent;"
                f"border:1.5px solid #800000;color:#800000;"
                f"border-radius:50px;padding:0.22rem 0.85rem;"
                f"font-size:0.82rem;font-weight:500;letter-spacing:0.04em;"
                f"margin:0.25rem;transition:all 0.22s ease;"
                f"font-family:DM Sans,sans-serif;'>"
                f"{kw}</span>"
                for kw in keywords
            )
            st.markdown(
                f"<div style='padding:1.2rem 0;'>{badges}</div>",
                unsafe_allow_html=True,
            )

        # ── QUESTIONS ────────────────────────────────────────────────────
        with tab3:
            st.subheader("Generated Questions")

            for i, keyword in enumerate(keywords[:5], start=1):
                for q_text in [
                    f"Explain the concept of <strong>{keyword}</strong>.",
                    f"What are the applications of <strong>{keyword}</strong>?",
                ]:
                    st.markdown(
                        f"<div style='"
                        f"background:#fff;border:1px solid rgba(128,0,0,0.18);"
                        f"border-radius:12px;padding:0.9rem 1.3rem;"
                        f"margin-bottom:0.55rem;"
                        f"box-shadow:0 2px 10px rgba(128,0,0,0.05);"
                        f"font-size:0.9rem;color:#333;line-height:1.65;'>"
                        f"<span style='color:#800000;font-weight:600;"
                        f"font-family:Cormorant Garamond,serif;margin-right:0.5rem;'>"
                        f"Q{i}.</span> {q_text}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── MCQs ─────────────────────────────────────────────────────────
        with tab4:
            st.subheader("Generated MCQs")

            # BACKEND: generate_mcqs (unchanged)
            mcqs = generate_mcqs(notes)

            if not mcqs:
                st.warning("Not enough content to generate MCQs.")

            for i, (question, options, answer) in enumerate(mcqs, start=1):
                options_html = ""
                for j, opt in enumerate(options):
                    is_answer = opt == answer
                    bg     = "rgba(0, 100, 0, 0.08)" if is_answer else "transparent"
                    border = "rgba(0, 100, 0, 0.45)"  if is_answer else "rgba(128, 0, 0, 0.12)"
                    weight = "600" if is_answer else "400"
                    color  = "#006400" if is_answer else "#8A4A4A"
                    options_html += (
                        f"<div style='background:{bg};border:1px solid {border};"
                        f"border-radius:8px;padding:0.48rem 0.9rem;"
                        f"margin:0.3rem 0;font-size:0.87rem;"
                        f"font-weight:{weight};color:{color};'>"
                        f"{chr(65+j)})  {opt}"
                        f"</div>"
                    )

                st.markdown(
                    f"<div style='background:#fff;border:1px solid rgba(128,0,0,0.18);"
                    f"border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1rem;"
                    f"box-shadow:0 2px 12px rgba(128,0,0,0.07);'>"
                    f"<div style='font-family:Cormorant Garamond,serif;"
                    f"color:#800000;font-size:1.05rem;font-weight:700;"
                    f"margin-bottom:0.7rem;'>Q{i}.  {question}</div>"
                    f"{options_html}"
                    f"<div style='margin-top:0.75rem;padding:0.4rem 0.8rem;"
                    f"background:rgba(128,0,0,0.05);border-radius:6px;"
                    f"font-size:0.8rem;font-weight:600;color:#800000;"
                    f"letter-spacing:0.05em;text-transform:uppercase;'>"
                    f"Answer &nbsp; {answer}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
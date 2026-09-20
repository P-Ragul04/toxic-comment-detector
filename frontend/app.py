"""
Streamlit demo UI for the toxic comment detector API.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud (share.streamlit.io) by
connecting this repo and pointing it at frontend/app.py.
"""

import time

import requests
import streamlit as st

DEFAULT_API_URL = "https://toxic-comment-detector-z8q2.onrender.com"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.set_page_config(page_title="Toxic Comment Detector", page_icon="🛡️", layout="centered")

# ---------------------------------------------------------------------------
# Theme: deep space background, glass panels, violet -> cyan glow accent.
# This is the one bold visual choice; everything else stays quiet and disciplined.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg-deep: #05060f;
        --panel: rgba(18, 20, 38, 0.55);
        --panel-border: rgba(140, 130, 255, 0.18);
        --accent-a: #8b7cff;
        --accent-b: #2fd6ff;
        --text-primary: #e9eaf6;
        --text-muted: #9aa0c4;
        --danger: #ff6b8b;
        --safe: #35e8b8;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    html {
        color-scheme: dark;
    }

    .stApp {
        background:
            radial-gradient(ellipse 120% 80% at 20% -10%, rgba(139, 124, 255, 0.18), transparent 60%),
            radial-gradient(ellipse 100% 60% at 100% 0%, rgba(47, 214, 255, 0.12), transparent 55%),
            radial-gradient(1.5px 1.5px at 8% 12%, rgba(255,255,255,0.65) 100%, transparent),
            radial-gradient(1px 1px at 18% 28%, rgba(255,255,255,0.4) 100%, transparent),
            radial-gradient(1.5px 1.5px at 32% 8%, rgba(255,255,255,0.5) 100%, transparent),
            radial-gradient(1px 1px at 48% 22%, rgba(255,255,255,0.35) 100%, transparent),
            radial-gradient(1.5px 1.5px at 62% 6%, rgba(255,255,255,0.55) 100%, transparent),
            radial-gradient(1px 1px at 75% 18%, rgba(255,255,255,0.4) 100%, transparent),
            radial-gradient(1.5px 1.5px at 88% 10%, rgba(255,255,255,0.5) 100%, transparent),
            radial-gradient(1px 1px at 95% 30%, rgba(255,255,255,0.35) 100%, transparent),
            radial-gradient(1px 1px at 10% 45%, rgba(255,255,255,0.3) 100%, transparent),
            radial-gradient(1.5px 1.5px at 25% 60%, rgba(255,255,255,0.5) 100%, transparent),
            radial-gradient(1px 1px at 40% 75%, rgba(255,255,255,0.35) 100%, transparent),
            radial-gradient(1.5px 1.5px at 55% 55%, rgba(255,255,255,0.45) 100%, transparent),
            radial-gradient(1px 1px at 68% 88%, rgba(255,255,255,0.3) 100%, transparent),
            radial-gradient(1.5px 1.5px at 80% 65%, rgba(255,255,255,0.5) 100%, transparent),
            radial-gradient(1px 1px at 92% 80%, rgba(255,255,255,0.35) 100%, transparent),
            radial-gradient(1.5px 1.5px at 5% 85%, rgba(255,255,255,0.4) 100%, transparent),
            radial-gradient(1px 1px at 15% 95%, rgba(255,255,255,0.3) 100%, transparent),
            var(--bg-deep);
        color: var(--text-primary);
    }

    /* Streamlit's native chrome (header bar, toolbar, top decoration
       stripe) is theme-driven separately from .stApp - force it dark
       regardless of the visitor's OS/browser light-mode preference. */
    header[data-testid="stHeader"] {
        background: rgba(5, 6, 15, 0.0) !important;
    }
    div[data-testid="stDecoration"] {
        background: linear-gradient(90deg, var(--accent-a), var(--accent-b)) !important;
    }
    div[data-testid="stToolbar"] {
        background: transparent !important;
    }
    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(ellipse 100% 60% at 50% 0%, rgba(139, 124, 255, 0.14), transparent 60%),
            radial-gradient(1.5px 1.5px at 15% 10%, rgba(255,255,255,0.5) 100%, transparent),
            radial-gradient(1px 1px at 40% 22%, rgba(255,255,255,0.35) 100%, transparent),
            radial-gradient(1.5px 1.5px at 70% 8%, rgba(255,255,255,0.45) 100%, transparent),
            radial-gradient(1px 1px at 85% 30%, rgba(255,255,255,0.3) 100%, transparent),
            radial-gradient(1.5px 1.5px at 25% 45%, rgba(255,255,255,0.4) 100%, transparent),
            radial-gradient(1px 1px at 55% 55%, rgba(255,255,255,0.3) 100%, transparent),
            radial-gradient(1.5px 1.5px at 80% 65%, rgba(255,255,255,0.45) 100%, transparent),
            radial-gradient(1px 1px at 12% 75%, rgba(255,255,255,0.3) 100%, transparent),
            radial-gradient(1.5px 1.5px at 45% 85%, rgba(255,255,255,0.4) 100%, transparent),
            radial-gradient(1px 1px at 90% 90%, rgba(255,255,255,0.3) 100%, transparent),
            var(--bg-deep) !important;
        border-right: 1px solid var(--panel-border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.05rem !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] p {
        color: var(--text-muted) !important;
    }
    /* Force Inter/Space Grotesk everywhere, overriding Streamlit's
       own font-family rules which otherwise win on specificity */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp button, .stMarkdown, .stCaption {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp h1, .stApp h2, .stApp h3,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Custom status pill, replacing the default st.success box which
       clashes with the glass/glow theme */
    .status-pill {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        background: rgba(53, 232, 184, 0.07);
        border: 1px solid rgba(53, 232, 184, 0.3);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        font-size: 0.88rem;
        color: #8ff4dc;
        backdrop-filter: blur(10px);
        margin-bottom: 0.8rem;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--safe);
        box-shadow: 0 0 8px var(--safe), 0 0 16px var(--safe);
        flex-shrink: 0;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        background: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
    }
    /* The blanket font override above also hits Streamlit's icon font
       (icons are rendered via font ligatures, e.g. "arrow_right" as
       literal text becomes an arrow glyph only with the right font) -
       restore it specifically so expander/chevron icons render correctly */
    [data-testid="stIconMaterial"],
    [class*="material-icons"],
    span[data-testid*="Icon"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    /* GitHub link, styled as a glass pill to match the rest of the sidebar */
    .github-link {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        font-size: 0.85rem;
        color: var(--text-primary) !important;
        text-decoration: none !important;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .github-link:hover {
        border-color: var(--accent-b);
        box-shadow: 0 0 16px rgba(47, 214, 255, 0.15);
    }
    .github-link svg {
        flex-shrink: 0;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* Hero title */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        line-height: 1.1;
        background: linear-gradient(120deg, var(--accent-a), var(--accent-b));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        color: var(--text-muted);
        font-size: 1.02rem;
        max-width: 640px;
        line-height: 1.55;
        margin-bottom: 1.8rem;
    }

    /* Glass panel wrapper for the input area */
    div[data-testid="stTextArea"] textarea {
        background: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        backdrop-filter: blur(12px);
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-b) !important;
        box-shadow: 0 0 0 1px var(--accent-b), 0 0 24px rgba(47, 214, 255, 0.15) !important;
    }
    div[data-testid="stTextArea"] label {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
    }

    /* Primary button */
    button[kind="primary"] {
        background: linear-gradient(120deg, var(--accent-a), var(--accent-b)) !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        box-shadow: 0 0 22px rgba(139, 124, 255, 0.35);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 0 32px rgba(47, 214, 255, 0.45);
        transform: translateY(-1px);
    }

    /* Result verdict banner */
    .verdict {
        border-radius: 14px;
        padding: 1rem 1.3rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 1.2rem 0 0.4rem 0;
        border: 1px solid;
        backdrop-filter: blur(12px);
    }
    .verdict.flagged {
        background: rgba(255, 107, 139, 0.08);
        border-color: rgba(255, 107, 139, 0.35);
        color: #ffb3c4;
        box-shadow: 0 0 24px rgba(255, 107, 139, 0.12);
    }
    .verdict.safe {
        background: rgba(53, 232, 184, 0.08);
        border-color: rgba(53, 232, 184, 0.35);
        color: #8ff4dc;
        box-shadow: 0 0 24px rgba(53, 232, 184, 0.12);
    }

    .response-time {
        color: var(--text-muted);
        font-size: 0.82rem;
        margin-bottom: 1.4rem;
    }

    .breakdown-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--text-primary);
        margin: 1.6rem 0 1rem 0;
    }

    /* Per-label rows, built as plain HTML for full styling control */
    .label-row {
        margin-bottom: 0.85rem;
    }
    .label-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }
    .label-name {
        color: var(--text-primary);
        font-weight: 500;
    }
    .label-flag {
        color: var(--danger);
        font-size: 0.78rem;
        margin-left: 0.4rem;
        font-weight: 600;
    }
    .label-pct {
        color: var(--text-muted);
        font-variant-numeric: tabular-nums;
    }
    .bar-track {
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
    .bar-fill.flagged {
        background: linear-gradient(90deg, #ff6b8b, #ff9770);
        box-shadow: 0 0 10px rgba(255, 107, 139, 0.5);
    }
    .bar-fill.clear {
        background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
        opacity: 0.55;
    }

    .footnote {
        color: var(--text-muted);
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: 2rem;
    }

    hr {
        border-color: var(--panel-border) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.markdown(
        '<div class="status-pill"><span class="status-dot"></span>Connected to live API</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This demo calls a live FastAPI service (DistilBERT, fine-tuned "
        "on the Jigsaw Toxic Comment dataset, served via ONNX Runtime)."
    )

    with st.expander("Advanced"):
        api_url = st.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")

    st.markdown(
        """
        <a href="https://github.com/P-Ragul04/toxic-comment-detector" target="_blank" class="github-link">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07
                -1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82
                a7.6 7.6 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            View source on GitHub
        </a>
        """,
        unsafe_allow_html=True,
    )

# --- Hero ---
st.markdown('<div class="hero-title">Toxic Comment Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A fine-tuned DistilBERT model scores text across six categories '
    '— toxic, severe toxic, obscene, threat, insult, identity hate — and flags anything that '
    'crosses a tuned decision threshold. Enter a comment below to see it in action.</div>',
    unsafe_allow_html=True,
)

text = st.text_area(
    "Comment text",
    placeholder="Type or paste a comment here...",
    height=120,
    label_visibility="collapsed",
)

submit = st.button("Analyze", type="primary", use_container_width=False)


def call_api(api_url: str, text: str, max_retries: int = 2):
    """Call the /predict endpoint. Free-tier hosts (Render) spin down
    after inactivity, so the first request after idle time can take
    ~50s to wake the service - we show a friendly message and retry
    rather than just showing a raw timeout error."""
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{api_url}/predict", json={"text": text}, timeout=70
            )
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                continue
            return None, "Request timed out. The API might be waking up from sleep - try again in a moment."
        except requests.exceptions.ConnectionError:
            return None, "Couldn't reach the API. Check the API URL in the sidebar."
        except requests.exceptions.HTTPError as e:
            return None, f"API returned an error: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"
    return None, "Failed after retries."


if submit:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner(
            "Analyzing... (if the API has been idle, this can take up to 50s to wake up)"
        ):
            start = time.time()
            result, error = call_api(api_url, text)
            elapsed = time.time() - start

        if error:
            st.error(error)
        else:
            verdict_class = "flagged" if result["is_toxic"] else "safe"
            verdict_text = "⚠ Flagged as toxic" if result["is_toxic"] else "✓ Not flagged"
            st.markdown(
                f'<div class="verdict {verdict_class}">{verdict_text}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="response-time">Response time: {elapsed:.1f}s</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="breakdown-heading">Per-label breakdown</div>', unsafe_allow_html=True)
            scores_by_label = {s["label"]: s for s in result["scores"]}

            rows_html = ""
            for label in LABELS:
                score = scores_by_label[label]
                prob = score["probability"]
                flagged = score["flagged"]
                label_display = label.replace("_", " ").title()
                flag_span = '<span class="label-flag">FLAGGED</span>' if flagged else ""
                bar_class = "flagged" if flagged else "clear"
                rows_html += f"""
                <div class="label-row">
                    <div class="label-header">
                        <span class="label-name">{label_display}{flag_span}</span>
                        <span class="label-pct">{prob:.1%}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill {bar_class}" style="width:{prob*100:.1f}%"></div>
                    </div>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)

            with st.expander("Raw JSON response"):
                st.json(result)

st.divider()
st.markdown(
    '<div class="footnote">Note: This is a portfolio/demo project, not a production '
    "moderation tool. The model can make mistakes, especially on rare categories like "
    "'threat' and 'severe_toxic' where training data was limited.</div>",
    unsafe_allow_html=True,
)

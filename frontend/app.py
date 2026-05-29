"""
LexiRAG Streamlit Frontend.
Run with: streamlit run frontend/app.py
"""
import uuid
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexiRAG — Indian Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
    .source-box { background: #f8f9fa; border-left: 3px solid #4a90d9;
                  padding: 0.6rem 1rem; border-radius: 4px; margin: 0.3rem 0;
                  font-size: 0.85rem; color: #333; }
    .answer-box { background: #f0f7ff; border-radius: 8px;
                  padding: 1rem 1.2rem; margin: 0.5rem 0; }
    .stChatMessage { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ───────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ LexiRAG")
    st.caption("AI-powered Indian Legal Assistant")
    st.divider()

    # API health
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        db_chunks = health.get("database", {}).get("document_chunks", 0)
        st.success(f"API connected — {db_chunks} chunks indexed")
    except Exception:
        st.error("API not reachable. Start the backend:\n`uvicorn main:app --reload`")

    st.divider()

    # Upload section
    st.markdown("### Upload Legal PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", key="pdf_uploader")
    category = st.selectbox("Document category", ["constitution", "judgment", "act", "legal"])

    if st.button("Ingest PDF", type="primary", disabled=uploaded_file is None):
        with st.spinner("Ingesting PDF..."):
            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"category": category},
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Done! {data['pages_loaded']} pages → "
                        f"{data['chunks_stored']} chunks stored."
                    )
                else:
                    st.error(f"Upload failed: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Session controls
    st.markdown("### Session")
    st.caption(f"ID: `{st.session_state.session_id[:8]}...`")
    if st.button("New conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        requests.post(f"{API_URL}/chat/clear",
                      json={"session_id": st.session_state.session_id},
                      timeout=5)
        st.rerun()

    st.divider()
    st.markdown("**Quick questions:**")
    quick_questions = [
        "What is Article 21?",
        "Explain Fundamental Rights",
        "What is IPC Section 302?",
        "Powers of Supreme Court",
        "What is Article 14?",
    ]
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}"):
            st.session_state.quick_question = q
            st.rerun()


# ─── Main area ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">⚖️ LexiRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Indian Constitutional & Legal AI Assistant</p>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Summarize", "🌐 Translate"])

# ═══ TAB 1: CHAT ════════════════════════════════════════════════════════════
with tab1:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f'📄 <b>{src["file"]}</b> — Page {src["page"]} '
                            f'({src["category"]})<br>'
                            f'<i>{src["snippet"]}</i>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Handle quick questions from sidebar
    if hasattr(st.session_state, "quick_question"):
        prompt = st.session_state.quick_question
        del st.session_state.quick_question
    else:
        prompt = st.chat_input("Ask a legal question... e.g. 'What is Article 21?'")

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Consulting legal database..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "question": prompt,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)
                        if sources:
                            with st.expander(f"Sources ({len(sources)})"):
                                for src in sources:
                                    st.markdown(
                                        f'<div class="source-box">'
                                        f'📄 <b>{src["file"]}</b> — Page {src["page"]} '
                                        f'({src["category"]})<br>'
                                        f'<i>{src["snippet"]}</i>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    else:
                        err = resp.json().get("detail", "Unknown error")
                        st.error(f"Error: {err}")
                except requests.Timeout:
                    st.error("Request timed out. The model may be loading — try again in a moment.")
                except Exception as e:
                    st.error(f"Connection error: {e}")


# ═══ TAB 2: SUMMARIZE ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("### Summarize or Simplify Legal Text")
    col1, col2 = st.columns([3, 1])

    with col2:
        mode = st.radio("Mode", ["Summarize", "Simplify for citizens"], key="summarize_mode")
        sum_mode = "summarize" if mode == "Summarize" else "simplify"

    with col1:
        # Text input option
        legal_text = st.text_area(
            "Paste legal text here",
            height=200,
            placeholder="Paste a judgment, legal section, or article text...",
        )

        # PDF upload option
        sum_pdf = st.file_uploader("Or upload a PDF", type="pdf", key="sum_pdf")

        if st.button("Process", type="primary"):
            if sum_pdf:
                with st.spinner("Extracting and summarizing PDF..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/summarize/pdf",
                            files={"file": (sum_pdf.name, sum_pdf.getvalue(), "application/pdf")},
                            timeout=180,
                        )
                        if resp.status_code == 200:
                            st.markdown("### Summary")
                            st.markdown(resp.json()["summary"])
                        else:
                            st.error(resp.json().get("detail", "Error"))
                    except Exception as e:
                        st.error(f"Error: {e}")
            elif legal_text.strip():
                with st.spinner("Processing..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/summarize",
                            json={"text": legal_text, "mode": sum_mode},
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            st.markdown("### Result")
                            st.markdown(resp.json()["result"])
                        else:
                            st.error(resp.json().get("detail", "Error"))
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please paste some text or upload a PDF.")


# ═══ TAB 3: TRANSLATE ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("### Translate Legal Text")
    col_a, col_b = st.columns(2)

    with col_a:
        src_lang = st.selectbox("Source language", ["en", "hi"], format_func=lambda x: "English" if x == "en" else "Hindi")
        input_text = st.text_area("Input text", height=250, key="trans_input")

    with col_b:
        tgt_lang = st.selectbox("Target language", ["hi", "en"], format_func=lambda x: "Hindi" if x == "hi" else "English")
        output_placeholder = st.empty()

    if st.button("Translate", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text to translate.")
        elif src_lang == tgt_lang:
            st.warning("Source and target languages are the same.")
        else:
            with st.spinner("Translating..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/translate",
                        json={"text": input_text, "source_lang": src_lang, "target_lang": tgt_lang},
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        with col_b:
                            st.text_area("Translation", value=resp.json()["translated"],
                                         height=250, key="trans_output")
                    else:
                        st.error(resp.json().get("detail", "Translation error"))
                except Exception as e:
                    st.error(f"Error: {e}")
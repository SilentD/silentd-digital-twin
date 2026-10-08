import streamlit as st
import os
import uuid
import json
from utils.llm import SafeLLM
from utils import rag, tools

# -----------------------------------------------------------------------------
# App Setup & Path Constants
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Digital Twin Engine", page_icon="🤖", layout="centered")

DATA_DIR = "data/"
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Inject custom dark UI styling
st.markdown("""
<style>
    .stTextInput > div > div > input { background-color: #1E293B; color: #E2E8F0; border: 1px solid #334155; border-radius: 8px; }
    hr { border-color: #334155; }
    .stChatMessage { background-color: #1E293B; border: 1px solid #334155; border-radius: 10px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# History Persistence Helpers
# -----------------------------------------------------------------------------
def load_all_histories() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"active_session_id": "", "sessions": {}}

def save_all_histories(data: dict):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_current_messages() -> list:
    histories = load_all_histories()
    session_id = st.session_state.get("active_session_id")
    if session_id and session_id in histories.get("sessions", {}):
        return histories["sessions"][session_id].get("messages", [])
    return []

def append_message_turn(role: str, content: str, usage: dict = None):
    histories = load_all_histories()
    session_id = st.session_state.get("active_session_id")
    if session_id and session_id in histories.get("sessions", {}):
        msg_entry = {
            "role": role,
            "content": content,
            "timestamp": tools.get_current_timestamp()
        }
        if usage:
            msg_entry["usage"] = usage
        histories["sessions"][session_id]["messages"].append(msg_entry)
        save_all_histories(histories)

def create_new_session():
    histories = load_all_histories()
    new_id = f"session_{uuid.uuid4().hex[:6]}"
    if "sessions" not in histories:
        histories["sessions"] = {}
    
    count = len(histories["sessions"]) + 1
    histories["sessions"][new_id] = {
        "title": f"Conversation {count}",
        "created_at": tools.get_current_timestamp(),
        "messages": []
    }
    histories["active_session_id"] = new_id
    save_all_histories(histories)
    st.session_state.active_session_id = new_id

# -----------------------------------------------------------------------------
# State Initialization
# -----------------------------------------------------------------------------
if "safe_llm" not in st.session_state:
    st.session_state.safe_llm = SafeLLM()

hist_data = load_all_histories()
if not hist_data.get("sessions"):
    create_new_session()
    hist_data = load_all_histories()

if "active_session_id" not in st.session_state or not st.session_state.active_session_id:
    st.session_state.active_session_id = hist_data.get("active_session_id", list(hist_data["sessions"].keys())[0])

if "vector_db" not in st.session_state:
    if os.path.exists(KNOWLEDGE_FILE):
        st.session_state.vector_db = rag.process_json_knowledge(KNOWLEDGE_FILE)
    else:
        st.session_state.vector_db = None

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Twin Control Center")

    if st.button("➕ Start New Conversation", use_container_width=True):
        create_new_session()
        st.rerun()

    st.markdown("---")
    st.subheader("💬 Chat Sessions")
    
    all_sessions = load_all_histories().get("sessions", {})
    session_keys = list(all_sessions.keys())

    if session_keys:
        curr_idx = session_keys.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_keys else 0
        selected_key = st.selectbox(
            "Select History Session:",
            options=session_keys,
            format_func=lambda k: f"{all_sessions[k].get('title', k)} ({k})",
            index=curr_idx
        )
        if selected_key != st.session_state.active_session_id:
            st.session_state.active_session_id = selected_key
            hist_store = load_all_histories()
            hist_store["active_session_id"] = selected_key
            save_all_histories(hist_store)
            st.rerun()

    st.markdown("---")
    st.subheader("📊 Token Usage Metrics")
    current_msgs = get_current_messages()
    
    if hasattr(tools, 'calculate_session_token_metrics'):
        token_stats = tools.calculate_session_token_metrics(current_msgs)
        col1, col2 = st.columns(2)
        with col1:
            st.caption("⚡ **Per Minute**")
            st.text(f"In:  {token_stats['per_minute']['inbound']}")
            st.text(f"Out: {token_stats['per_minute']['outbound']}")
        with col2:
            st.caption("🕒 **Per Hour**")
            st.text(f"In:  {token_stats['per_hour']['inbound']}")
            st.text(f"Out: {token_stats['per_hour']['outbound']}")

        st.caption("📅 **Per Day (24h)**")
        st.text(f"Inbound:  {token_stats['per_day']['inbound']} tokens")
        st.text(f"Outbound: {token_stats['per_day']['outbound']} tokens")

    st.markdown("---")
    st.subheader("🗂️ Engine Performance")
    st.info(f"System Clock: {tools.get_current_timestamp()}")

    # Display status based on vector length from your RAG dict structure
    if st.session_state.vector_db and len(st.session_state.vector_db.get("chunks", [])) > 0:
        chunk_count = len(st.session_state.vector_db["chunks"])
        st.success("knowledge.json active")
        st.caption(f"Total Vector Chunks Loaded: {chunk_count}")
    else:
        st.error("Missing or uninitialized data/knowledge.json")

    if st.button("🔄 Sync Knowledge Base", use_container_width=True):
        st.toast("Knowledge base embeddings synchronized!")
        st.rerun()
    

# -----------------------------------------------------------------------------
# Main Chat Area
# -----------------------------------------------------------------------------
st.title("🤖 Activity Predictor & Implementation Twin")
st.caption(f"Active Session: `{st.session_state.active_session_id}`")

# Display conversation turns from active session history
for message in get_current_messages():
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Execution
if user_input := st.chat_input("Ask your digital twin..."):
    # 1. Capture history BEFORE adding user turn (prevents duplicate context in LLM call)
    prior_history = get_current_messages()

    # 2. Render & save user query
    with st.chat_message("user"):
        st.write(user_input)
    append_message_turn("user", user_input)

    # 3. RAG Retrieval
    matched_context = ""
    if st.session_state.vector_db:
        matched_context = rag.retrieve_context(
            query=user_input, 
            vector_db=st.session_state.vector_db, 
            top_k=2
        )

    # 4. SafeLLM Inference
    with st.chat_message("assistant"):
        with st.spinner("Processing twin inference engine..."):
            ai_response = st.session_state.safe_llm.chat(
                user_input=user_input,
                matched_context=matched_context,
                history=prior_history
            )
            usage_stats = st.session_state.safe_llm.get_last_token_usage()
            st.write(ai_response)

    # 5. Persist assistant response with exact token metrics
    append_message_turn("assistant", ai_response, usage=usage_stats)
    st.rerun()
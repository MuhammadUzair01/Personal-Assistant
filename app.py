import streamlit as st
from chatbot import chat
import datetime

# Page configuration
st.set_page_config(
    page_title="Smart AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Sidebar info
with st.sidebar:
    st.title("🧠 Smart Chatbot")
    st.markdown("Built with:")
    st.markdown("- `LangChain` + `Groq (Gemma)`")
    st.markdown("- `ChromaDB` for memory")
    st.markdown("- `Streamlit` for UI")
    st.markdown("---")
    st.markdown("Ask anything. The bot will remember previous messages to help contextualize your question.")
    st.caption("© 2025 SmartBot Inc.")

# Title & subtitle
st.title("🤖 Smart Context-Aware Chatbot")
st.subheader("An AI that remembers what you say 🔁")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Type your message...")

if query:
    try:
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Call backend and display response
        with st.spinner("Thinking..."):
            response = chat(query)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error("⚠️ Something went wrong. Please try again.")
        st.exception(e)

# Optional footer or message
st.markdown("<hr style='margin-top:2rem;margin-bottom:1rem;'>", unsafe_allow_html=True)
st.caption(f"🕒 {datetime.datetime.now().strftime('%B %d, %Y — %I:%M %p')}")

import streamlit as st
from chatbot import chat, reset_memory
import datetime

# Page configuration
st.set_page_config(
    page_title="Smart AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Sidebar
with st.sidebar:
    
    if st.button("🧹 Confirm Reset Memory"):
        reset_memory()
        st.session_state.messages = []
        st.success("✅ Memory has been reset.")
    st.markdown("---")   
    st.title("🧠 Smart Chatbot")
    st.markdown("Built with:")
    st.markdown("- `LangChain` + `ChatGroq`")
    st.markdown("- `ChromaDB` for memory")
    st.markdown("- `Streamlit` for UI")
    st.markdown("---")
    st.markdown("Ask anything. The bot remembers previous messages.")
    st.markdown("Powered by [Groq](https://groq.com) and [ChromaDB](https://www.trychroma.com)")



# Title and subtitle
st.title("🤖 Smart Context-Aware Chatbot")
st.subheader("An AI that remembers what you say 🔁")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input & response
query = st.chat_input("Type your message...")

if query:
    try:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            response = chat(query)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error("⚠️ Something went wrong.")
        st.exception(e)

# Footer timestamp
st.markdown("<hr style='margin-top:2rem;margin-bottom:1rem;'>", unsafe_allow_html=True)
st.caption(f"🕒 {datetime.datetime.now().strftime('%B %d, %Y — %I:%M %p')}")

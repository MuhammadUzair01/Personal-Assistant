# run python -m venv venv
# run venv\Scripts\activate
# pip install -r requirements.txt
# streamlit run chatbot.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from chromadb.utils import embedding_functions
import chromadb
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# === Embedding Setup ===
print("🔧 Initializing embedding function")
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# === ChromaDB Setup ===
print("📁 Setting up ChromaDB client")

# Always use path relative to this file (not working dir)
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "chroma_db_store")
client = chromadb.PersistentClient(path=db_path)

collection = client.get_or_create_collection(
    name="chatbot-history",
    embedding_function=embedding_function
)

print(f"📦 ChromaDB initialized. Documents stored: {collection.count()}")
if collection.count() == 0:
    print("🆕 No memory found. Starting fresh.")
else:
    print("✅ Previous memory loaded.")

# === LLM Setup ===
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY2"),
    model="gemma2-9b-it"
)

# === Prompt ===
prompt = """
You are a helpful AI assistant. Answer the user's questions based on the provided context.
If you don't know the answer, say "I don't know" or "I'm not sure".
If the question is not related to the context, politely inform the user that you can only answer questions based on the provided information.
If the user asks for help with a specific topic, provide a brief explanation or direct them to relevant resources.
If the user asks for a summary, provide a concise summary of the context.
If the user asks for previous conversation turns, retrieve and display the last few messages.
If the user asks for a specific piece of information, try to find it in the context and provide it.
"""

# === Context Retrieval ===
def retrieve_context(query: str, k: int = 5) -> list:
    """
    Retrieve up to k most relevant conversation turns for the given query.
    Returns a list of HumanMessage and AIMessage.
    """
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    history = []
    for text, meta in zip(docs, metas):
        role = meta.get("role", "user")
        if role == "assistant":
            history.append(AIMessage(content=text))
        else:
            history.append(HumanMessage(content=text))
    return history

# === Chat Function ===
def chat(query: str) -> str:
    """
    Process a user query, retrieve context, generate a response, and store the turn.
    """
    # Retrieve relevant past context
    context_msgs = retrieve_context(query)

    # Build message history for LLM
    messages = [SystemMessage(content=prompt)] + context_msgs + [HumanMessage(content=query)]

    # Get assistant response
    response = llm.invoke(messages)
    answer = response.content

    # Store new conversation turn in ChromaDB
    query_id = str(uuid.uuid4())
    answer_id = str(uuid.uuid4())
    collection.add(
        ids=[query_id, answer_id],
        documents=[query, answer],
        metadatas=[{"role": "user"}, {"role": "assistant"}]
    )

    return answer


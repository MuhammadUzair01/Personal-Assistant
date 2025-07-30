# run python -m venv venv
# run venv\Scripts\activate
# pip install -r requirements.txt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

from langchain_groq import ChatGroq
from chromadb.utils import embedding_functions
import chromadb
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv
import uuid  # Add import for generating unique IDs
load_dotenv()

print("Initializing Sentence Transformer embeddings")
# Initialize embeddings
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

print("Initializing ChromaDB client")
# Initialize persistent ChromaDB client and collection
client = chromadb.PersistentClient(path="./chroma_db_store")

collection = client.get_or_create_collection(
    name="chatbot-history",
    embedding_function=embedding_function
)


print("ChromaDB client and collection initialized")
# Initialize Groq LLM (ensure GROQ_API_KEY is set in environment)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY2"),
    model="gemma2-9b-it"  # Using Mixtral model
)


def retrieve_context(query: str, k: int = 5) -> list:
    """
    Retrieve up to k most relevant conversation turns for the given query.
    Returns a list of HumanMessage and AIMessage.
    """
    print(f"Retrieving context for query: {query}")
    # Query the collection
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

prompt = """
You are a helpful AI assistant. Answer the user's questions based on the provided context.
If you don't know the answer, say "I don't know" or "I'm not sure".
If the question is not related to the context, politely inform the user that you can only answer questions based on the provided information.
If the user asks for help with a specific topic, provide a brief explanation or direct them to relevant resources.
If the user asks for a summary, provide a concise summary of the context.
If the user asks for previous conversation turns, retrieve and display the last few messages.
If the user asks for a specific piece of information, try to find it in the context and provide it.

"""

def chat(query: str) -> str:
    """
    Process a user query, retrieve context, generate a response, and store the turn.
    """
    print(f"Processing user query: {query}")
    # Retrieve previous context
    context_msgs = retrieve_context(query)

    # Build messages for LLM
    system_msg = SystemMessage(content=prompt)
    human_msg = HumanMessage(content=query)
    messages = [system_msg] + context_msgs + [human_msg]

    # Get assistant response
    response = llm.invoke(messages)
    answer = response.content

    # Generate unique IDs for the documents
    query_id = str(uuid.uuid4())
    answer_id = str(uuid.uuid4())

    # Persist user query and assistant answer
    collection.add(
        ids=[query_id, answer_id],
        documents=[query, answer],
        metadatas=[{"role": "user"}, {"role": "assistant"}]
    )

    return answer

if __name__ == "__main__":
    print("Chatbot initialized. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        reply = chat(user_input)
        print(f"Assistant: {reply}")
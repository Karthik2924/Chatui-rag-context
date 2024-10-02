import os
from typing import Dict, Any
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from operator import itemgetter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import ChatCohere
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

vector_store = FAISS.load_local(
    "faiss_index_cohere", embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever()

# Initialize FastAPI app
app = FastAPI(title="AI Chatbot API", description="An API for interacting with an AI chatbot using history.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# In-Memory History Management
# ---------------------------

# Store for chat histories, keyed by session_id
store: Dict[str, BaseChatMessageHistory] = {}

# Simple Message class to simulate chat messages
class SimpleMessage:
    def __init__(self, content: str, message_type: str):
        self.content = content
        self.type = message_type

# In-memory history class
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages: list[SimpleMessage] = []

    def add_message(self, message: SimpleMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []

# Function to retrieve session history
def get_session_history(session_id: str) -> InMemoryHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# ---------------------------
# AI Model and Prompt Setup
# ---------------------------

# Fake retriever to simulate document search
def retriever(query: str):
    assert isinstance(query, str)
    return [
        Document(page_content="London is the capital of France"),
        Document(page_content="Rome is the capital of UK"),
    ]

retriever = RunnableLambda(retriever)

# Initialize the Cohere AI model
model = ChatCohere(model="command-r-plus")

# Chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant who's good at {ability}. Here is some {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 1: Retrieve documents from the query
context = itemgetter("question") | retriever | format_docs

# Step 2: Build a chain for processing the request
first_step = RunnablePassthrough.assign(context=context)
chain = first_step | prompt | model

# ---------------------------
# RunnableWithMessageHistory Setup
# ---------------------------

# History management within the chain
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,  # Expects session_id
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for the session.",
            default="",
            is_shared=True,
        ),
    ],
)

# ---------------------------
# Chat Processing Function
# ---------------------------

def process_chat(session_id: str, question: str, ability: str) -> str:
    """
    Process a chat message and return the AI's response.

    Args:
        session_id (str): Unique identifier for the chat session.
        question (str): The user's question.
        ability (str): The ability or context for the AI assistant.

    Returns:
        str: The AI-generated answer.
    """
    try:
        # Run the chain with message history
        response = with_message_history.invoke(
            {
                "ability": ability,
                "question": question,
            },
            config={
                "configurable": {
                    "session_id": session_id,  # Pass session_id explicitly
                }
            },
        )
        
        # Extract the answer
        answer = response.content if hasattr(response, 'content') else str(response)
        return answer
    
    except Exception as e:
        print(f"Error processing chat: {str(e)}")
        return None

# ---------------------------
# Input Validation Function
# ---------------------------

def validate_chat_input(data: Dict[str, Any]) -> str:
    """
    Validate the chat input data.

    Args:
        data (Dict[str, Any]): The input data containing session_id, question, and ability.

    Returns:
        str: An error message if validation fails, otherwise an empty string.
    """
    required_fields = ['session_id', 'question', 'ability']
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"
        if not isinstance(data[field], str):
            return f"Field '{field}' must be a string"
    return ""

# ---------------------------
# FastAPI Endpoints
# ---------------------------

class ChatInput(BaseModel):
    session_id: str
    question: str
    ability: str

class ChatOutput(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatOutput)
async def chat_endpoint(chat_input: ChatInput):
    """
    Chat API endpoint to process the user's message and return an AI-generated response.
    """
    validation_error = validate_chat_input(chat_input.dict())
    if validation_error:
        raise HTTPException(status_code=422, detail=validation_error)
    
    start_time = time.time()
    answer = process_chat(
        session_id=chat_input.session_id,
        question=chat_input.question,
        ability=chat_input.ability
    )
    end_time = time.time()

    if not answer:
        raise HTTPException(status_code=500, detail="Failed to generate response")
    
    return {"answer": answer, "execution_time": f"{end_time - start_time:.2f} seconds"}

# ---------------------------
# Run the Application with Uvicorn
# ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="PDF AI Assistant 🚀", layout="centered", initial_sidebar_state="expanded")

# --- Custom CSS for a better look and feel ---
st.markdown("""
<style>
    /* General body styling */
    body {
        color: #F0F2F6; /* Light gray text */
        background-color: #0E1117; /* Dark background */
    }
    .stApp {
        background-color: #0E1117; /* Ensure app background is dark */
    }

    /* Main header styling */
    .main-header {
        font-size: 3em;
        color: #4CAF50; /* Green */
        text-align: center;
        margin-bottom: 25px;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
        padding-top: 15px;
    }

    /* Sub-headers */
    h2 {
        color: #66BB6A; /* Lighter green for subheadings */
        font-size: 1.8em;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    h3 {
        color: #81C784; /* Even lighter green */
        font-size: 1.4em;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Text input fields */
    .stTextInput>div>div>input {
        background-color: #1A1C22; /* Darker input background */
        color: #F0F2F6;
        border-radius: 8px;
        border: 1px solid #4CAF50;
        padding: 12px 15px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
    }
    .stTextInput label {
        color: #F0F2F6;
        font-weight: bold;
    }

    /* File uploader */
    .stFileUploader>div>div>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stFileUploader>div>div>button:hover {
        background-color: #43A047; /* Darker green on hover */
    }
    .stFileUploader label {
        color: #F0F2F6;
        font-weight: bold;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #43A047;
    }

    /* Alert messages */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #c3e6cb;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #ffeeba;
    }
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #bee5eb;
    }
    .stError {
        background-color: #f8d7da; /* Light red */
        color: #721c24; /* Dark red */
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #f5c6cb;
    }

    /* Chat message containers */
    .chat-container {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .user-message {
        color: #FFD700; /* Gold for user messages */
        font-weight: bold;
    }
    .assistant-message {
        color: #F0F2F6; /* Light gray for assistant messages */
    }

    /* Expander for chat history */
    .streamlit-expanderHeader {
        background-color: #1e1e1e;
        color: #81C784;
        border-radius: 8px;
        padding: 10px 15px;
        font-weight: bold;
        margin-top: 20px;
    }
    .streamlit-expanderContent {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 15px;
    }

    /* Horizontal rule */
    hr {
        border-top: 2px solid #333;
        margin: 30px 0;
    }

    /* Sidebar specific styling */
    .stSidebar {
        background-color: #1A1C22; /* Darker sidebar background */
        color: #F0F2F6;
        padding: 20px;
    }
    .stSidebar .stTextInput>div>div>input,
    .stSidebar .stSlider>div>div>div {
        background-color: #262730; /* Adjust input background in sidebar */
    }
    .stSidebar h2, .stSidebar h3 {
        color: #4CAF50; /* Green for sidebar headers */
    }
    .st-emotion-cache-1jmve5h { /* Targeting the radio button options to make them more clickable */
        background-color: #262730 !important;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 5px;
        transition: background-color 0.3s;
    }
    .st-emotion-cache-1jmve5h:hover {
        background-color: #333640 !important;
    }
    .st-emotion-cache-1jmve5h input[type="radio"]:checked + div {
        background-color: #4CAF50 !important; /* Highlight for selected radio button */
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<h1 class="main-header">📄 PDF AI Assistant: Your Document Companion 🧠</h1>', unsafe_allow_html=True)

# --- Sidebar for Navigation and Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3653/3653950.png", width=100) # Optional: Add a logo
    st.markdown("## Navigation 🧭")

    # Set 'About This Project ✨' as the default selection using key and default value
    page_selection = st.radio(
        "Go to",
        ["About This Project ✨", "Chatbot 💬", "GitHub 🐙"],
        index=0, # Set default to the first item (About This Project)
        key="page_selector"
    )

    api_key = None
    session_id = "default_session"
    temperature = 0.7
    top_p = 1.0
    max_tokens = 1024
    uploaded_files = None

    if page_selection == "Chatbot 💬":
        st.markdown("---")
        st.markdown("## Configuration ⚙️")

        api_key = st.text_input("🔑 **Enter your Groq API Key**", type="password", help="You can get your Groq API key from https://console.groq.com/keys")

        session_id = st.text_input("🆔 **Session ID**", value="default_session", help="Use a unique ID for each conversation if you want to keep separate chat histories.")

        st.markdown("### Advanced Model Settings (Optional) 🛠️")
        temperature = st.slider("🌡️ **Temperature**", min_value=0.0, max_value=2.0, value=0.7, step=0.1,
                                help="Controls the randomness of the output. Higher values mean more creative, lower values mean more deterministic.")
        top_p = st.slider("🔝 **Top P**", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                          help="Controls nucleus sampling. Only considers tokens whose cumulative probability exceeds top_p.")
        max_tokens = st.slider("📏 **Max Tokens**", min_value=50, max_value=4096, value=1024, step=50,
                               help="The maximum number of tokens to generate in the response.")
    elif page_selection == "GitHub 🐙":
        st.markdown("---")
        st.markdown("""
        ## GitHub Repository 🐙

        Explore the full source code and contribute to this project on GitHub!
        
        👉 **[Visit the GitHub Repo](https://github.com/divyaraj-vihol)**
        """)
        st.stop() # Stop further execution if GitHub is selected

# --- Main Content Area ---
if page_selection == "About This Project ✨":
    st.markdown("""
    ## About This Project ✨

    This project is a **Conversational AI Assistant** 🤖 built to interact with your PDF documents effortlessly. It's powered by **Retrieval-Augmented Generation (RAG)**, which means it intelligently fetches information from your uploaded PDFs to provide accurate and context-aware answers, all while maintaining a natural and fluid conversation.

    ### Key Features:
    * **PDF Uploads** 📂: Easily upload one or multiple PDF files to create a dynamic knowledge base.
    * **Conversational Chat History** 💬: The assistant remembers your previous interactions, allowing for a seamless and intuitive chat experience.
    * **Contextual Understanding** 💡: Get precise answers derived directly from the content of your uploaded documents.
    * **Groq API Integration** ⚡: Leverages the high-performance Groq API for incredibly fast and accurate responses.
    * **Local Vector Store** 🗄️: Efficiently stores and retrieves document embeddings for rapid access to information.

    ### How it Works:
    1.  **Document Ingestion**: You upload your PDF files, and the system extracts their textual content.
    2.  **Smart Indexing**: The extracted text is then chunked into manageable pieces and converted into numerical representations (embeddings). These embeddings are stored in a local vector database.
    3.  **Intelligent Retrieval**: When you ask a question, the assistant first uses your chat history to refine the query for better context. Then, it searches the vector database to retrieve the most relevant document snippets.
    4.  **Answer Generation**: The retrieved context, combined with your refined question, is fed to the powerful Groq language model (specifically `llama3-70b-8192`) to generate a concise and accurate answer.
    5.  **Seamless Interaction**: The generated answer is presented to you, and the entire conversation history is preserved, enabling you to continue the dialogue naturally.
    """)

elif page_selection == "Chatbot 💬":
    # --- Document Upload Section (Main Content Area - only if Chatbot is selected) ---
    st.markdown("---")
    st.markdown("## Upload Your Documents 📂")
    uploaded_files = st.file_uploader("📥 **Upload PDF file(s)**", type="pdf", accept_multiple_files=True)

    # In-memory chat store (must be initialized before chains are created)
    if "store" not in st.session_state:
        st.session_state.store = {}

    # --- Processing and Chat Logic ---
    if api_key and uploaded_files:
        # Load and parse PDFs
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"❌ Error loading PDF **{uploaded_file.name}**: {str(e)}")
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if not documents:
            st.error("🚫 No valid PDF documents were loaded. Please check your files.")
            st.stop()

        # Text splitting and embedding
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        # Embedding model and local chroma vector store
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            persist_directory = "chroma_store"
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            retriever = vectorstore.as_retriever()
            st.success("✅ Documents processed and vector store created!")
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            st.stop()

        # Load LLM with configurable parameters
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama3-70b-8192",
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            st.success("✅ Groq LLM initialized!")
        except Exception as e:
            st.error(f"❌ Error initializing Groq client: {str(e)}")
            st.stop()

        # Prompt for question reformulation (history-aware)
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, reformulate it into a standalone question. If unnecessary, return as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        try:
            history_aware_retriever = create_history_aware_retriever(
                llm=llm, retriever=retriever, prompt=contextualize_q_prompt
            )
            st.info("ℹ️ History-aware retriever initialized.")
        except Exception as e:
            st.error(f"❌ Error creating history-aware retriever: {str(e)}")
            st.stop()

        # Prompt for final answering
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant. Use the provided context to answer questions as accurately as possible. If the answer is not in the context, clearly state that you don't have enough information. Keep answers concise, preferably within 3-4 sentences.\n\nContext: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        try:
            document_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)
            st.info("ℹ️ RAG chain ready for queries.")
        except Exception as e:
            st.error(f"❌ Error creating retrieval chain: {str(e)}")
            st.stop()

        # Manage chat history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        # Combine chain with memory
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # --- Chat Interface ---
        st.markdown("---")
        st.markdown("## Start Chatting! 💬")
        user_input = st.text_input("**Ask your question about the PDFs:**", key="user_question_input")
        
        if user_input:
            history = get_session_history(session_id)
            with st.spinner("Generating answer... ⏳"):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.markdown("### 🤖 Assistant:")
                    st.success(response["answer"])
                    
                    # Display chat history
                    st.markdown("---")
                    st.markdown("## 🧠 Chat History")
                    with st.expander("Click to view full chat history", expanded=False):
                        # Display messages in reverse chronological order (latest first)
                        messages = history.messages
                        for i in range(len(messages) - 2, -1, -2): # Iterate backwards by 2
                            user_msg = messages[i]
                            ai_msg = messages[i + 1]

                            if user_msg.type == "human" and ai_msg.type == "ai":
                                st.markdown(
                                    f"""
                                    <div class="chat-container">
                                    <p class="user-message">🧑 **You:** {user_msg.content}</p>
                                    <p class="assistant-message">🤖 **Assistant:** {ai_msg.content}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                except Exception as e:
                    st.error(f"🔥 Oh no! An error occurred while generating the response: {str(e)}")

    # --- Initial prompts for empty inputs (only if Chatbot is selected) ---
    else: # This block runs if Chatbot is selected but API key or files are missing
        if not api_key:
            st.warning("⚠️ **Please enter your Groq API key** in the sidebar to proceed.")
        elif not uploaded_files:
            st.info("📎 **Please upload at least one PDF file** to begin.")
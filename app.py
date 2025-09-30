import streamlit as st
import os
import json
from rag import call_rag

# Function to generate chatbot response and citation
def get_response(user_input, sources, include_arxiv):
    arxiv_source = True if include_arxiv else False

    # Handle empty sources
    if not sources:
        sol = call_rag(file_path=None, question=user_input, include_arxis=arxiv_source)
    else:
        # If multiple sources, you can use the first one or extend to multiple
        sol = call_rag(file_path=os.path.join("chat_sources", sources[0]), question=user_input, include_arxis=arxiv_source)

    response_content, citation = sol['result'], sol['citations']
    return response_content, citation

# Initialize session state
if 'chats' not in st.session_state:
    st.session_state['chats'] = {}
if 'current_chat' not in st.session_state:
    st.session_state['current_chat'] = None
if 'sources' not in st.session_state:
    st.session_state['sources'] = {}
if 'include_arxiv' not in st.session_state:
    st.session_state['include_arxiv'] = False

# Directories for storing chat data and sources
chat_file = "chats.json"
source_dir = "chat_sources"
os.makedirs(source_dir, exist_ok=True)

# Load chats from a file
def load_chats():
    if os.path.exists(chat_file):
        with open(chat_file, 'r') as file:
            return json.load(file)
    return {}

# Save chats to a file
def save_chats():
    with open(chat_file, 'w') as file:
        json.dump(st.session_state['chats'], file, indent=4)

# Load chat history into session state on startup
if not st.session_state['chats']:
    st.session_state['chats'] = load_chats()

# Sidebar: Chat Management
st.sidebar.header("Chat Management")
chat_names = list(st.session_state['chats'].keys())
selected_chat = st.sidebar.radio("Select or create a new chat", ["New Chat"] + chat_names)

# Create new chat
if selected_chat == "New Chat":
    new_chat_name = st.sidebar.text_input("Enter a new chat name")
    if st.sidebar.button("Create Chat"):
        if new_chat_name and new_chat_name not in st.session_state['chats']:
            st.session_state['chats'][new_chat_name] = []
            st.session_state['sources'][new_chat_name] = []
            st.session_state['current_chat'] = new_chat_name
            save_chats()
            st.rerun()
        elif new_chat_name in st.session_state['chats']:
            st.sidebar.error("Chat with this name already exists!")
        else:
            st.sidebar.error("Please enter a valid chat name.")
else:
    st.session_state['current_chat'] = selected_chat

# Main chat interface
if st.session_state['current_chat']:
    chat_name = st.session_state['current_chat']
    st.header(f"Chat: {chat_name}")

    # Checkbox to include ArXiv as a data source
    include_arxiv = st.checkbox("Include arXiv.org as a data source", value=st.session_state['include_arxiv'])
    st.session_state['include_arxiv'] = include_arxiv

    # Upload PDF sources
    st.subheader("Upload Sources")
    uploaded_files = st.file_uploader(
        "Drag and drop or upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(source_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.session_state['sources'][chat_name] = st.session_state['sources'].get(chat_name, [])
            if uploaded_file.name not in st.session_state['sources'][chat_name]:
                st.session_state['sources'][chat_name].append(uploaded_file.name)
        st.success("Files uploaded successfully!")

    # Display uploaded sources
    st.write("### Sources for this chat:")
    if chat_name in st.session_state['sources']:
        for source in st.session_state['sources'][chat_name]:
            st.markdown(f"- {source}")
    else:
        st.write("No sources uploaded yet.")

    # Display chat history
    chat_history = st.session_state['chats'][chat_name]
    for entry in chat_history:
        with st.container():
            # User messages
            st.markdown(
                f"<div style='text-align: left; color: white; background-color: #007BFF; padding: 8px; border-radius: 5px;'>"
                f"**You:** {entry['user']}</div>",
                unsafe_allow_html=True
            )
            # Bot responses
            st.markdown(
                f"<div style='text-align: left; color: black; background-color: #E8E8E8; padding: 8px; margin-top: 5px; border-radius: 5px;'>"
                f"**Bot:** {entry['response']}<br><i>{entry['citation']}</i></div>",
                unsafe_allow_html=True
            )

    # Input for new messages
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        if user_input:
            sources = st.session_state['sources'].get(chat_name, [])
            response, citation = get_response(user_input, sources, include_arxiv)
            chat_history.append({
                "user": user_input,
                "response": response,
                "citation": citation
            })
            st.session_state['chats'][chat_name] = chat_history
            save_chats()
            st.rerun()
        else:
            st.error("Please enter a message to send.")
else:
    st.write("Please create or select a chat to start.")

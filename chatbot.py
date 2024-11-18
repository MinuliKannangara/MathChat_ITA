import streamlit as st
from dotenv import load_dotenv
import os
import shelve

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
import re

from Chains.generate_examples import generate_examples
from Chains.load_prompt import load_prompt
from Chains.rag_chain import create_rag_chain
# Import the additional chains and settings
from Configs.config import settings
from Chains.solve_problem import solve_problem
from Chains.guide_student import guide_student
from Chains.identify_sub_topics import identify_subtopic
from Chains.lesson_selector import lesson_selector
from Chains.curriculum_guide_selector import curriculum_guide_selector

load_dotenv()
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# API Client setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model=settings.MODEL_NAME, streaming=True)
# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
# Additional configurations for OAuth and session data
client_id = os.getenv('OAUTH_GOOGLE_CLIENT_ID')
client_secret = os.getenv('OAUTH_GOOGLE_CLIENT_SECRET')
api_key = settings.OPENAI_API_KEY
model = settings.MODEL_NAME

# Initialize session data storage
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

# Ensure openai_model is initialized in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model

# Ensure examples state is tracked
if "show_examples_clicked" not in st.session_state:
    st.session_state.show_examples_clicked = False

# Load chat history from shelve file by chat title
def load_chat_history_key(chat_title):
    with shelve.open("chat_history") as db:
        return db.get(chat_title, [])


# Save chat history to shelve file by chat title
def save_chat_history_key(chat_title, messages):
    with shelve.open("chat_history") as db:
        db[chat_title] = messages


# Add CSS to adjust button width to match the dropdown
st.markdown(
    """
    <style>
    .sidebar .stButton button {
        width: 200px;  /* Make all buttons in the sidebar full width */
    }
    .sidebar .stButton {
        margin-bottom: 5px;  /* Add some spacing between buttons */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>Grade 09 - Mathematics Chatbot</h1>", unsafe_allow_html=True)

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize chat titles in session state
if "chat_titles" not in st.session_state:
    with shelve.open("chat_history") as db:
        st.session_state.chat_titles = list(db.keys())

# Initialize the current chat title
if "current_chat_title" not in st.session_state:
    # Set to a default value, such as the most recent chat or create a new one
    if st.session_state.chat_titles:
        st.session_state.current_chat_title = st.session_state.chat_titles[-1]
        st.session_state.messages = load_chat_history_key(st.session_state.current_chat_title)
    else:
        st.session_state.current_chat_title = "Chat 1"
        st.session_state.chat_titles.append(st.session_state.current_chat_title)
        st.session_state.sessions[st.session_state.current_chat_title] = {
            "counter": 0,
            "summary_generated": False,
            "summary_content": "",
            "previous_question": "",
            "expecting_response": False  # A new flag to indicate if a follow-up response is expected
        }

# Sidebar for categories and adding/deleting chat history
with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>Select Subject</h3>", unsafe_allow_html=True)

    # Subject Dropdown - Styled to fit the sidebar length
    subject = st.selectbox("Select Subject", ["Mathematics", "Science", "History"], index=0, key="subject_select")

    # Button to add a new chat
    if st.button("New Chat"):
        # Initialize a new chat
        new_chat_title = f"Chat {len(st.session_state.chat_titles) + 1}"
        st.session_state.messages = []
        st.session_state.current_chat_title = new_chat_title
        st.session_state.chat_titles.append(new_chat_title)
        st.session_state.current_session = {
            "counter": 0,
            "summary_generated": False,
            "summary_content": "",
            "previous_question": "",
            "expecting_response": False
        }
        st.session_state.sessions[new_chat_title] = st.session_state.current_session

    # Load past chats and display them in the sidebar
    st.markdown("<h3 style='text-align: center;'>Past Chats</h3>", unsafe_allow_html=True)
    for chat_title in st.session_state.chat_titles:
        if st.button(chat_title, key=f"chat_{chat_title}"):
            # Load the selected chat into the main chat window
            st.session_state.messages = load_chat_history_key(chat_title)
            st.session_state.current_chat_title = chat_title
            st.session_state.current_session = st.session_state.sessions.get(chat_title, {
                "counter": 0,
                "summary_generated": False,
                "summary_content": "",
                "previous_question": "",
                "expecting_response": False
            })

    # Button to delete chat history
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        st.session_state.chat_titles = []
        with shelve.open("chat_history") as db:
            db.clear()


def is_new_question(user_query):
    """Determines if a user query is likely a new question."""
    new_question_keywords = ["solve", "how to", "find", "calculate", "explain", "new problem"]
    for keyword in new_question_keywords:
        if keyword in user_query.lower():
            return True
    return False


def process_user_query(user_query, session, pdfFilePath, subtopic, syllabus_response_content):
    """Process user query and generate response"""
    try:
        # Check if the query is a new question or if the user completed the previous question
        if is_new_question(user_query) or not session["expecting_response"]:
            # Reset summary if it's a new question
            print("New question detected.")
            session["summary_generated"] = False
            session["summary_content"] = ""
            session["expecting_response"] = True

        # Generate summary if it doesn't exist
        if not session["summary_generated"]:
            print("summary not generated")
            summary = solve_problem(user_query, pdfFilePath, subtopic, syllabus_response_content)
            if isinstance(summary, dict) and 'messages' in summary:
                session["summary_content"] = '\n'.join(map(str, summary['messages']))
                session["summary_generated"] = True

        # Update previous question if it's a new question
        if is_new_question(user_query):
            session["previous_question"] = user_query

        # Generate guidance based on the current query without using chat history for current response
        guidance = guide_student(
            summary=session["summary_content"],
            student_input=user_query,
            lesson=lesson,
            subtopic=subtopic,
            syllabus_response_content=syllabus_response_content,
            chat_history=st.session_state.messages  # Pass chat history for assessment, not for generating response
        )

        print("guidance=============" + guidance)
        return guidance

    except Exception as e:
        return f"I encountered an error: {str(e)}. Could you please try again with a different problem?"


# Main Chat Interface
# Display chat messages from history on app return
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input and immediate response
if user_query := st.chat_input("How can I help?"):
    session_id = st.session_state.current_chat_title

    # Initialize session if needed
    if session_id not in st.session_state.sessions:
        st.session_state.sessions[session_id] = {
            "counter": 0,
            "summary_generated": False,
            "summary_content": "",
            "previous_question": "",
            "expecting_response": False
        }

    session = st.session_state.sessions[session_id]
    session["counter"] += 1

    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    st.session_state.messages.append({"role": "user", "content": user_query})
    syllabus_response_content = ""
    lesson = lesson_selector()
    pdfFilePath = curriculum_guide_selector("grade-9", "mathematics", lesson)
    subtopic_raw = identify_subtopic("grade-9", "mathematics", lesson, user_query)
    subtopic = re.sub(r'\s+', '-', subtopic_raw.strip().lower())
    print(subtopic)

    rag_chain = create_rag_chain(
        pdf_file_path=pdfFilePath,
        llm=llm,
        embedding_model=embedding_model
    )
    # Step 1: Use the RAG chain to retrieve syllabus content related to the student's question
    syllabus_response = rag_chain.invoke({"input": user_query})

    if isinstance(syllabus_response, dict) and 'content' in syllabus_response:
        syllabus_response_content = syllabus_response['content']
    else:
        syllabus_response_content = "Unable to retrieve syllabus content."


    # Generate and display response immediately
    response = process_user_query(user_query, session, pdfFilePath, subtopic, syllabus_response_content)
    # If the student has completed the question, mark it as solved

    # If the student has completed the question, mark it as solved
    if "great job! you did it!" in response.lower():
        common_mistake_corrections = load_prompt("grade-9", "mathematics", lesson, "", "common-mistake-corrections")
        examples = generate_examples(user_query, common_mistake_corrections)
        print("examples: " + examples)
        # Using session state to keep the examples visible after clicking the button
        st.session_state.show_examples_clicked = True

        # Display "Show Examples" if clicked
        if st.session_state.show_examples_clicked:
            with st.expander("Examples to Help You Understand Better"):
                st.markdown(examples)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Save chat history after each interaction
    save_chat_history_key(st.session_state.current_chat_title, st.session_state.messages)

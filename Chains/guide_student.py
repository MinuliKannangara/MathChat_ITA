import os
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from Configs.config import settings
import chainlit as cl

model = ChatOpenAI(model=settings.MODEL_NAME, streaming=True)

# Store for chat histories
chat_histories = {}
base_path = "Prompts"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


def load_prompt(base_path, grade, subject, topic, sub_topic, prompt_type):
    """
    Load the appropriate prompt template based on the provided grade, subject, topic, and prompt type.
    """

    # Handle the common mistake correction type differently due to its folder structure
    if prompt_type == "common-mistake-corrections":
        prompt_file_path = os.path.join(base_path, prompt_type, subject, f"{topic}.txt")
    else:
        prompt_file_path = os.path.join(base_path, prompt_type, grade, subject, topic, f"{sub_topic}.txt")

    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, 'r') as file:
            prompt_data = file.read()
            return prompt_data
    else:
        return None


# Path to the general guidance file
guidance_path = "Prompts/common-guidance/general-guidance.txt"
common_mistakes_path = ""
# Load general guidance content
guidance_data = ""
if not os.path.exists(guidance_path):
    guidance_data = "Prompt not found"
else:
    with open(guidance_path, 'r') as file:
        guidance_data = file.read()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

def identify_subtopic(base_path,grade, subject, topic,prompt_type):

    prompt_file_path = os.path.join(base_path, prompt_type,grade, subject, f"{topic}.txt")

def guide_student(summary, student_input):
    # Use the summary content directly in the system prompt to help guide the student.
    session_id = cl.user_session.get("id")
    print("inside guide_student")

    # Extract relevant subtopics from the summary
    subtopic = None
    if "Subtopics:" in summary:
        subtopic_parts = summary.split("Subtopics:")
        if len(subtopic_parts) > 1:
            subtopic_line = subtopic_parts[1].strip().split("\n")[0]
            if subtopic_line:
                subtopic = subtopic_line
    if not subtopic:
        subtopic = "General"  # Default case if subtopic extraction fails
    print("guidance data----------------" + guidance_data)
    # Common Guidance: Applicable to all subtopics
    common_guidance = guidance_data

    # Common Mistake Corrections
    common_mistake_corrections = (
        load_prompt(base_path,"grade-9", "mathematics", "algebraic-expressions", "", "common-mistake-corrections")
    )

    # Subtopic-Specific Guidance
    subtopic_specific_guidance = load_prompt(base_path,"grade-9", "mathematics", "algebraic-expressions", subtopic, "teaching-strategies")

    # Combine common and subtopic-specific guidance
    full_guidance_prompt = (
            common_guidance + "\n\n" +
            subtopic_specific_guidance + "\n\n" +
            common_mistake_corrections
    )

    # Generate response using the combined prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an intelligent tutoring assistant (ITA) helping a Grade 9 student in Sri Lanka solve 
                   mathematics problems based on their school syllabus. Your role is to guide the student through the 
                   problem-solving process step by step, without directly providing the solution. Always respond with a 
                   question or prompt to encourage the student's independent thinking.

                   Summary of the problem, syllabus content and steps to solve it:
                   {summary} """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", full_guidance_prompt),
        ]
    )
    chat_history = get_session_history(session_id)

    chain = prompt | model

    cl.user_session.set("chain", chain)

    response = chain.invoke({
        "input": student_input,
        "chat_history": chat_history.messages
    })

    # Add the student's input and the assistant's response to the chat history
    chat_history.add_user_message(student_input)
    chat_history.add_ai_message(response.content)

    print("Guide response: " + response.content)

    return response.content

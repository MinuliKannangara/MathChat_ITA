from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from Configs.config import settings
import chainlit as cl

model = ChatOpenAI(model=settings.MODEL_NAME, streaming=True)
session_id = settings.SESSION_ID
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    print("inside get_session_history")
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def guide_student(summary, student_input):
    # Use the summary content directly in the system prompt to help guide the student.
    print("inside guide_student")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an intelligent tutoring assistant (ITA) helping a Grade 9 student in Sri Lanka solve 
                mathematics problems based on their school syllabus. Your role is to guide the student through the 
                problem-solving process step by step, without directly providing the solution. Always respond with a 
                question or prompt to encourage the student's independent thinking.

                Below is a summary of the student's problem and the steps involved in solving it. First, identify the 
                question posed by the student. Then, use the information from the syllabus to guide the student, 
                ensuring that your guidance adheres to the syllabus and follows these guidelines:

                Summary of the problem, syllabus content and steps to solve it:
                {summary}

                Guidelines:
                1. Start by asking a suitable question to start the conversation with the student based on the given 
                problem.
                2. Ask the student if they have any initial thoughts on how to approach the problem. If they do, 
                   guide them based on their thoughts. If not, provide a hint or explain a relevant concept.
                3. Guide the student to identify the steps needed, one at a time, by asking questions.
                4. If the student suggests an incorrect step, don't directly correct them. Instead, ask them to 
                reconsider 
                   or provide a hint that leads them to the correct approach.
                5. If the student is stuck, explain the relevant mathematical concepts with simple examples before 
                   returning to the problem.
                6. After each step, ask what they think should be done next.
                7. Provide positive reinforcement for correct steps and good reasoning.
                8. Never solve any part of the problem directly. Always ask the student to perform the calculations 
                   and explain their thinking.
                9. Keep track of the conversation history and avoid repeating questions that have already been answered.

                Remember, your role is to guide and prompt, not to solve. Always respond with a question or a prompt that 
                encourages the student to think and take the next step in solving the problem.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
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

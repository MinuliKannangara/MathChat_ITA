import os
from typing import Dict, Optional

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from Configs.config import settings
from Chains.solve_problem import solve_problem
from Chains.guide_student import guide_student

load_dotenv()

api_key = settings.OPENAI_API_KEY
model = ChatOpenAI(model=settings.MODEL_NAME, streaming=True)

# Store session data
sessions: Dict[str, Dict] = {}

# Access the variables
client_id = os.getenv('OAUTH_GOOGLE_CLIENT_ID')
client_secret = os.getenv('OAUTH_GOOGLE_CLIENT_SECRET')

print("Client ID: ", client_id)  # Just to check if the variables are loaded


@cl.oauth_callback
def oauth_callback(
        provider_id: str,
        token: str,
        raw_user_data: Dict[str, str],
        default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


@cl.on_chat_start
async def start():
    session_id = cl.user_session.get("id")
    sessions[session_id] = {
        "counter": 0,
        "summary_generated": False,
        "summary_content": ""
    }
    await cl.Message(
        content="Welcome to the Math Assistant! I can help you solve math problems step by step. What problem would you like help with?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("id")
    if session_id not in sessions:
        await cl.Message(content="Session not found. Please restart the conversation.").send()
        return

    session = sessions[session_id]
    session["counter"] += 1

    print("inside main, session:" + session_id)

    try:
        if not session["summary_generated"]:
            print("summary not generated")
            summary = solve_problem(message.content)

            if isinstance(summary, dict) and 'messages' in summary:
                session["summary_content"] = '\n'.join(map(str, summary['messages']))  # Ensure all messages are strings
                session["summary_generated"] = True

                # Extract subtopic from the summary
                subtopic = "General"
                if "Subtopics:" in session["summary_content"]:
                    subtopic_parts = session["summary_content"].split("Subtopics:")
                    if len(subtopic_parts) > 1:
                        subtopic_line = subtopic_parts[1].strip().split("\n")[0]
                        if subtopic_line:
                            subtopic = subtopic_line

                print(
                    "topics --------------------------------------------------------------------------------" + subtopic)
                guidance = guide_student(session["summary_content"], message.content)
                await cl.Message(content=guidance).send()
            else:
                print("can't process the summary correctly.")
                await cl.Message(
                    content="I'm sorry, I couldn't process that problem. Could you please rephrase it?").send()
        else:
            print("start guiding")
            guidance = guide_student(session["summary_content"], message.content)
            await cl.Message(content=guidance).send()

    except Exception as e:
        await cl.Message(
            content=f"I encountered an error: {str(e)}. Could you please try again with a different problem?").send()


@cl.on_chat_end
async def end():
    session_id = cl.user_session.get("id")
    if session_id in sessions:
        del sessions[session_id]


if __name__ == "__main__":
    cl.run()

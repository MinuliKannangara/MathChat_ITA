import os
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

@cl.on_chat_start
async def start():
    await cl.Message(
        "Welcome to the Math Assistant! I can help you solve math problems step by step. What problem would you like "
        "help with?").send()


# To track if the summary has been generated
summary_generated = False
summary_content = ""  # hold the summary after it's generated


@cl.on_message
async def main(message: cl.Message):
    global summary_generated, summary_content
    try:
        # Check if the summary has already been generated
        if not summary_generated:
            print("insode main: summary is not generated")
            # Call solve_problem function to get the summary
            summary = solve_problem(message.content)

            # Ensure that summary is a dictionary before proceeding
            if isinstance(summary, dict) and 'messages' in summary:

                summary_content = '\n'.join(summary['messages'])
                summary_generated = True

                guidance = guide_student(summary_content, message.content)
                await cl.Message(content=guidance).send()
            else:
                await cl.Message(
                    content="I'm sorry, I couldn't process that problem. Could you please rephrase it?").send()
        else:

            guidance = guide_student(summary_content, message.content)
            await cl.Message(content=guidance).send()

    except Exception as e:
        await cl.Message(
            content=f"I encountered an error: {str(e)}. Could you please try again with a different problem?").send()


if __name__ == "__main__":
    cl.app()

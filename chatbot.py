import os
from operator import itemgetter
import chainlit as cl
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, Graph
from sympy import *

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
store = {}

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


config = {"configurable": {"session_id": "abc2"}}


# Function to create RAG chain
def create_rag_chain(pdf_path, embedding_model, llm, chunk_size=400, chunk_overlap=50):
    # Step 1: Load the PDF document (Syllabus)
    loader = PyPDFLoader("C:/Users/Minuli/PycharmProjects/MathChat/Algebraic_expressions.pdf")
    docs = loader.load()

    # Step 2: Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3: Split the documents into chunks
    splits = text_splitter.split_documents(docs)

    # Step 4: Create a vector store from the document chunks
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

    # Step 5: Create a retriever object to get data from the vector store
    retriever = vectorstore.as_retriever()

    # Step 6: Define the system prompt
    system_prompt = (
        "You are an intelligent chatbot. Use the following context to  retrieve the content related to the given "
        "problem from the syllabus\n\n{context}"
    )

    # Step 7: Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Step 8: Create the QA chain
    qa_chain = create_stuff_documents_chain(llm, prompt)

    # Step 9: Create the RAG chain (QA chain + retriever)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # Step 10: Return the RAG chain
    return rag_chain


def solve_problem(user_question):
    # Define AgentState
    AgentState = {"messages": []}

    # Initialize the RAG chain for syllabus retrieval
    rag_chain = create_rag_chain(
        pdf_path="path_to_syllabus.pdf",
        embedding_model=embedding_model,
        llm=model
    )

    # Step 1: Use the RAG chain to retrieve syllabus content related to the student's question
    syllabus_response = rag_chain.invoke({"input": user_question})

    if isinstance(syllabus_response, dict) and 'content' in syllabus_response:
        syllabus_response_content = syllabus_response['content']
    else:
        syllabus_response_content = "Unable to retrieve syllabus content."

    # Define the node functions
    def reasoning(state):
        messages = state['messages']
        user_input = messages[-1]

        # Prompt refined for clarity and meaning
        complete_query = (
            f"Analyze the student's problem and establish a clear objective. Use relevant theories and concepts from "
            f"the syllabus ({syllabus_response_content}) to guide your reasoning."
            f"Ensure that the reasoning process follows the methods outlined in the syllabus and stays within the "
            f"scope of the curriculum. Break the problem down into logical steps"
            f"to guide the student through understanding how to approach and solve it. However, do not provide the "
            f"final solution."
            f"Focus on identifying and explaining the steps involved, whether the problem is a mathematical "
            f"expression or a word problem."
            f"User's query: {user_input}"
        )

        response = model.invoke(complete_query)
        state['messages'].append(response.content)
        return state

    def condition(state):
        messages = state['messages']
        reasoning_data = messages[-1]
        math_problem = messages[0]
        complete_query = (
            f"Identify all the necessary conditions required to solve the algebraic problem provided by the "
            f"student. Make sure to base the conditions on the logical reasoning provided. "
            f"Problem: {math_problem}. Reasoning: {reasoning_data}.")
        response = model.invoke(complete_query)
        state['messages'].append(response.content)
        return state

    def judge(state):
        messages = state['messages']
        reasoning_data = messages[1]
        math_problem = messages[0]
        conditions = messages[-1]
        complete_query = (
            f"Check whether the identified conditions are correct and sufficient to solve the algebraic problem. "
            f"Assess whether all conditions are logically consistent with the reasoning provided and are enough "
            f"to lead to a solution. "
            f"Problem: {math_problem}. Reasoning: {reasoning_data}. Conditions: {conditions}.")
        response = model.invoke(complete_query)
        state['messages'].append(response.content)
        return state

    def newCondition(state):
        messages = state['messages']
        reasoning_data = messages[1]
        math_problem = messages[0]
        conditions = messages[2]
        judge = messages[-1]
        complete_query = (
            f"Generate new conditions to solve the algebraic problem if the judge tool has identified that the "
            f"existing conditions are incorrect or insufficient. Remove any incorrect conditions. "
            f"Provide the correct set of conditions to solve the problem. "
            f"Problem: {math_problem}. Reasoning: {reasoning_data}. Conditions: {conditions}. "
            f"Judge Assessment: {judge}.")
        response = model.invoke(complete_query)
        state['messages'].append(response.content)
        return state

    def algebraic_solver(state):
        messages = state['messages']
        problem = messages[0]
        reasoning_data = messages[1]
        refined_conditions = messages[-1]
        complete_query = (
            f"Based on the problem, reasoning and conditions, formulate the algebraic expression or equation to solve. "
            f"If it's an expression to evaluate, provide the expression and any variable values. "
            f"If it's an equation to solve, provide the equation. "
            f"Problem: {problem} \nReasoning: {reasoning_data} \nConditions: {refined_conditions}"
        )
        response = model.invoke(complete_query)
        formulation = response.content
        try:
            if "=" in formulation:
                left, right = formulation.split('=')
                eq = Eq(simplify(left.strip()), simplify(right.strip()))
                solution = solve(eq)
            elif "expand" in problem.lower():
                expr = simplify(formulation)
                solution = expand(expr)
            else:
                expr, *var_values = formulation.split(',')
                expr = simplify(expr.strip())
                var_dict = {}
                for var_value in var_values:
                    var, value = var_value.split('=')
                    var_dict[var.strip()] = float(value.strip())
                solution = expr.evalf(subs=var_dict)
            state['solution'] = str(solution)
            state['messages'].append(solution)
        except Exception as e:
            state['solution'] = f"Error in solving: {str(e)}"
            state['messages'].append(str(e))
        return state

    def summarize(state):
        messages = state['messages']
        problem = messages[0]
        reasoning_data = messages[1]
        refined_conditions = messages[4]
        solution = messages[-1]
        summary_prompt = f"""
        Summarize the algebraic problem-solving process:

        Original problem: {problem}
        syllabus_content:{syllabus_response_content}
        Reasoning: {reasoning_data}
        Conditions: {refined_conditions}
        Solution: {solution}

        Provide a step-by-step explanation of how we arrived at the final answer.
        """
        response = model.invoke(summary_prompt)
        state['messages'].append(response.content)
        return state

    # Define the Langchain graph
    workflow = Graph()

    workflow.add_node("reasoning", reasoning)
    workflow.add_node("condition_extractor", condition)
    workflow.add_node("judge_conditions", judge)
    workflow.add_node("new_condition_generator", newCondition)
    workflow.add_node("algebraic_solver", algebraic_solver)
    workflow.add_node("summarize", summarize)

    # Define the edges to connect the nodes
    workflow.add_edge('reasoning', 'condition_extractor')
    workflow.add_edge('condition_extractor', 'judge_conditions')
    workflow.add_edge('judge_conditions', 'new_condition_generator')
    workflow.add_edge('new_condition_generator', 'algebraic_solver')
    workflow.add_edge('algebraic_solver', 'summarize')

    workflow.set_entry_point("reasoning")
    workflow.set_finish_point("summarize")

    app = workflow.compile()

    result = app.invoke({"messages": ["solve m when 2m+1=9"]})
    print(result)

    # Return the final summary
    return result


def guide_student(summary, student_input):
    # Use the summary content directly in the system prompt to help guide the student.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an intelligent tutoring assistant (ITA) helping a Grade 9 student in Sri Lanka solve 
                mathematics problems based on their school syllabus. Your role is to guide the student through the 
                problem-solving process step by step, without directly providing the solution. Always respond with a 
                question or prompt to encourage the student's independent thinking.

                Below is a summary of the student's problem and the steps involved in solving it. First, identify the question posed 
                by the student. Then, use the information from the syllabus to guide the student, ensuring that your guidance adheres 
                to the syllabus and follows these guidelines:

        Summary of the problem, syllabus content and steps to solve it:
        {summary}

        Guidelines:
        1. Start by asking a suitable question to start the conversation with the student based on the given problem.
        2. Ask the student if they have any initial thoughts on how to approach the problem. If they do, 
           guide them based on their thoughts. If not, provide a hint or explain a relevant concept.
        3. Guide the student to identify the steps needed, one at a time, by asking questions.
        4. If the student suggests an incorrect step, don't directly correct them. Instead, ask them to reconsider 
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

    chat_history = get_session_history(config["configurable"]["session_id"])

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

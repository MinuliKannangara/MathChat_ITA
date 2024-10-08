from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import Graph

from Chains.rag_chain import create_rag_chain
from Configs.config import settings
from sympy import *

model = ChatOpenAI(model=settings.MODEL_NAME, streaming=True)
# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

def solve_problem(user_question):
    # Define AgentState
    AgentState = {"messages": []}

    print("inside solve_problem")

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

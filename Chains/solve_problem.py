from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import Graph
from Chains.rag_chain import create_rag_chain
from Configs.config import settings
from sympy import *

model = ChatOpenAI(model=settings.MODEL_NAME, streaming=True)


def solve_problem(user_question, pdf_file_path, sub_topic, syllabus_response_content):
    # Define AgentState
    AgentState = {"messages": []}

    print("inside solve_problem")

    def reasoning(state):
        print("def 1 ---------------")
        messages = state['messages']
        user_input = messages[-1]

        # Modified prompt to analyze the student's question and provide guidance
        complete_query = (
            f"Analyze the student's problem and provide a structured response. "
            f"First, review the following syllabus content to identify the relevant subtopic(s) for this question:\n"
            f"{syllabus_response_content}\n\n"
            f"Format your response as follows:\n"
            "---ANALYSIS---\n"
            "<Your detailed analysis and guidance>\n\n"
            f"When providing analysis:\n"
            f"1. Use relevant theories and concepts from the syllabus\n"
            f"2. Follow the methods outlined in the syllabus\n"
            f"3. Break down the logical steps needed\n"
            f"4. Do not provide the final solution\n"
            f"5. Focus on explaining the approach\n\n"
            f"User's query: {user_question}"
        )

        response = model.invoke(complete_query)

        # Extract response content
        content = response.content if hasattr(response, 'content') else ""
        analysis = ""

        # Check if content exists and extract the analysis
        if content:
            analysis = content.strip()

        # Update state with new information
        if analysis:
            state['messages'].append(analysis)
        else:
            state['messages'].append("I couldn't find a clear analysis. Let's try a different approach.")

        return state

    def condition(state):
        print("def 2 ---------------")
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
        print("def 3 ---------------")
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
        print("def 4 ---------------")
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
        print("def 5 ---------------")
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
        print("def 6 ---------------")
        messages = state['messages']

        # Ensure there are enough messages to avoid indexing errors
        if len(messages) < 5:
            state['messages'].append(
                "I couldn't find enough information to create a summary. Let's revisit the problem.")
            return state

        # Convert all message components to strings to avoid type errors
        problem = str(messages[0])
        reasoning_data = str(messages[1])
        refined_conditions = str(messages[4])
        solution = str(messages[-1])

        summary_prompt = f"""
        Summarize the algebraic problem-solving process:

        Original problem: {problem}
        Reasoning: {reasoning_data}
        Conditions: {refined_conditions}
        Solution: {solution}

        Provide a step-by-step explanation of how we arrived at the final answer. Add the subtopics to the summary as well.
        """

        response = model.invoke(summary_prompt)
        response_content = response.content if hasattr(response, 'content') else str(response)
        print("summary in summary function ---------------" + response_content)

        # Append the response content to the messages if it's available
        if response_content:
            state['messages'].append(response_content)
        else:
            state['messages'].append("I couldn't generate a summary. Let's try a different approach.")

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
    print(app.get_graph().draw_mermaid())
    result = app.invoke({"messages": [user_question]})
    print(result)

    # Return the final summary
    return result

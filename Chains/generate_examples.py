import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_examples(student_question, common_mistakes):
    """
    Generate examples to explain solving a problem based on potential student mistakes.

    Parameters:
    student_question (str): The question provided by the student.
    common_mistakes (str): Information on common mistakes to consider.

    Returns:
    str: Generated examples to help students solve the given problem.
    """
    # Define the complete prompt to generate examples
    complete_prompt = (
        f"""
        The student asked to solve the problem: {student_question}

        Generate simple examples similar to the given problem to explain the concepts needed to solve it. Use a few-shot prompting method as demonstrated below:

        Example 1: Let's consider a simpler equation similar to the given problem. Suppose we have "2x = 10".
        To isolate x, note that it is multiplied by 2. Therefore, we need to remove the multiplication by dividing both sides by 2.
        This gives us "x = 10 / 2", and hence, "x = 5".

        Example 2: If the given equation involves a negative coefficient, let's look at "-2x = 10".
        In this case, we still need to isolate x by dividing both sides by -2. This means "x = 10 / -2", which results in "x = -5".
        This example shows that when dividing a positive number by a negative, the result is negative.

        Example 3: If we have an equation as x + 5 = 20, this is done by adding 5 to x, and it is equal to 20. So to 
        isolate x, we can remove the addition by subtracting 5 from both sides: "x + 5 - 5 = 20 - 5", then "x = 20 - 
        5", which results in "x = 15".

        Now generate similar examples for the given problem.
        When you generate the examples, consider the mistakes that can happen when solving the given problem. {common_mistakes} Identify the mistakes that can happen and consider them also when generating the examples.

        Ensure that the generated examples are separated by double newlines, so that each example can be easily formatted as:

        Example 1: [description]
        Example 2: [description]
        etc.
        """
    )

    # Load the LLM model
    llm = ChatOpenAI(model_name="gpt-4", streaming=False)


    # Step 7: Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", complete_prompt),
        ("human", "{input}"),
    ])

    # # initialize the output parser (to maintain the same format throughout the chatbot)
    # parser = StrOutputParser()

    # create a chain using the prompt, LLM and the output parser
    chain = prompt | llm

    try:
        # Perform the LLM call
        # Correctly pass the input to the invoke method
        response = chain.invoke({"input": student_question})

        # Return response content directly
        return response.content

    except Exception as e:
        # Handle exceptions and return an error message
        return f"Error generating examples: {str(e)}"

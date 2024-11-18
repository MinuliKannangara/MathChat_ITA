import os
from langchain_openai import ChatOpenAI


def identify_subtopic(grade, subject, lesson, student_question):
    """
    Identifies the subtopic related to a given question by performing an LLM call.

    Parameters:
    - base_path: The base path for the prompt library.
    - grade: The grade of the student (e.g., "grade9").
    - subject: The subject related to the question (e.g., "mathematics").
    - topic: The topic being asked about.
    - prompt_type: The type of prompt to be loaded (e.g., "identify_subtopics").
    - student_question: The question provided by the student.

    Returns:
    - A response string indicating the identified subtopic.
    """

    # prompt_data = load_prompt("grade-9","mathematics", lesson, "","identify-subtopics")
    # prompt_file_path = "Prompts/identify-subtopics/grade-9/mathematics/algebraic-expressions"
    prompt_file_path = os.path.join("Prompts", "identify-subtopics", "grade-9", "mathematics", "algebraic-expressions")
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, 'r') as file:
            prompt_data = file.read()

    else:
        prompt_data = "Prompt not found."

    # Append the student's question to the loaded prompt
    complete_prompt = (
        f"{prompt_data}\n\n"
        f"Now identify the relevant sub topic based on the student's question, note "
        f"that there can be only one selected topic, and write the name of the topic as it given in the prompt. Don't "
        f"do any changes. return only the name of the subtopic, without any other texts.:\n"
        f"Question: {student_question}"
    )

    # Load the LLM model
    model = ChatOpenAI(model="gpt-4", streaming=False)

    # Perform the LLM call
    response = model.invoke(complete_prompt)

    # Extract the response content
    response_content = response.content if hasattr(response, 'content') else str(response)

    return response_content

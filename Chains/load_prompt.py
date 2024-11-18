import os


def load_prompt(grade, subject, lesson, sub_topic, prompt_type):
    """
    Load the appropriate prompt template based on the provided grade, subject, topic, and prompt type.
    """

    # Handle the common mistake correction type differently due to its folder structure
    if prompt_type == "common-mistake-corrections":
        prompt_file_path = os.path.join("Prompts", prompt_type, subject, f"{lesson}.txt")
    elif prompt_type == "identify-subtopics":
        prompt_file_path = os.path.join("Prompts", prompt_type, grade, subject, f"{lesson}.txt")
    elif prompt_type == "common-guidance":
        prompt_file_path = os.path.join("Prompts", prompt_type, subject, "general-guidance.txt")
    else:
        prompt_file_path = os.path.join("Prompts", prompt_type, grade, subject, lesson, f"{sub_topic}.txt")
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, 'r') as file:
            prompt_data = file.read()
            return prompt_data
    else:
        return "Prompt not found."

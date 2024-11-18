from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from Configs.config import settings
from Chains.load_prompt import load_prompt
import streamlit as st

# Initialize the OpenAI model
model = ChatOpenAI(model=settings.MODEL_NAME, streaming=True)


def assess_student_proficiency(chat_history: List[Dict[str, str]]) -> str:
    """Determine the student's proficiency based on recent chat history."""
    if not chat_history:
        return "beginner"

    recent_responses = chat_history[-3:]
    if all("correct" in entry["content"].lower() or "great job" in entry["content"].lower() for entry in
           recent_responses):
        return "advanced"
    elif any("struggling" in entry["content"].lower() or "confusing" in entry["content"].lower() for entry in
             recent_responses):
        return "beginner"

    return "beginner"


def generate_dynamic_guidance(proficiency: str, subtopic_specific_guidance: str) -> str:
    """Generate guidance dynamically based on the student's proficiency level."""
    if proficiency == "beginner":
        return f"Let's take it step-by-step. {subtopic_specific_guidance} Remember, take your time and ask questions if you're confused."
    elif proficiency == "intermediate":
        return f"You're doing well! {subtopic_specific_guidance} Let's try to solve this next step together."
    elif proficiency == "advanced":
        return f"Excellent! Since you're getting the hang of this, {subtopic_specific_guidance} Let's see if you can tackle this step without hints."


def guide_student(summary: str, student_input: str, lesson: str, subtopic: str, syllabus_response_content: str, chat_history: List[Dict[str, str]]) -> str:
    """Guide the student based on input, lesson summary, and chat history."""
    print("guide student")
    # Load guidance content
    guidance_data = load_prompt("", "mathematics", "", "", "common-guidance")
    common_mistake_corrections = load_prompt("grade-9", "mathematics", lesson, "",
                                             "common-mistake-corrections")
    subtopic_specific_guidance = load_prompt("grade-9", "mathematics", lesson, subtopic,
                                             "teaching-strategies")

    # Assess student's proficiency level
    student_proficiency = assess_student_proficiency(chat_history)

    print("student proficiency:", student_proficiency)

    # Combine guidance prompts and dynamically adjust based on proficiency
    dynamic_guidance_prompt = generate_dynamic_guidance(student_proficiency, subtopic_specific_guidance)

    print("dynamic guidance:", dynamic_guidance_prompt)

    # Combine all guidance prompts with common mistake corrections
    full_guidance_prompt = (
            guidance_data + "\n\n" +
            dynamic_guidance_prompt + "\n\n" +
            common_mistake_corrections
    )

    # Create prompt template including chat history
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are an intelligent tutoring assistant (ITA) helping a Grade 9 student in Sri Lanka solve 
                mathematics problems based on their school syllabus. Your role is to guide the student through the 
                problem-solving process step by step, without directly providing the solution. Always respond with a 
                question or prompt to encourage the student's independent thinking.
                
                After the student solve the question and he provide the final correct answer, Always add the phrase 
                "Great job! you did it!"

                {full_guidance_prompt}
                
                Syllabus content to refer when guide the student:
                {syllabus_response_content}

                Summary of the problem, syllabus content, and steps to solve it:
                {summary}"""
        ),
        # Include past interactions from chat history to provide context
        MessagesPlaceholder(variable_name="chat_history"),
        # Current student input
        ("human", student_input)
    ])

    # Generate response using chat history
    chat_history_messages = [{"role": entry["role"], "content": entry["content"]} for entry in chat_history]
    chain = prompt | model

    # Invoke the chain to generate the response with chat history as context
    response = chain.invoke({
        "input": student_input,
        "chat_history": chat_history_messages
    })

    return response.content

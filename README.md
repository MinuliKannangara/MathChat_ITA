# 📚 Intelligent Tutoring Assistant (ITA) for Personalized Mathematics Education in Sri Lanka

**Development of a Chat Assistant Using LLMs for Personalized Mathematics Tutoring to Overcome Educational Disparities in Sri Lanka**  
🔬 *Individual Undergraduate Research Project*  
📅 Published in **ICARC 2025** – 5th International Conference on Advanced Research in Computing  
🏛️ *Sabaragamuwa University of Sri Lanka (IEEE Conference)*
🔗 [View the IEEE Xplore publication](https://ieeexplore.ieee.org/document/10963323)
---

## 🧠 Overview

This project presents an **Intelligent Tutoring Assistant (ITA)** powered by **Large Language Models (LLMs)** to provide **personalized mathematics tutoring** for Grade 9 students in Sri Lanka.  

Focusing on the topic **"Algebraic Expressions"**, the ITA aligns with the **Sri Lankan national curriculum** and incorporates **iterative teacher feedback** to deliver curriculum-relevant, step-by-step guidance. It bridges the gap between traditional classroom instruction and the need for individualized student support.

---

## ✨ Features

- 🧮 Personalized tutoring in Algebraic Expressions
- 🇱🇰 Aligned with the Sri Lankan Grade 9 mathematics syllabus
- 🗣️ Text-based input support
- 🔁 Step-by-step problem-solving and feedback
- 🎯 Encourages critical thinking and independent learning
- 🧩 Supports subtopic classification and custom guidance
- 🧠 Powered by GPT-4 and LangChain with LangGraph workflow

---

## 📌 Table of Contents

- [Methodology](#-methodology)
- [Architecture](#-architecture)
- [System Workflow](#-system-workflow)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [User Interface](#-user-interface)

---

## 🧪 Methodology

This research follows the **Design Science Research (DSR)** methodology to iteratively design, develop, and evaluate the ITA:

1. **Problem Identification:** Lack of personalized learning in mathematics.
2. **Objective Definition:** Design an LLM-powered Intelligent Tutoring Assistant.
3. **Design & Development:** Built and improved the ITA through iterative feedback.
4. **Demonstration:** Two rounds of live demonstrations with teachers.
5. **Evaluation:** Analyzed feedback through narrative evaluation.
6. **Communication:** Published findings in IEEE (ICARC 2025).

---

## 🏗 Architecture

The ITA integrates several components working together to analyze student queries, select appropriate content, and guide students through problem-solving:

- **Lesson Selector**  
- **Curriculum Guide Selector + Integrator (RAG + Vector DB)**  
- **Subtopic Identifier (LLM with few-shot learning)**  
- **Problem Solver (LangGraph + Python logic)**  
- **Guidance System (step-by-step coaching)**  
- **Prompt Selector (with multiple categories)**  
- **LLM Integration (GPT-4)**  

---

## 🔁 System Workflow

1. **Grade & Subject Selection**  
   - Students select Grade 9 → Mathematics → Algebraic Expressions

2. **Query Submission & Analysis**  
   - Via text/voice → analyzed to detect lesson and subtopic

3. **Curriculum Integration**  
   - RAG-based retrieval from pre-processed syllabus documents

4. **Problem-Solving & Reasoning**  
   - Using LangGraph and Python to build structured solution path

5. **Step-by-Step Guidance Delivery**  
   - Interactive hints, analogies, and explanations for student learning

---

## ⚙️ How It Works

### 🧩 Key System Components

- **Curriculum Integrator:** Embeds and retrieves syllabus content via ChromaDB
- **Subtopic Identifier:** Classifies questions using example-driven prompts
- **Problem Solver (LangGraph):** Modular agent-based structure for reasoning, condition extraction, and computation
- **Guidance Student Agent:** Offers personalized step-by-step guidance
- **Prompt Selector:** Dynamically selects prompts from 5 categories:
  - General guidance
  - Mistake corrections
  - Teaching strategies
  - Subtopic identification
  - Example generation

### 🧠 LLM-Powered Reasoning Example:
> “Imagine you have 3 baskets with 9 apples in total. To find how many apples are in each basket, you divide 9 by 3 = 3.”  
> This analogy helps students understand **division** in the context of **solving `3x = 9`**.

---

## 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| Language Model | GPT-4 (OpenAI) |
| Framework | LangChain, LangGraph |
| Backend | Python |
| Vector Store | ChromaDB |
| Frontend | Streamlit |
| Methodology | Design Science Research (DSR) |

---

## 🖼️ User Interface

The system features a clean, chat-based interface built using **Streamlit**:

- 🌐 Text input
- 💬 Past conversation history
- 🔁 Real-time tutoring feedback
- 📚 Curriculum-linked content

---

> 📌 *This project contributes to improving equitable access to quality education using cutting-edge AI technologies tailored to local educational contexts.*


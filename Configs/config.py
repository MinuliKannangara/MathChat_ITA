import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    PDF_PATH = "C:/Users/Minuli/PycharmProjects/MathChat/Algebraic_expressions.pdf"
    MODEL_NAME = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-3-small"


settings = Settings()

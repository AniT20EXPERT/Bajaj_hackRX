import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)


def answer_question_without_system(question: str, context: str):
    """
    Answer a question based on provided context using Groq's Llama model.

    Args:
        question (str): The question to answer
        context (str): The context to base the answer on

    Returns:
        str: The generated answer
    """
    prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question: {question}
Answer:"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,  # Lower temperature for more focused answers
            max_tokens=1000,  # Adjust based on your needs
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"


# Alternative version with system message for better instruction following
def answer_question_with_system(question: str, context: str):
    """
    Answer a question with system message for better instruction following.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based strictly on the provided context. If the answer cannot be found in the context, say so clearly."
                },
                {
                    "role": "user",
                    "content": f"""Context: {context}

Question: {question}"""
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000,
        )
        # print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"


# Example usage


def answer_question(question: str, context: str):
    return answer_question_with_system(question, context)



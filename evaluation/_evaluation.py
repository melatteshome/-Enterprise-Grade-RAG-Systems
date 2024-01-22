import os
import json
import sys
from openai import OpenAI
from math import exp
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from evaluation._data_generation import file_reader, pdf_reader
from evaluation._data_generation import get_completion

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def evaluate(
    prompt: str, user_message: str, context: str, use_test_data: bool = False
) -> str:
    """Return the classification of the hallucination.
    @parameter prompt: the prompt to be completed.
    @parameter user_message: the user message to be classified.
    @parameter context: the context of the user message.
    @returns classification: the classification of the hallucination.
    """
    num_test_output = str(10)
    API_RESPONSE = get_completion(
        [
            {
                "role": "system",
                "content": prompt.replace("{Context}", context).replace(
                    "{Question}", user_message
                ),
            }
        ],
        model=str("gpt-3.5-turbo-16k"),
        logprobs=True,
        top_logprobs=1,
    )

    system_msg = str(API_RESPONSE.choices[0].message.content)

    for i, logprob in enumerate(
        API_RESPONSE.choices[0].logprobs.content[0].top_logprobs, start=1
    ):
        output = f"\nhas_sufficient_context_for_answer: {system_msg}, \nlogprobs: {logprob.logprob}, \naccuracy: {np.round(np.exp(logprob.logprob)*100,2)}%\n"
        print(output)
        if system_msg == "true" and np.round(np.exp(logprob.logprob) * 100, 2) >= 95.00:
            classification = "true"
        elif (
            system_msg == "false"
            and np.round(np.exp(logprob.logprob) * 100, 2) >= 95.00
        ):
            classification = "false"
        else:
            classification = "false"
    return classification


if __name__ == "__main__":
    context_message = pdf_reader("rag/data/RAG.pdf")
    prompt_message = file_reader("prompts/generic_evaluation_prompt.txt")
    context = str(context_message)
    prompt = str(prompt_message)

    user_message = str(input("question: "))

    print(evaluate(prompt, user_message, context))

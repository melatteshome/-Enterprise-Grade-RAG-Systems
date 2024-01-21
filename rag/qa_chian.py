from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from load_to_chroma import Load_VD
import os


class Chat:
    def __init__(self) -> None:
        pass

    def connect_openai():
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-16k")
        return chat

    def qa_chain(self):
        retriver = Load_VD()
        primary_qa = self.connect_openai()
        qa_chain = RetrievalQA.from_chain_type(
            primary_qa, retriver, return_source_documents=True
        )
        return qa_chain


Chat

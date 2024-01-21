import sys
import os
import streamlit as st

# Add parent directory of rag to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rag.chat import chat
from rag.load_to_chroma import Load_VD

with st.sidebar:
    uploadedfile = st.file_uploader("upload your pdf", type=["pdf"])
    if uploadedfile:
        with open("uploaded_file.txt", "wb") as file:
            file.write(uploadedfile.getvalue())
            path = os.path.abspath("uploaded_file.txt")
            Load_VD(file=path)

        st.success("File uploaded successfully!")


def generate_output(input_text):
    chat_obj = chat()
    output_text = chat_obj.qa_chain(query=input_text)
    return output_text


def main():
    st.title("RAG")
    input_text = st.text_input("Ask:")

    if st.button("Generate Output"):
        output_text = generate_output(input_text)
        st.write("Output:")
        st.write(output_text["result"])


if __name__ == "__main__":
    main()

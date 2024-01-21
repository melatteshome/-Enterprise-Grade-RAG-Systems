import streamlit as st


def generate_output(input_text):
    # Logic to generate the output text based on the input
    output_text = "You entered: " + input_text
    return output_text


def main():
    st.title("RAG")
    input_text = st.text_input("Enter text:")

    if st.button("Generate Output"):
        output_text = generate_output(input_text)
        st.write("Output:")
        st.write(output_text)


if __name__ == "__main__":
    main()

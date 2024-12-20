from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import streamlit as st
from transformers import pipeline   


def get_llm_response(form_input, email_sender, email_recipient, email_style):
   # Load a HuggingFace model online for text generation
    generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

    # Template for building the PROMPT
    template = """
    Write a email with {style} style and includes topic:{email_topic}.
    Sender: {sender}\n
    Recipient: {recipient}
    \n\nEmail Text:
    """

     # Creating the final PROMPT
    prompt = template.format(
        email_topic=form_input,
        sender=email_sender,
        recipient=email_recipient,
        style=email_style
    )

    # Generating the response using the HuggingFace model
    response = generator(prompt, max_length=256, temperature=0.01)
    # Extracting the generated text
    generated_text = response[0]["generated_text"]

    return generated_text



def main():
    st.set_page_config(page_title="Generate Emails",
                       layout='centered',
                       initial_sidebar_state='collapsed')
    st.header("Generate Emails @@@@")
    email_topic = st.text_area('Enter the email topic', height=275)

    # Creating columns for the UI - To receive inputs from user
    col1, col2, col3 = st.columns([10, 10, 5])
    with col1:
        email_sender = st.text_input('Sender Name')
    with col2:
        email_recipient = st.text_input('Recipient Name')
    with col3:
        email_style = st.selectbox('Writing Style',
                                   ('Formal', 'Appreciating', 'Not Satisfied', 'Neutral'),
                                   index=0)
    submit = st.button("Generate")
    if submit:
        email = get_llm_response(
            form_input=email_topic,
            email_sender=email_sender,
            email_recipient=email_recipient,
            email_style=email_style
        )
        st.write(email)


if __name__ == '__main__':
    main()

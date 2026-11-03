import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from model_selector import ModelSelector
from pdf_processor import PDFProcessor
from rag_system import RAGSystem


## MAIN FLOW
def main():
    # init_session_state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'embedding_selected' not in st.session_state:
        st.session_state.current_embedding_selected = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

    ### Left SideBar MODEL SELECTION
    selector = ModelSelector()
    st.sidebar.title("Models Selection:")

    llm_selected = st.sidebar.radio("Choose an LLM Model: ",
                                    options=list(selector.llm_models.keys()),
                                    format_func=lambda x: selector.llm_models[x]
                                    )
    st.sidebar.write(f"=> {llm_selected} selected!")

    embedding_selected = st.sidebar.radio("Choose an Embedding Model: ",
                                          options=list(selector.embedding_models.keys()),
                                          format_func=lambda x: selector.embedding_models[x]["name"]
                                          )
    st.sidebar.write(f"=> {embedding_selected} selected!")

    # If embedding models changed?
    if embedding_selected != st.session_state.current_embedding_selected:
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_selected = embedding_selected
        st.session_state.rag_system = None
        st.warning("Embedding Model changed. Please upload your documents again.")

    # Initialize RAG system
    if st.session_state.rag_system is None:
        st.session_state.rag_system = RAGSystem(embedding_model=embedding_selected, llm_model=llm_selected)
        st.session_state.collection_name = st.session_state.rag_system.collection_name
        st.info(f"Using collection name '{st.session_state.collection_name}' for storing embeddings.")

    # Upload a pdf file
    pdf_file = st.file_uploader(label="Upload a PDF", type="pdf")
    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        processor = PDFProcessor()
        with st.spinner("Processing your PDF..."):
            try:
                text = processor.read_pdf(pdf_file)
                chunks = processor.create_chunk(pdf_text=text, pdf_file=pdf_file)
                if st.session_state.rag_system.add_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # Query
        if st.session_state.processed_files:
            st.subheader("💻Query your documents")
            query = st.text_input("❓Ask a question:")
            if query:
                with st.spinner("👀Read documents and Generating response..."):
                    results = st.session_state.rag_system.query_documents(query)
                    if results and results["documents"]:
                        response = st.session_state.rag_system.generate_response(query=query,
                                                                                 context=results["documents"][0])
                        if response:
                            st.subheader("📝Answer:")
                            st.write(response)
                            with st.expander("Source References:"):
                                for idx, doc in enumerate(results["documents"][0], 1):
                                    st.markdown(f"Ref {idx}")
                                    st.info(doc)
    else:
        st.info("Please upload a PDF to get started!")


if __name__ == '__main__':
    main()
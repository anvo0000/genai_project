import streamlit as st
import pandas as pd
import time
from typing import List, Tuple
from io import StringIO
from rag_app import RAGPipeline
from embedding_model import EmbeddingModel
from llm_model import LLMModel

# Session state initialization
def init_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'collection_ready' not in st.session_state:
        st.session_state.collection_ready = False
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'ids' not in st.session_state:
        st.session_state.ids = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


def create_pipeline(embedding_choice: "1", llm_choice: "1", collection_name: str,
                    reset_db: bool) -> RAGPipeline:
    """Create and configure the RAG pipeline"""
    with st.spinner("Initializing models..."):
        # Initialize models, default is Nomic and Ollama
        embedding_model = EmbeddingModel(embedding_choice)
        llm_model = LLMModel(llm_choice)

        # Create pipeline
        pipeline = RAGPipeline(
            embedding_model=embedding_model,
            llm_model=llm_model,
            collection_name=collection_name,
            db_path="./db/chromadb_rag.db"
        )
        # Set reset flag for database setup
        pipeline.reset = reset_db
        return pipeline


def load_data(pipeline: RAGPipeline, data_source: str, file_obj=None) -> Tuple[List[str], List[str]]:
    """Load data into the pipeline from different sources"""
    documents = []
    ids = []

    with st.spinner("Loading data..."):
        if data_source == "CSV File Upload":
            if file_obj is not None:
                # Read uploaded CSV
                csv_content = StringIO(file_obj.getvalue().decode("utf-8"))
                df = pd.read_csv(csv_content, delimiter=";")
                documents = df["details"].to_list()
                ids = [str(i) for i in range(len(documents))]

        elif data_source == "Demo Data":
            # Use built-in demo data (from CSV file)
            documents, ids = pipeline.load_csv()

        elif data_source == "Sample Text":
            # Use sample text data
            documents = [
                "Bananas are rich in potassium, vitamin C, and vitamin B6. They're a great source of energy.",
                "Apples contain fiber, vitamin C, and various antioxidants. They may help reduce the risk of certain diseases.",
                "Spinach is high in iron, vitamin K, and folate. It's a versatile leafy green vegetable.",
                "Salmon is an excellent source of omega-3 fatty acids, protein, and vitamin D. It's good for heart health.",
                "Blueberries are packed with antioxidants and are known for their cognitive benefits."
            ]
            ids = [str(i) for i in range(len(documents))]
    return documents, ids

def setup_database(pipeline: RAGPipeline, documents: List[str]) -> bool:
    """Set up the vector database with the provided documents"""
    try:
        with st.spinner("Setting up database..."):
            pipeline.setup_chromadb(documents)
            return True
    except Exception as e:
        st.error(f"Error setting up database: {str(e)}")
        return False


def process_query(pipeline: RAGPipeline, query: str, top_k: int) -> None:
    """Process a query and display results"""
    if not query:
        st.warning("Please enter a query.")
        return

    with st.spinner("Processing query..."):
        try:
            # Show progress bar
            progress_bar = st.progress(0)

            # Update progress
            progress_bar.progress(30)
            time.sleep(0.3)

            # Process the query
            response, references = pipeline.process_query(query, top_k=top_k)

            # Update progress
            progress_bar.progress(100)
            time.sleep(0.3)
            progress_bar.empty()

            # Add to history
            st.session_state.query_history.append({
                "query": query,
                "response": response,
                "references": references
            })

            # Display the result
            st.markdown(f"<div class='result-box'><h3>Response:</h3>{response}</div>", unsafe_allow_html=True)

            # Display references
            st.markdown("<h4>References:</h4>", unsafe_allow_html=True)
            for i, ref in enumerate(references, 1):
                st.markdown(f"<div class='reference-box'><strong>Source {i}:</strong><br>{ref}</div>",
                            unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")


def display_query_history():
    """Display the history of queries and responses"""
    if not st.session_state.query_history:
        st.info("No query history yet.")
        return

    for i, item in enumerate(reversed(st.session_state.query_history), 1):
        with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {item['query']}"):
            st.markdown(f"**Response:**\n{item['response']}")
            st.markdown("**References:**")
            for j, ref in enumerate(item['references'], 1):
                st.markdown(f"Source {j}: {ref}")

def main():
    init_session_state()
    st.title("RAG Knowledge Base Explorer")
    st.markdown("""
        <div class='info-box'>
        This application demonstrates a Retrieval Augmented Generation (RAG) pipeline.
        Upload your data or use sample data, then ask questions to get AI-generated answers
        based on the relevant information in your knowledge base.
        </div>
        """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])  # Create two columns for the layout
    with col1:
        st.markdown("### Configuration")

        # Model selection
        st.subheader("Model Selection")
        embedding_choice = st.selectbox(
            "Embedding Model",
            options=["1", "2"],
            format_func=lambda x: {
                "1": "Ollama - Nomic Embedded",
                "2": "OpenAI Embedding"
            }.get(x, x)
        )

        llm_choice = st.selectbox(
            "LLM Model",
            options=["1", "2"],
            format_func=lambda x: {
                "1": "Ollama Llama3",
                "2": "OpenAI GPT"
            }.get(x, x)
        )

        # Database configuration
        st.subheader("Database Settings")
        collection_name = st.text_input("Collection Name", value="nutrition_facts")
        reset_db = st.checkbox("Reset Database", value=True)

        # Data source selection
        st.subheader("Data Source")
        data_source = st.radio(
            "Select Data Source",
            options=["CSV File Upload", "Demo Data", "Sample Text"]
        )

        # File upload for CSV
        uploaded_file = None
        if data_source == "CSV File Upload":
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                st.success("File uploaded successfully!")

        # Initialize button
        if st.button("Initialize Pipeline"):
            # Create pipeline
            st.session_state.pipeline = create_pipeline(
                embedding_choice,
                llm_choice,
                collection_name,
                reset_db
            )

            # Load data
            st.session_state.documents, st.session_state.ids = load_data(
                st.session_state.pipeline,
                data_source,
                uploaded_file
            )

            # Setup database
            if st.session_state.documents:
                st.session_state.collection_ready = setup_database(
                    st.session_state.pipeline,
                    st.session_state.documents
                )

                if st.session_state.collection_ready:
                    st.success("Pipeline initialized and data loaded successfully!")
                else:
                    st.error("Failed to set up the database. Please check configuration.")
            else:
                st.error("No documents were loaded. Please check your data source.")

        # Reset button
        if st.button("Reset App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        # Document preview
        if st.session_state.documents:
            with st.expander("Preview Documents"):
                for i, doc in enumerate(st.session_state.documents):
                    st.markdown(f"**Document {i + 1}:** {doc[:150]}...")

    # Main content for querying
    with col2:
        st.markdown("### Ask Questions")

        # Query section
        if st.session_state.collection_ready:
            query = st.text_area("Enter your question", height=100)
            top_k = st.slider("Number of references to retrieve", min_value=1, max_value=5, value=2)

            if st.button("Submit Query"):
                process_query(st.session_state.pipeline, query, top_k)

            # Display query history
            st.markdown("### Query History")
            display_query_history()
        else:
            st.info("Please initialize the pipeline and load data first.")

    # Footer
    st.markdown("---")
    st.markdown("RAG Pipeline Explorer | Built with Streamlit | anvo0000")


if __name__ == '__main__':
    main()
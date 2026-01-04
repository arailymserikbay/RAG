import streamlit as st
from rag_functions import PDFRAGSystem

# Configure Streamlit page
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = PDFRAGSystem()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'pdfs_processed' not in st.session_state:
        st.session_state.pdfs_processed = False

    if 'generated_test' not in st.session_state:
        st.session_state.generated_test = None

def display_sidebar():
    """Display the sidebar with PDF upload and settings"""
    with st.sidebar:
        st.header("📁 Upload PDFs")
        
        # PDF upload
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )

        if uploaded_files:
            if st.button("📚 Process PDFs", type="primary"):
                # Get current settings before processing
                current_chunk_size = st.session_state.get('chunk_size', 1000)
                process_pdfs(uploaded_files, current_chunk_size)

        # Settings
        st.header("⚙️ Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="Size of text chunks for processing")

        # Store settings in session state for use during processing
        st.session_state.chunk_size = chunk_size

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        return chunk_size

def process_pdfs(uploaded_files, chunk_size=1000):
    """Process uploaded PDF files"""
    with st.spinner("Processing PDFs..."):
        try:
            # Load documents
            docs = st.session_state.rag_system.load_pdf_documents(uploaded_files)
            
            if docs:
                # Split documents with user-specified chunk size
                splits = st.session_state.rag_system.split_documents(
                    docs, 
                    chunk_size=chunk_size,
                    chunk_overlap=int(chunk_size * 0.2)  # 20% overlap
                )
                
                # Create vector store
                st.session_state.rag_system.create_vectorstore(splits)

                # Create RAG chain with default number of retrieved chunks
                st.session_state.rag_system.create_rag_chain()
                
                st.session_state.pdfs_processed = True
                st.session_state.chat_history = []  # Clear chat history
                st.success(f"✅ Successfully processed {len(docs)} PDF file(s)")
                st.rerun()
            else:
                st.error("❌ No documents could be processed")
        except Exception as e:
            st.error(f"❌ Error processing PDFs: {str(e)}")

def display_chat_history():
    """Display the chat history"""
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.expander(f"Q{i+1}: {question[:50]}..." if len(question) > 50 else f"Q{i+1}: {question}"):
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {answer}")


def handle_question_input():
    """Handle question input and processing"""
    # Question input
    question = st.text_input(
        "Ask a question about your PDFs:",
        placeholder="What is the main topic of this document?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🤖 Ask", type="primary")
    
    with col2:
        if st.button("🔍 Show Relevant Chunks"):
            if question:
                show_relevant_chunks(question)
    
    # Process question
    if ask_button and question:
        process_question(question)

def show_relevant_chunks(question):
    """Show relevant document chunks"""
    with st.spinner("Finding relevant chunks..."):
        try:
            # Use default k from the RAG system
            relevant_docs = st.session_state.rag_system.get_relevant_docs(question)
            
            if relevant_docs:
                st.subheader("📄 Relevant Document Chunks:")
                for i, doc in enumerate(relevant_docs, 1):
                    with st.expander(f"Chunk {i} (Source: {doc.metadata.get('source', 'Unknown')})"):
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            else:
                st.warning("No relevant chunks found")
        except Exception as e:
            st.error(f"❌ Error retrieving chunks: {str(e)}")

def process_question(question):
    """Process a question and get answer"""
    with st.spinner("Thinking..."):
        try:
            answer = st.session_state.rag_system.query(question)
            
            if answer:
                # Add to chat history
                st.session_state.chat_history.append((question, answer))
                
                # Display answer
                st.subheader("🤖 Answer:")
                st.write(answer)
                
                # Show relevant chunks
                with st.expander("📄 Relevant Sources"):
                    relevant_docs = st.session_state.rag_system.get_relevant_docs(question)
                    for i, doc in enumerate(relevant_docs, 1):
                        st.write(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                        st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                        st.write("---")
            else:
                st.error("❌ Failed to get answer. Please try again.")
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")

def display_test_generation():
    """UI for generating tests using RAG"""
    st.subheader("📝 Generate Test from PDFs")

    col1, col2, col3 = st.columns(3)

    with col1:
        subject = st.text_input("Subject", value="Computer Vision")
        grade = st.text_input("Grade", value="Graduate")

    with col2:
        topic = st.text_input("Topic", value="Object Detection Metrics")
        question_count = st.number_input("Number of Questions", 1, 20, 5)

    with col3:
        question_type = st.selectbox(
            "Question Type",
            options=["mcq", "msq", "fill_blank"]
        )

    goals = st.text_area(
        "Learning Goals",
        value="Assess understanding of key concepts from the uploaded documents"
    )

    if st.button("🧠 Generate Test", type="primary"):
        inputs = {
            "subject": subject,
            "topic": topic,
            "grade": grade,
            "goals": goals,
            "question_count": question_count
        }

        generate_test(inputs, question_type)

def generate_test(inputs, question_type):
    """Generate test questions using RAG"""
    with st.spinner("Generating test questions..."):
        try:
            test = st.session_state.rag_system.generate_questions_rag(
                inputs=inputs,
                question_type=question_type
            )
            st.session_state.generated_test = test
            st.success("✅ Test generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating test: {str(e)}")

def display_generated_test():
    """Display generated test questions"""
    test = st.session_state.generated_test
    if not test:
        return

    st.subheader("📋 Generated Test")
    st.write(test.get("message", ""))

    for i, q in enumerate(test["questions"], 1):
        with st.expander(f"Question {i}"):
            st.write(f"**Type:** {q['question_type'].upper()}")
            st.write(f"**Question:** {q['question']}")

            for ans in q["answers"]:
                icon = "✅" if ans["is_correct"] else "❌"
                st.write(f"{icon} **{ans['text']}**")
                st.caption(ans["explanation"])


def main():
    """Main function"""
    st.title("📚 PDF RAG Chatbot")
    st.markdown(
        "Upload PDF files and interact with their content using "
        "**Retrieval-Augmented Generation (RAG)** — chat or generate tests."
    )

    # Initialize session state
    initialize_session_state()

    # Sidebar (PDF upload + settings)
    display_sidebar()

    # Main content area
    if not st.session_state.pdfs_processed:
        st.info("👆 Please upload and process PDF files using the sidebar to continue.")
        return

    # PDFs are ready
    st.success("✅ PDFs processed successfully!")

    # Tabs for different functionalities
    tab_chat, tab_test = st.tabs(["💬 Chat", "📝 Generate Test"])

    # ---------------- CHAT TAB ----------------
    with tab_chat:
        st.subheader("💬 Chat with your PDFs")

        # Display previous Q&A
        display_chat_history()

        # Question input + ask button
        handle_question_input()

    # ---------------- TEST TAB ----------------
    with tab_test:
        display_test_generation()
        display_generated_test()


if __name__ == "__main__":
    main()

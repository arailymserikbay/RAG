
import os
import tempfile
import json
import PyPDF2
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()


class PDFRAGSystem:
    def __init__(self, api_key: str = None):
        """Initialize the PDF RAG system"""
        self.api_key="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"  # replace with your API key

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required or pass api_key explicitly")

        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=self.api_key
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)

        # Initialize storage
        self.vectorstore = None
        self.retriever = None
        self.documents = []

    # ---------------- PDF Processing ---------------- #
    def extract_text_from_pdf(self, pdf_file_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file_path)
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"--- Page {i + 1} ---\n{page_text}\n\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_file_path}: {str(e)}")

    def load_pdf_documents(self, uploaded_files: list) -> list:
        """Load PDF files into Document objects"""
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                text = self.extract_text_from_pdf(tmp_path)
                if text:
                    documents.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
            finally:
                os.unlink(tmp_path)

        self.documents = documents
        return documents

    # ---------------- Document Splitting ---------------- #
    def split_documents(self, docs: list, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
        """Split documents into smaller chunks for embedding"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)

    # ---------------- Vector Store & Retriever ---------------- #
    def create_vectorstore(self, splits: list, collection_name: str = "pdf_rag_collection"):
        """Create vector store and retriever from document splits"""
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=collection_name
        )
        self.retriever = self.vectorstore.as_retriever(search_type="similarity")

    # ---------------- RAG Query ---------------- #
    def query(self, question: str, k: int = 3) -> str:
        """Query the RAG system directly"""
        if not self.retriever:
            raise ValueError("Retriever not initialized. Run create_vectorstore() first.")

        docs = self.retriever.get_relevant_documents(question)[:k]
        context_text = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are a helpful assistant that answers questions based on the provided PDF context.

Context:
{context_text}

Question: {question}

Instructions:
- Answer only based on the context.
- If answer is not in the context, say "I don't have enough information to answer this question."
- Be concise and indicate the source page if relevant.
Answer:
"""
        return self.llm.invoke(prompt)

    # ---------------- Structured Test Question Generation ---------------- #
    def generate_questions_rag(self, inputs: dict, question_type: str = "mcq", k: int = 5) -> dict:
        """
        Generate structured questions (MCQ, MSQ, Fill-in-the-Blank) from PDFs.
        inputs: dict with keys: subject, topic, grade, goals, question_count
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Run create_vectorstore() first.")

        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(inputs.get("topic", ""))[:k]
        context_text = "\n\n".join(doc.page_content for doc in docs)

        schema = """
{
  "name": "questions",
  "strict": true,
  "schema": {
    "name": "questions",
    "type": "object",
    "properties": {
      "message": {"type": "string", "minLength": 1},
      "questions": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "question_type": {"type": "string", "enum": ["mcq", "msq", "fill_blank"]},
            "question": {"type": "string"},
            "answers": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "text": {"type": "string"},
                  "explanation": {"type": "string"},
                  "is_correct": {"type": "boolean"}
                },
                "required": ["text","explanation","is_correct"],
                "additionalProperties": false
              }
            }
          },
          "required": ["question_type","question","answers"],
          "additionalProperties": false
        }
      }
    },
    "required": ["message","questions"],
    "additionalProperties": false
  }
}
"""
        prompt_template = f"""
You are an expert educator generating test questions STRICTLY based on the provided context.

Context (from PDFs):
{context_text}

Task:
Generate a {inputs['question_count']}-question {question_type} test.

Inputs:
- Subject: {inputs['subject']}
- Topic: {inputs['topic']}
- Grade: {inputs['grade']}
- Goals: {inputs['goals']}

Rules:
1. Use ONLY the information from the context.
2. Do NOT use external knowledge.
3. Each question must be of type "{question_type}".
4. For MCQ/MSQ: provide exactly 4 answer options.
5. For Fill-in-the-Blank: provide the blank and correct answer.
6. Provide explanations for all answers.
7. Output STRICTLY valid JSON following this schema:
{schema}
"""

        raw_output = self.llm.invoke(prompt_template)

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON. Model output:\n" + raw_output)

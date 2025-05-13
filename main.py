# main.py
import os
import shutil
import time
import gc
import stat
import json

from chromadb.config import Settings as ChromaSettings

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings  # You can swap this if needed
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

VECTOR_DB_PATH = os.path.join("vectorstores", "default")
UPLOADED_FILES_LOG = os.path.join(VECTOR_DB_PATH, "files.txt")


class ChatPDF:
    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Sei un assistente multilingue per compiti di domanda-risposta. Utilizza solo il contesto fornito per rispondere alla domanda nella stessa lingua in cui è stata posta. Se non conosci la risposta, dì chiaramente che non è disponibile nel contesto. [/INST] </s>
            [INST] Domanda: {question}
            Contesto: {context}
            Risposta: [/INST]
            """
        )
        self.db_path = VECTOR_DB_PATH
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self._load_vector_store()

    def _chroma_settings(self):
        return ChromaSettings(
            persist_directory=self.db_path,
            allow_reset=True,
            anonymized_telemetry=False
        )

    def _load_vector_store(self):
        if os.path.exists(os.path.join(self.db_path, "chroma.sqlite3")):
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=FastEmbedEmbeddings(),
                client_settings=self._chroma_settings()
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.3},
            )
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )

    def ingest(self, file_path: str):
        os.makedirs(self.db_path, exist_ok=True)

        ext = os.path.splitext(file_path)[-1].lower()
        docs = []

        if ext == ".pdf":
            docs = PyPDFLoader(file_path=file_path).load()

        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "defectList" not in data:
                raise ValueError("This JSON file is not in the expected defect format.")

            for defect in data["defectList"]:
                content = "\n".join([
                    f"Catalog: {defect.get('catalog', '')}",
                    f"Type: {defect.get('type', '')}",
                    f"Measure: {defect.get('measure', '')}",
                    f"ID: {defect.get('id', '')}",
                    f"Custom Info: {json.dumps(defect.get('customData', {}))}",
                    f"Polyline: {defect.get('poly', '')[:100]}..."
                ])
                docs.append(Document(page_content=content))

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if not docs:
            raise ValueError("No valid documents found in the file.")

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        file_name = os.path.basename(file_path)
        for chunk in chunks:
            chunk.metadata['source_file'] = file_name

        if not self.vector_store:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=FastEmbedEmbeddings(),
                persist_directory=self.db_path,
                client_settings=self._chroma_settings()
            )
        else:
            self.vector_store.add_documents(chunks)

        self._log_uploaded_file(file_name)
        self._load_vector_store()

    def _log_uploaded_file(self, file_name: str):
        if not os.path.exists(UPLOADED_FILES_LOG):
            with open(UPLOADED_FILES_LOG, 'w') as f:
                f.write(file_name + '\n')
        else:
            with open(UPLOADED_FILES_LOG, 'r+') as f:
                files = f.read().splitlines()
                if file_name not in files:
                    f.write(file_name + '\n')

    def list_uploaded_files(self):
        if not os.path.exists(UPLOADED_FILES_LOG):
            return []
        with open(UPLOADED_FILES_LOG, 'r') as f:
            return f.read().splitlines()

    def delete_all_data(self):
        self.clear()
        time.sleep(0.5)
        gc.collect()
        if os.path.exists(self.db_path):
            try:
                shutil.rmtree(self.db_path, onerror=self._handle_remove_readonly)
            except Exception as e:
                print(f"Error deleting DB folder: {e}")

    def _handle_remove_readonly(self, func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF or JSON document first."
        return self.chain.invoke(query)

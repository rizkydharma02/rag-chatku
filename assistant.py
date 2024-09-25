from typing import Optional
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
from supabase import create_client, Client
from sqlalchemy import create_engine

# URL Supabase Postgres yang sudah diperbaiki
db_url = "postgresql://ngsuidrvwvmomhlkouvd:Rayaku020602@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

# URL Supabase dan kunci API
supabase_url = "https://ngsuidrvwvmomhlkouvd.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nc3VpZHJ2d3Ztb21obGtvdXZkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjcyNTU5MDgsImV4cCI6MjA0MjgzMTkwOH0.G-hu_TUz5Iskd4c35JGTvLgJ42gi1QOY8AcHudLcT_o"

# Membuat client Supabase
def create_supabase_client() -> Client:
    return create_client(supabase_url, supabase_key)

# Membuat engine PostgreSQL menggunakan SQLAlchemy
def create_pg_engine(db_url: str):
    return create_engine(db_url)

# Membuat instance Assistant
def get_groq_assistant(
    llm_model: str = "llama3-70b-8192",
    embeddings_model: str = "text-embedding-3-small",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq RAG Assistant."""

    # Tentukan embedder berdasarkan model embeddings
    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    
    # Tentukan tabel embeddings berdasarkan model embeddings
    embeddings_table = (
        "groq_rag_documents_ollama" if embeddings_model == "nomic-embed-text" else "groq_rag_documents_openai"
    )
    
    # Buat Supabase client
    supabase_client = create_supabase_client()

    # Buat PostgreSQL engine untuk koneksi ke database
    pg_engine = create_pg_engine(db_url)

    return Assistant(
        name="groq_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="groq_rag_assistant", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=embedder,
            ),
            # 2 referensi ditambahkan ke prompt
            num_documents=2,
        ),
        description="You are an AI called 'GroqRAG' and your task is to answer questions using the provided information",
        instructions=[
            "When a user asks a question, you will be provided with information about the question.",
            "Carefully read this information and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        # Pengaturan ini menambahkan referensi dari knowledge_base ke prompt pengguna
        add_references_to_prompt=True,
        # Pengaturan ini membuat LLM memformat pesan dalam markdown
        markdown=True,
        # Pengaturan ini menambahkan histori chat ke pesan
        add_chat_history_to_messages=True,
        # Pengaturan ini menambahkan 4 pesan sebelumnya dari histori chat ke pesan
        num_history_messages=4,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )

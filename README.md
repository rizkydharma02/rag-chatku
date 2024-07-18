# Petunjuk Penggunaan Chatku AI

### 1. Clone Github Repository

```shell
conda create -n phidata python=3.11 -y
conda activate phidata
pip install -r requirements.txt
```

### 2. Export Groq API Key

```shell
export GROQ_API_KEY=***
```

### 3. Pakai Ollama untuk embeddings

```shell
ollama pull llama3 //optional
ollama pull nomic-embed-text
ollama run nomic-embed-text
```

### 6. Buat dan Jalankan Database Docker

```shell
docker ps
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16
```

### 6. Jalankan streamlit App

```shell
streamlit run app.py
```

- open [localhost:8501](http://localhost:8501)

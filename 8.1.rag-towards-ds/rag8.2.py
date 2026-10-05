import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama #, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# * загрузка -> чанкинг -> эмбеддинги -> локальная БД -> поиск -> генерация
# * Впереди продвинутый поиск (Hybrid Search, Re-ranking), оценка качества (RAGAS) и кэширование.

llm = ChatOllama(model="llama3", temperature=0.3)

# embeddings = OllamaEmbeddings(model="qwen2.5:14b") #"llama3") # текущая версия Ollama для Endpoint /api/embed в некоторых конфигурациях (особенно при запуске через автозапуск или системные службы) не активирует подмодуль эмбеддингов для тяжелых моделей вроде qwen2.5:14b.

# Локальные эмбеддинги через HuggingFace (работают на процессоре/GPU без Ollama)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# * Парсинг папки RAG files: скрипт находит все .txt файлы, считывает их содержимое через TextLoader и складывает в единый массив документов.
text_folder="RAG files"
documents=[]
for filename in os.listdir(text_folder):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(text_folder, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# Splitting to chunks - chunking
# * Чанкинг (разбиение документов): длинные тексты режутся на куски по 1000 символов с нахлестом (overlap) в 100 символов, чтобы контекст на стыках не терялся.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# Creating vector database FAISS
# * Векторизация текстов с помощью embeddings, сохранение их в оперативной векторной базе данных FAISS и создание поискового инструмента (retriever).
vector_store = FAISS.from_documents(split_docs, embeddings)
retriever = vector_store.as_retriever() # * поисковик по векторной базе ищет куски текста (чанки) и берет, например, топ-3 или топ-5 самых близких

def main():
    print("Welcome to the Local RAG Assistant Ollama. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if (user_input.lower() == "exit"):
            print("Exiting...")
            break

        # * Поиск в базе данных FAISS: по вопросу пользователя (user_input) система находит наиболее релевантные куски текста и объединяет их в единую строку контекста.
        relevant_docs = retriever.invoke(user_input)
        retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])

        system_prompt = (
            "You are a helpfull assistant. "
            "Use ONLY the following knowledge base context to answer the user. "
            f"Context:\n{retrieved_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": user_input}
        ]

        response = llm.invoke(messages)
        print(f"\nAssistant: {response.content.strip()}\n")

if __name__ == "__main__":
    main()

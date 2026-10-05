import chromadb
# 1\. Инициализируем локальную базу данных (сохраняется в папку на диске)
client = chromadb.PersistentClient(path="./my_chroma_db")
# # 2\. Создаем или получаем коллекцию (аналог таблицы в SQL)
collection = client.get_or_create_collection(name="ai_knowledge_base")
# 3\. Добавляем текстовые документы
# ChromaDB автоматически сгенерирует для них векторные эмбеддинги
collection.add(
    documents=[
        "Vector databases store embeddings and enable semantic search based on meaning.",
        "Model training requires splitting data into training and evaluation sets to prevent overfitting.",
        "AI Agents use tools like terminal and browser to perform autonomous decisions and actions."
    ],
    metadatas=[{"category": "vectordb"}, {"category": "training"}, {"category": "agents"}],
    ids=["doc1", "doc2", "doc3"]
)
# 4\. Выполняем семантический поиск по смыслу (без совпадения ключевых слов)
query = "Как искать информацию не по точным словам, а по смыслу?"
results = collection.query( query_texts=[query], n_results=1 )

# Посмотрим, во что превратился текст запроса
embedded_query = collection._embedding_function([query])
print("Размерность вектора:", len(embedded_query[0]))
print("Первые 5 чисел вектора:", embedded_query[0][:5])

print("--- Результат поиска ---")
print(f"Найденный документ: {results['documents']}")
print(f"ID документа: {results['ids']}")

# **Что произошло под капотом:**

# 1. Текст `query` автоматически превратился в вектор.
# 2. ChromaDB сравнила косинусное расстояние между вектором запроса и векторами из коллекции[1].
# 3. База вернула документ про `Vector databases`, хотя в исходном запросе слова «Vector database» вообще не фигурировали[1].

# roadmap-sh % python3 5.3.vector_demo_chromaDB.py
# Размерность вектора: 384
# Первые 5 чисел вектора: [ 0.06086183  0.01496605 -0.07812544  0.03281106 -0.00627938]
# --- Результат поиска ---
# Найденный документ: [['Vector databases store embeddings and enable semantic search based on meaning.']]
# ID документа: [['doc1']]

import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

API_KEY = os.getenv('PINECONE_API_KEY')
ENV = os.getenv('PINECONE_ENVIRONMENT')
INDEX_NAME = os.getenv('PINECONE_INDEX')

# initializing Pinecone client with API key
pc = Pinecone(api_key=API_KEY)

# checking if the pinecone index already exists
# if not, creating it with the specified dimension (384 for all-MiniLM-L6-v2 embeddings),
# similarity metric (cosine), and serverless spec (cloud provider and region)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=ENV)
    )

# connecting to the existing or newly created index for querying and upserting vectors
index = pc.Index(INDEX_NAME)

# loading embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_books(file_path='books.json'):
    """
    loading the books data from a JSON file.
    each book should contain fields like id, title, author, genre, and year.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def upsert_books_to_pinecone(books):
    """
    for each book, we create a text description combining key metadata.
    encode the text into an embedding vector.
    prepare the vectors with metadata and upload them to the Pinecone index.
    this enables semantic search over our book data
    """
    vectors = []
    for book in books:
        text = f'{book['title']} by {book['author']}, Genre: {book['genre']}, Year: {book['year']}'
        embedding = model.encode(text).tolist() # converting embedding to list for JSON compatibility
        vectors.append({
            'id': str(book['id']),
            'values': embedding,
            'metadata': book
        })
    print(f'Upserting {len(vectors)} books')
    index.upsert(vectors=vectors)
    print('done uploading')


def chatbot_query(query_text, top_k=3):
    query_vector = model.encode(query_text).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    if results and 'matches' in results and results['matches']:
        response = 'Here are some books I found:\n'
        for match in results['matches']:
            meta = match['metadata']
            response += f"- \"{meta['title']}\" by {meta['author']} ({meta['year']}), Genre: {meta['genre']}\n"
    else:
        response = 'Sorry, no books matched your query.'
    return response

def main():
    books = load_books()
    # uncomment to upload books only once, then comment again
    # upsert_books_to_pinecone(books)

    print('📚 Welcome to the Book Chatbot! Ask about books. Type "exit" to quit.')
    while True:
        user_input = input('You: ').strip()
        if user_input.lower() == 'exit':
            print('👋 Goodbye!')
            break
        answer = chatbot_query(user_input)
        print('Bot:', answer)

if __name__ == '__main__':
    main()







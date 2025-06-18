import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

API_KEY = os.getenv('PINECONE_API_KEY')
ENV = os.getenv('PINECONE_ENVIRONMENT')

INDEXES = {
    'title': 'title-index',
    'author': 'author-index',
    'genre': 'genre-index',
}

# initializing Pinecone client with API key
pc = Pinecone(api_key=API_KEY)

for name in INDEXES.values():
    if name not in pc.list_indexes():
        pc.create_index(
            name=name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=ENV)
        )

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
    for key, index_name in INDEXES.items():
        print(f'upserting to index: {index_name}')
        idx = pc.Index(name=index_name)
        vectors = []
        for book in books:
            text = book[key]
            embedding = model.encode(text).tolist()
            vectors.append({
                'id': f"{key}-{book['id']}",
                'values': embedding,
                'metadata': book
            })
        idx.upsert(vectors=vectors)
    print('all books upserted to all indexes.')


def chatbot_query(query_text, top_k=3, score_threshold=0.5):
    query_vector = model.encode(query_text).tolist()
    all_matches = []

    for key, index_name in INDEXES.items():
        idx = pc.Index(index_name)
        results = idx.query(vector=query_vector, top_k=top_k, include_metadata=True)
        matches = results.get('matches', [])
        all_matches.extend([m for m in matches if m['score'] >= score_threshold])

    # sort combined matches by score, highest first
    all_matches.sort(key=lambda x: x['score'], reverse=True)

    if all_matches:
        response = '📚 Here are some books I found:\n'
        seen_ids = set()
        for match in all_matches:
            meta = match['metadata']
            if meta['id'] not in seen_ids:
                response += f"- \"{meta['title']}\" by {meta['author']} ({meta['year']}), Genre: {meta['genre']}\n"
                seen_ids.add(meta['id'])
    else:
        response = 'Sorry, no good matches found.'

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

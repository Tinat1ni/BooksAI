# Book Chatbot

A simple chatbot that lets you search a collection of books using natural language queries. It uses sentence embeddings from the SentenceTransformers library and Pinecone vector database to find relevant books based on your questions.

---

## Features

- Embeds book metadata (title, author, genre, year) using `all-MiniLM-L6-v2` SentenceTransformer model.
- Stores book embeddings in a Pinecone vector index for fast similarity search.
- Answers user queries by retrieving the most relevant books.
- Command line interface for easy interaction.
- Exit anytime by typing `exit`.

---

## How It Works

1. **Load book data** from a JSON file (`books.json`), containing book details.
2. **Generate vector embeddings** for each book's metadata using the SentenceTransformer model.
3. **Upsert embeddings into Pinecone** vector database (one-time operation).
4. **Accept user queries** via command line input.
5. **Embed the query** and search Pinecone for similar book embeddings.
6. **Return and display top matching books** to the user.

---

## Example Conversation

📚 Welcome to the Book Chatbot! Ask about books. Type "exit" to quit.
You: do you have philosophy books? 
Bot: Here are some books I found:
- "Leviathan" by Thomas Hobbes (1651.0), Genre: Political Philosophy
- "Beyond Good and Evil" by Friedrich Nietzsche (1886.0), Genre: Philosophy
- "The Republic" by Plato (-380.0), Genre: Philosophy

You: something from Friedrich Nietzsche?
Bot: Here are some books I found:
- "On the Genealogy of Morals" by Friedrich Nietzsche (1887.0), Genre: Philosophy
- "The Birth of Tragedy" by Friedrich Nietzsche (1872.0), Genre: Philosophy
- "Beyond Good and Evil" by Friedrich Nietzsche (1886.0), Genre: Philosophy

You: exit
👋 Goodbye!

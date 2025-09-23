# AI-Powered Semantic Quote Finder

This project is a Python-based semantic search engine that finds quotes based on their meaning, not just keywords. It uses a SentenceTransformer model to convert a large database of quotes into vector embeddings and leverages FAISS (Facebook AI Similarity Search) for high-speed similarity search!

## Features

-   **Data Processing**: A script (`make_index.py`) to process and embed nearly 500,000 quotes into a vector database.
-   **Semantic Search**: A script (`find_quote.py`) that takes a user's paraphrased quote and returns the three most conceptually similar quotes from the database.
-   **Efficient Indexing**: Uses FAISS for fast and memory-efficient vector similarity searches.

## Technologies Used

-   **Python 3**
-   **SentenceTransformers**: For generating high-quality text embeddings from a HuggingFace model (`avsolatorio/GIST-large-Embedding-v0`).
-   **FAISS (Facebook AI Similarity Search)**: For building and searching the vector index.
-   **NumPy**: For numerical operations on the vector embeddings.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_NAME]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Data:**
    This project requires the `quotes.csv` file, which is not included in this repository. Please place it in the root directory of the project.

## How to Use

1.  **Create the Index:**
    First, run the `make_index.py` script to process the `quotes.csv` file, generate the embeddings, and save the FAISS index.
    ```bash
    python3 make_index.py
    ```
    This process will take a significant amount of time depending on your hardware.

2.  **Find a Quote:**
    Once the index is built, you can run the interactive search script.
    ```bash
    python3 find_quote.py
    ```
    The program will then prompt you to enter a paraphrased quote to search for[cite: 16].

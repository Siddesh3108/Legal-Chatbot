# Legal Chatbot

A Flask-based legal question-answering chatbot that provides information based on a predefined legal database and user-uploaded PDF documents. It leverages advanced Natural Language Processing (NLP) models for semantic search, question answering, and text summarization to deliver relevant legal insights.

## Features

  * **Intelligent Q\&A:** Answers legal questions by searching a built-in knowledge base.
  * **Custom Document Integration:** Allows users to upload their own PDF legal documents (e.g., contracts, case files) and query them.
  * **Semantic Search:** Utilizes Sentence Transformers for intelligent, context-aware document retrieval.
  * **Precise Answer Extraction:** Employs Hugging Face's Question Answering models (e.g., `deepset/roberta-base-squad2`) to pinpoint exact answers within relevant text.
  * **Text Summarization:** Summarizes lengthy content using BART for concise information delivery.
  * **Dynamic Legal Database:** Includes pre-populated information on common legal topics in India (e.g., leave, bail, evidence, property, divorce, criminal law, traffic law, family law, employment law), referencing relevant Indian acts.
  * **PDF Text Extraction:** Extracts text from uploaded PDF files for analysis.
  * **Web Interface:** Simple Flask web interface for interaction (upload and chat).

## Technologies Used

  * **Backend:** Python 3.9+
  * **Web Framework:** Flask
  * **NLP:**
      * Hugging Face Transformers (`pipeline`, `AutoModelForQuestionAnswering`, `AutoTokenizer`)
      * Sentence Transformers (`SentenceTransformer`, `util`)
      * PyMuPDF (`fitz`) for PDF text extraction
  * **Utility Libraries:** `os`, `logging`, `numpy`, `torch`, `uuid`, `re`, `werkzeug.utils`, `sklearn`

## Setup and Installation

### Prerequisites

  * Python 3.9 or higher
  * `pip` (Python package installer)

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Siddesh3108/Legal-Chatbot-.git
    cd Legal-Chatbot-
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

      * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
      * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt`, you can create one (see next section) or install them manually:

    ```bash
    pip install flask transformers sentence-transformers pymupdf numpy scikit-learn
    ```

5.  **Place your PDF files:**
    Ensure the pre-specified PDF files (`sakshya.pdf`, `Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf`, `bharatiya-sakshya-bill-511326.pdf`, `Laws.pdf`) are in the same directory as your `app.py` (or `main.py` if you rename it). The application is configured to read these automatically.
    You also need to create `static` and `templates` directories for the web interface:

    ```bash
    mkdir static
    mkdir templates
    mkdir static/css
    ```

    *You'll need to add your `index.html` to the `templates` folder and any CSS files to `static/css` for the web interface to work.*

6.  **Run the Flask application:**

    ```bash
    python app.py
    ```

    The application will start on `http://0.0.0.0:5000/`. You can access it via `http://localhost:5000` in your web browser.

## Creating `requirements.txt`

If you don't have a `requirements.txt` file, create one in your project's root directory and add the following:

```
flask
transformers
sentence-transformers
pymupdf
numpy
torch # Transformers and Sentence Transformers rely on PyTorch
scikit-learn
werkzeug
```

Then run `pip install -r requirements.txt`.

## Usage

### Web Interface

Once the server is running, open your web browser and go to `http://localhost:5000`.

  * **Upload PDFs:** Use the provided interface to upload your legal PDF documents. These documents will be processed and become searchable by the chatbot.
  * **Chat:** Type your legal questions into the chat interface. The bot will respond with information from its internal database or your uploaded PDFs.

### API Endpoints

The chatbot also exposes REST API endpoints for programmatic interaction.

  * **`GET /`**: Renders the main HTML page (`index.html`).
  * **`GET /static/<path:path>`**: Serves static files (CSS, JS, images).
  * **`POST /upload-pdf`**:
      * **Method:** `POST`
      * **Headers:** `Content-Type: multipart/form-data`
      * **Form Fields:**
          * `pdfFile`: The PDF file to upload.
          * `documentType` (optional): A string categorizing the document (e.g., "contract", "case\_law").
          * `documentDescription` (optional): A brief description of the document.
      * **Response:** JSON indicating success/failure and `id` of the uploaded PDF.
  * **`GET /pdf-list`**:
      * **Method:** `GET`
      * **Response:** JSON list of uploaded PDF metadata (id, filename, type, description, processed status).
  * **`DELETE /delete-pdf/<pdf_id>`**:
      * **Method:** `DELETE`
      * **Path Parameter:** `pdf_id` (the ID of the PDF to delete, returned during upload).
      * **Response:** JSON indicating success/failure.
  * **`POST /chat`**:
      * **Method:** `POST`
      * **Headers:** `Content-Type: application/json`
      * **Body:**
        ```json
        {
            "question": "Your legal question here?"
        }
        ```
      * **Response:** JSON containing the `answer`, `legal_reference` (if any), and `confidence` score.

## Code Structure

  * `app.py`: The main Flask application file.
  * `uploads/`: Directory for storing uploaded PDF files.
      * `uploads/pdfs/`: Subdirectory specifically for PDF files.
  * `templates/`: Contains HTML template files (e.g., `index.html`).
  * `static/`: Contains static assets like CSS and JavaScript files.
      * `static/css/`: For CSS files.

## Important Notes

  * **Model Loading:** The application downloads pre-trained NLP models on the first run. This might take some time and requires an internet connection. Subsequent runs will use cached models.
  * **Resource Usage:** NLP models, especially larger ones, can be memory and CPU intensive. Consider deploying on a machine with sufficient resources.
  * **Legal Advice Disclaimer:** This chatbot is for informational purposes only and should **not** be considered legal advice. Always consult with a qualified legal professional for specific legal guidance.
  * **PDF Extraction Quality:** Text extraction from PDFs can vary based on the PDF's structure and quality. Scanned PDFs might result in poor extraction.
  * **Context Window:** Large documents might be truncated for processing by some NLP models due to context window limitations. The current implementation truncates to 5000 characters for the initial `legal_database` entry and processes PDFs in paragraphs for `custom_pdf_data`.

## Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request.


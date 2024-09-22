# Medical Chatbot Project Documentation

## 1. Introduction

### 1.1 Project Overview
This project aims to develop an AI-powered medical chatbot capable of providing symptom-based advice, recommending doctors, and retrieving medical documents. It uses advanced natural language processing models, memory mechanisms, and document retrieval tools to ensure personalized and informative responses.

### 1.2 Key Features
- **Symptom Checking**: Users can input their symptoms, and the chatbot responds with potential causes and recommended actions.
- **Doctor Recommendation**: Based on user symptoms, the chatbot recommends which specialized doctors they should go to.
- **Document Retrieval**: The chatbot can retrieve relevant information from a knowledge base of medical PDFs.
- **Memory-based Conversations**: The chatbot retains conversation history to enhance the user experience.
- **Drug Interaction Alert System**: (Planned feature) Provides warnings on potential drug interactions.

---

## 2. Project Setup
### 2.1 Prerequisites
- **Python 3.10**
  
- **Required Libraries**:
  - [Hugging Face Transformers](https://huggingface.co/transformers/)
  - [LangChain](https://python.langchain.com/)
  - [Pinecone](https://www.pinecone.io/) (for vector storage)
  - [Sentence-Transformers](https://www.sbert.net/) (for document embedding)
  
- **Required Model**:
  - [BioMistral-7B](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/) (for generating medical advice)

- **Other Tools**:
  - `wget` (for downloading models)

### 2.2 Project Structure

The project is structured as follows:
``` css
root/ 
│ ├── app.py 
│ ├── requirements.txt 
│ └── README.md 
└── src/ 
     ├── __init__.py 
     ├── conversation.py 
     ├── data_processing.py 
     ├── models.py
     └── pinecone_setup.py
     └── vectorstore.py
└── data/ 
     ├── Cognitive Behavior Therapy.pdf
```

### 2.3 File Descriptions

- **app.py**: Main entry point to run the application.
- **requirements.txt**: Contains the list of dependencies required to run the chatbot.
- **README.md**: Documentation file providing an overview of the project.

### src/
  - **`__init__.py`**: Initializes the `src` package.
  - **conversation.py**: Handles the chatbot's logic and manages conversation chains.
  - **data_processing.py**: Loads, processes, and chunks the PDF documents into searchable text.
  - **models.py**: Manages loading and switching between different language models, including BioMistral-7B.
  - **pinecone_setup.py**: Sets up and configures the Pinecone vector database for document search.
  - **vectorstore.py**: Responsible for embedding the document data into vectors and storing them in Pinecone for fast retrieval.

### data/
  - **Cognitive Behavior Therapy.pdf**: Example medical document used for embedding and document retrieval.


### 2.4 Pinecone API Key

To use Pinecone for vector storage, you need to set your Pinecone API key as an environment variable. Add your API key to the environment before running the application:

```python
import os

api_key = os.getenv('PINECONE_API_KEY')
```

To set the environment variable, use the following command (replace your-api-key with your actual Pinecone API key):

- Linux/macOS:
  ```sh
  export PINECONE_API_KEY=your-api-key
  ```

- Windows
  ```sh
  set PINECONE_API_KEY=your-api-key
  ```

## 3. Setup Instructions

To run the project locally, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/https://github.com/TabibSystem/AI-Repo.git
    cd AI-Repo
    ```

2. **Install the Required Packages**: Make sure you have Python installed, then run:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Application**: Execute the following command to start the Gradio interface:
    ```sh
    python app.py
    ```

## 4. Technologies Used
This section highlights the key technologies utilized in the development of the medical chatbot.

- **Transformers Library**: Used for natural language processing tasks and to load the BioMistral-7B model for generating medical advice.
- **LangChain**: Provides the structure for creating conversation chains, managing memory, and interfacing with the large language model.
- **Sentence Transformers**: Used to create sentence embeddings from text chunks for document retrieval.
- **Pinecone**: A vector database for storing and retrieving document embeddings. This allows the chatbot to quickly search for relevant text based on user input.
- **PyPDFLoader**: A document loader used to extract content from PDF files and prepare them for processing.
- **TQDM**: Used to display progress bars for various processes, such as populating the Pinecone index.
- **UUID**: Generates unique identifiers for each text chunk when upserting to Pinecone.

---

## 5. Data

### 5.1 Data Sources
- **PDF Documents**: Medical PDFs were used as an initial data source. The chatbot retrieves information from these PDFs when prompted by a user query.
- **Drug Interaction Database** (planned): DrugBank or OpenFDA databases could be integrated to enhance the drug recommendation/interaction functionality.

### 5.2 Data Preprocessing
To use the information from PDF documents in the chatbot, the following steps were performed:
1. **Loading PDF Files**: Using the PyPDFLoader, PDFs are loaded and parsed.
2. **Splitting Text**: The documents are split into smaller text chunks using RecursiveCharacterTextSplitter. 
    - **Chunk Size**: 700 characters
    - **Overlap**: 70 characters
3. **Embedding**: Each text chunk is encoded into vector form using `SentenceTransformer("NeuML/pubmedbert-base-embeddings")`.

---

## 6. Model Architecture

### 6.1 Language Model (LLM)
The core of the chatbot is powered by a transformer model, specifically **BioMistral-7B** loaded via `CTransformers`. This LLM was selected for its ability to understand medical texts and generate responses based on user inputs.

- **Model Path**: `./BioMistral-7B.Q8_0.gguf`
- **Model Type**: LLaMA-based model with special tuning for medical conversations.
- **Configuration**:
  - `max_new_tokens`: 1024 (Controls the length of generated responses)
  - `temperature`: 0.1 (Determines randomness in response generation)
- **Callbacks**: A streaming callback (`StreamingStdOutCallbackHandler`) is used to handle real-time response generation.

### 6.2 Embedding Model
For encoding the text, the **SentenceTransformer** model `NeuML/pubmedbert-base-embeddings` was used. This model is fine-tuned on medical data, making it well-suited for the project’s needs.

---

## 7. Experiments and Results

### 7.1 Pinecone Integration for Vector Storage
The project uses **Pinecone** for storing and retrieving embeddings. Pinecone allows fast search and retrieval based on semantic similarity, which is essential for responding to user queries by referencing pre-processed medical documents.

#### 7.1.1 Pinecone Setup
- **Index Creation**: An index named `med-base` is created with 768 dimensions and the dotproduct metric for similarity scoring.
  - **Cloud**: AWS
  - **Region**: US-East-1

#### 7.1.2 Document Embedding and Upsert
- Text chunks from the loaded PDF are encoded and inserted into Pinecone in batches (up to 100 at a time). Each chunk is associated with metadata that includes the chunk number and the source file it came from.
  - **Batch Limit**: 100 chunks at a time
  - **Metadata**: Each chunk has metadata like the source PDF and chunk number.

### 7.2 Memory for Long-term Interactions
To enable more natural and coherent conversations, the chatbot uses a **ConversationSummaryBufferMemory**. This memory keeps track of the conversation history and summarizes it over time. The memory is session-based, ensuring personalized interactions for each user session.

- **Session ID**: A unique ID is assigned to each conversation to track its history.
- **Max Token Limit**: 1024 tokens (memory is limited to avoid overwhelming the model with too much information).

---

## 8. User Guide

### 8.1 How to Use the Chatbot
Users can interact with the chatbot by asking symptom-related questions, requesting doctor recommendations, or querying medical PDFs.

- **Symptom Checking**:
  - Input example: "I have a headache and nausea."
  - Response: The chatbot provides potential causes and medical advice.

- **Document Retrieval**:
  - Input example: "What are the treatments for anxiety in CBT?"
  - Response: The chatbot retrieves relevant text chunks from the **Cognitive Behavioral Therapy.pdf** document.

### 8.2 Customizing the Chatbot
Developers can modify the chatbot to suit their needs:
- **Prompt Tuning**: Adjust the `PromptTemplate` to change the style of responses.
- **Memory Size**: Modify the token limit for conversation history in `ConversationSummaryBufferMemory`.
- **Adding New Documents**: New PDF files can be added to the system for retrieval. Follow the same steps for loading and splitting PDFs.

---

## 9. Conclusion
This project has demonstrated the successful integration of AI-powered language models, document retrieval, and long-term memory for a medical chatbot. The current system allows for:
- Real-time symptom checking.
- Retrieval of medical knowledge from PDF files.
- Personalized interactions using memory-based conversations.

### Future Directions
- **Drug Interaction System**: Implement a feature that warns users about potential drug interactions using databases like DrugBank.
- **Expanding the Knowledge Base**: Add more medical documents or integrate external APIs to broaden the chatbot’s knowledge.

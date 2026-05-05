import os
import json
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#LOAD THE DOCS
def load_documents(directory_path):
    """Loads PDFs and Word documents from a specified directory."""
    documents = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Handle PDFs
        if filename.endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
        # Handle Word Documents
        elif filename.endswith(".docx"):
            print(f"Loading Word Doc: {filename}")
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
            
        else:
            print(f"Skipping unsupported file format: {filename}")
            
    return documents

# CHUCK THE DOCUMENTS
def chunk_documents(raw_documents):
    """Splits large documents into smaller chunks for the Vector DB."""
    # RecursiveCharacterTextSplitter is the standard for natural language
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Number of characters per chunk
        chunk_overlap=200,     # Overlap to maintain context between chunks
        length_function=len,
        is_separator_regex=False,
    )
    
    chunked_docs = text_splitter.split_documents(raw_documents)
    return chunked_docs
# CHUNCKS SAVES INTO JSON FILE IN ./output directory
def save_chunks_to_json(chunks, output_dir="./output", filename="processed_chunks.json"):
    """Serializes LangChain Document chunks and saves them locally."""
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    formatted_data = []
    
    # Extract the data from the LangChain objects
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "id": f"chunk_{i}", # Give each chunk a temporary ID
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }
        formatted_data.append(chunk_data)
        
    output_path = os.path.join(output_dir, filename)
    
    # Write the data to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        # indent=4 makes the JSON readable for humans
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
        
    print(f"\nSaved {len(chunks)} chunks to {output_path}")
    return output_path

# --- Execution ---
if __name__ == "__main__":
    DATA_DIR = "./data"  # Path to your documents folder
    OUTPUT_DIR = "./output"  # Where the JSON will be saved
    # Step 1: Load the raw text
    raw_docs = load_documents(DATA_DIR)
    print(f"\nSuccessfully loaded {len(raw_docs)} document pages/sections.")
    
    # Step 2: Split into manageable chunks
    chunks = chunk_documents(raw_docs)
    print(f"Split data into {len(chunks)} chunks.")
    
    # Step 3: Save to local directory
    if chunks:
        save_chunks_to_json(chunks, OUTPUT_DIR)

        
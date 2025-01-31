import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch



st.set_page_config(page_title="Deep dive AI")

st.title("Welcome to Deep Dive AI 🚀")
st.subheader("In-depth insights from AI Research papers")


#Load the Hugging Face model gpt2
def load_huggingface_model(prompt):
    modelname = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    generator=pipeline('text-generation', model=model, tokenizer=tokenizer)
    answer=generator(prompt, max_new_tokens=150, num_return_sequences=1)
    return answer



def extract_text_from_pdf(uploaded_file):
    
   # Read the PDF file
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    st.write(f"Number of pages: {num_pages}")
    text=""

    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def preprocess_text(text):
    # Split the text into paragraphs based on double newlines (i.e., paragraphs are separated by two line breaks)
    paragraphs = text.split('\n\n')
    
    # Clean up each paragraph by stripping extra spaces or other preprocessing tasks
    paragraphs = [para.strip() for para in paragraphs if para.strip() != ""]
    
    return paragraphs


#generate embeddings for each paragraph
def generate_embeddings(paragraphs):
    model= SentenceTransformer('all-MiniLM-L6-v2') #load the model
   
    embeddings = model.encode(paragraphs, convert_to_tensor=True) #generate embeddings
    # Ensure embeddings are on CPU and convert to numpy
    if embeddings.device.type == 'mps':  # If the tensor is on MPS (Apple Silicon GPU)
        embeddings = embeddings.cpu()  # Move it to CPU
    embeddings_numpy=embeddings.detach().numpy() #convert embeddings to numpy array


    return embeddings_numpy


#indexing with FAISS
def index_faiss(embeddings):

    #embeddings=np.array(embeddings).astype('float32') #convert embeddings to numpy array
    index=faiss.IndexFlatL2(embeddings.shape[1]) #create a Faiss index for L2 distance
    index.add(embeddings) #add the embeddings to the index
    return index

#searching with FAISS
def search_faiss(query, index, paragraphs,model, k=1):
    query_embedding = model.encode([query]).astype('float32')     #generate the query embedding
    D, I = index.search(query_embedding, k) #search the index for the nearest neighbors
    results = [(paragraphs[i], D[0][j]) for j, i in enumerate(I[0])] #get the paragraphs and distances of the nearest neighbors
    return results


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
text=''
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    #st.write(text)
    paragraphs = preprocess_text(text)
    embeddings = generate_embeddings(paragraphs)
    index = index_faiss(embeddings)
    model= SentenceTransformer('all-MiniLM-L6-v2')
    query = st.text_input("Enter your question here:")
    results = search_faiss(query, index, paragraphs, model,k=1)
    joined_results = ''.join([text for text, _ in results])  # This works because you are extracting the string part

    #using Hugging Face model to generate answers
    prompt = f"Answer the following question using the provided information:\n\nQuestion: {query}\n\nContext:\n" + "\n".join(joined_results) + "\n\nAnswer:"
    generated_answer = load_huggingface_model(prompt)[0]['generated_text']
else:
    st.write("Please upload a PDF file to get started")

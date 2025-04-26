import streamlit as st
import pandas as pd
import os
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA

# Load laptop dataset
df = pd.read_csv("laptop_price.csv", encoding='latin1')

# Create description text
df['description'] = (
    "Brand: " + df['Company'] + ", " +
    "Product: " + df['Product'] + ", " +
    "Type: " + df['TypeName'] + ", " +
    "Screen: " + df['ScreenResolution'] + ", " +
    "CPU: " + df['Cpu'] + ", " +
    "RAM: " + df['Ram'] + ", " +
    "Memory: " + df['Memory'] + ", " +
    "GPU: " + df['Gpu'] + ", " +
    "OS: " + df['OpSys'] + ", " +
    "Weight: " + df['Weight'] + ", " +
    "Price: " + df['Price_euros'].astype(str) + " Euros."
)

# Prepare documents
documents = []
for idx, row in df.iterrows():
    doc = Document(page_content=row['description'], metadata={"price": row['Price_euros']})
    documents.append(doc)

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"


# Embedding and Vectorstore
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding)

# Setup retriever
retriever = vectorstore.as_retriever()

# Build Streamlit App
st.title("💬 SmartShop Assist AI")
st.write("Welcome! Search for laptops by typing your needs:")

user_query = st.text_input("What are you looking for?")

if user_query:
    results = retriever.get_relevant_documents(user_query, k=5)  # get top 5 matching laptops
    
    st.subheader("Matching Laptops:")
    for i, doc in enumerate(results):
        st.write(f"**{i+1}.** {doc.page_content}")

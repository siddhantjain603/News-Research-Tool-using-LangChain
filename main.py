import os
import time
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

from dotenv import load_dotenv
load_dotenv()

st.title("InsightBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # splitting the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " "],  
        chunk_size = 1000,
        chunk_overlap  = 200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # create embeddings and saving vector index locally
    embeddings = OpenAIEmbeddings()
    vectorindex_openai = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(3)
    vectorindex_openai.save_local("faiss_store")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists("faiss_store"):
        vector_index = FAISS.load_local("faiss_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever= vector_index.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # showing the results
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if availavle
        sources = result.get("sources","") # using "" to display nothing if sources are not present
        if sources:
            st.subheader("Sources...")
            sources_list = sources.split("\n") # Splitting the sources by new line
            for source in sources_list:
                st.write(source)











from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

'''def load_or_create_index(chunks,embeddings: HuggingFaceEmbeddings):
    if os.path.exists("faiss_index.faiss"):
        return FAISS.load("faiss_index.faiss")
    else:
        knowledge_base=FAISS.from_texts(chunks,embeddings)
        knowledge_base.save_local("faiss_index.faiss")
        return knowledge_base'''


def main():
    load_dotenv()
    #st.set_page_config(page_title="MY PDF CHATBOT1")
    #st.header("MY PDF CHATBOT")
    # Allow user to ask a question
    user_question = st.text_input("Ask a question:")
    
    # If a question is asked
    if user_question:
        # Load environment variables
        #index_path = os.getenv("INDEX_PATH", "faiss_index.faiss")
        st.write(os.path.exists("faiss_index.faiss1"))
        embeddings = HuggingFaceEmbeddings()
        if not os.path.exists("faiss_index.faiss1"):
            pdf = st.file_uploader("Upload your pdf",type="pdf")
            if pdf is not None:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # spilit ito chuncks
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=250,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)

                # create embedding
        
                knowledge_base=FAISS.from_texts(chunks,embeddings)
                knowledge_base.save_local("faiss_index.faiss1")
        else:    
            # Create or load FAISS index
            st.write("loading from local")
            knowledge_base = FAISS.load_local("faiss_index.faiss1",embeddings)

        # Ask the question about the PDF
        docs = knowledge_base.similarity_search(user_question)
        llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":2, "max_length":1000})
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)

        # Display the response
        st.write(response)


        # st.write(chunks)

if __name__ == '__main__':
    main()
## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")



# set up StreamLit App
st.title('Conversational RAG with PDF uploads & chat history')
st.write("Upload PDF's and chat with their content")

# Input The Groq API Key
api_key = st.text_input('Enter your GROQ API Key:', type='password')

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name='Gemma2-9b-it')
    
    session_id = st.text_input('Session ID', value='session-000')

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader('Choose A PDF file', type='pdf', accept_multiple_files=True)
    if uploaded_files:
        # process pdf
        documents = []
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        # Create a Vector DB
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        v_store = FAISS.from_documents(splits, embeddings)
        retriever = v_store.as_retriever()


        # Create a prompt to formulate a better question
        context_q_sys_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        context_q_prompt = ChatPromptTemplate.from_messages([
            ('system', context_q_sys_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

        # Create a prompt to Answer a Question
        sys_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ('system',sys_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ])

        qa_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Chat Hastory Handling
        def get_session_history (session:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        convo_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input('Your Question: ')
        if user_input:
            session_history = get_session_history(session_id)
            response = convo_rag_chain.invoke(
                {'input':user_input},
                config = {
                    'configurable': {'session_id':session_id}
                },
            )
            st.write(st.session_state.store)
            st.write('Assistant:', response['answer'])
            st.write('Chat History:', session_history.messages)
    else:
        st.warning('Please enter the GROQ API KEY')
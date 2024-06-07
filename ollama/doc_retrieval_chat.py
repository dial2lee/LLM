# import required libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from langchain.llms import HuggingFaceHub
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# Load the pdf file and split it into smaller chunks
loader = PyPDFLoader('F:\LLM\ollama\AOSP和LineageOS.pdf')
documents = loader.load()

# Split the documents into smaller chunks 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# We will use HuggingFace embeddings 
# embeddings = HuggingFaceEmbeddings()
embeddings = OllamaEmbeddings(model="qwen2:7b")

#Using Chroma vector database to store and retrieve embeddings of our text
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={'k': 2}) 

# We are using Mistral-7B for this question answering 
# repo_id = "mistralai/Mistral-7B-v0.1"
# llm = HuggingFaceHub(huggingfacehub_api_token='your huggingface access token here', 
#                      repo_id=repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})
llm = Ollama(model="qwen2:7b")

# Create the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever,return_source_documents=True) 

#We will run an infinite loop to ask questions to LLM and retrieve answers untill the user wants to quit
import sys
chat_history = []
while True:
    query = input('Prompt: ')
    #To exit: use 'exit', 'quit', 'q', or Ctrl-D.",
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer'])) 

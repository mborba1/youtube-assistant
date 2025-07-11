from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

video_url = "https://youtu.be/rtgN27z0oi0"  # Replace with your YouTube video URL
def create_vector_db_from_youtube(video_url, open_api_key, persist_dir="vectorstores") -> FAISS:
    from hashlib import md5
    video_id = md5(video_url.encode()).hexdigest()
    persist_path = os.path.join(persist_dir, f"{video_id}")
  
    if os.path.exists(persist_path):
        return FAISS.load_local(persist_path, OpenAIEmbeddings(openai_api_key=open_api_key),
        allow_dangerous_deserialization=True)

    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    embeddings = OpenAIEmbeddings(openai_api_key=open_api_key)
    db = FAISS.from_documents(docs, embeddings)
    
    os.makedirs(persist_dir, exist_ok=True)
    db.save_local(persist_path)
    return db
   
def get_response_from_query(db, query,open_api_key=None):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, openai_api_key=open_api_key)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=open_api_key)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
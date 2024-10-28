# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()
embeddings = OpenAIEmbeddings()


def get_data_from_file() -> FAISS:
    # loader = TextLoader("./data/sample.txt", encoding = 'UTF-8')
    # document = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # docs = text_splitter.split_documents(document)

     #use langchain PDF loader
    loader = PyPDFLoader("./data/project.pdf")

    #split the document into chunks
    pages = loader.load_and_split()


    db = FAISS.from_documents(documents=pages, embedding=embeddings)
    
    return db


def get_response_from_query(db, query, k=4):

    docs = db.similarity_search(query, k=k)

    docs_page_content = " ".join([d.page_content for d in docs])

    openAi = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        Answer the following question: {question}
        By searching the following document: {docs}
        Only use the factual information from the document to answer the question.
        If you feel like you don't have enough information to answer the question,
        say "I don't know" .
        Your answers should be detailed.
        """,
    )

    # Your answers should be verbose and detailed.



    chain = LLMChain(llm=openAi, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
    
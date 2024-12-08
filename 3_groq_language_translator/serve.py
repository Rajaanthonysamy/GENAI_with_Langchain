from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes
load_dotenv()


groq_api_key=os.getenv("GROQ_API_KEY")
groq=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

generic_template="Translate the following into {language}"
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",generic_template),
        ("user","{text}")
    ]
)

parser=StrOutputParser()

chain=prompt|groq|parser


app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple api server using Langchain runnable interfaces")

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8020)
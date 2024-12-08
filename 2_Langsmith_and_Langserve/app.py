import os 
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


prompt=ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a helpful assistant. Please respond to my question"),
        ("user","Question:{question}")
    ]
)

st.title("Langchain Demo With Gemma:2B")
input_text=st.text_input("Question ")

model=Ollama(model="gemma:2b")

output_parser=StrOutputParser()
chain=prompt|model|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

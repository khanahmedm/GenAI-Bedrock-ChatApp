from langchain.chains import LLMChain
#from langchain.llms.bedrock import Bedrock
from langchain_aws import BedrockLLM
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

os.environ["AWS_PROFILE"] = "testadmin"

#bedrock client

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

modelID = "anthropic.claude-3-haiku-20240307-v1:0"


llm = ChatBedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens": 2000}
)

def my_chatbot(language,freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response=bedrock_chain({'language':language, 'freeform_text':freeform_text})
    return response

#print(my_chatbot("english","who is buddha?"))


st.title("Bedrock Chatbot")

language = st.sidebar.selectbox("Language", ["english", "spanish"])

if language:
    freeform_text = st.sidebar.text_area(label="what is your question?",
    max_chars=100)

if freeform_text:
    response = my_chatbot(language,freeform_text)
    st.write(response['text'])

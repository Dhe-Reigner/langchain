from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7, openai_api_key = api_key)

prompt = PromptTemplate(
    input_variables = ['girlfriend'],
    template="Genarate a poem about my {girlfriend}"
)
chains = LLMChain(llm=llm, prompt=prompt)

print(chains.run('Love poem to my girlfriend'))

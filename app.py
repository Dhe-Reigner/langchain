# from langchain_openai import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from dotenv import load_dotenv
# import os

# load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")

# llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7, openai_api_key = api_key)

# prompt = PromptTemplate(
#     input_variables = ['girlfriend'],
#     template="Genarate a poem about my {girlfriend}"
# )
# chains = LLMChain(llm=llm, prompt=prompt)

# print(chains.run('Love poem to my girlfriend'))



# # Memory
# from langchain_openai import OpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from dotenv import load_dotenv
# import os


# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.7, openai_api_key = api_key)

# conversation = ConversationChain(
#     llm = llm,
#     verbose = True,
#     memory = ConversationBufferMemory()
# )

# conversation.predict(input = "Tell me about yourself. ")
# conversation.predict(input = "What can you do? ")
# conversation.predict(input = "How can you help me with data analysis? ")

# print(conversation)


# # Token Usage
# from langchain_openai import OpenAI
# from langchain_community.callbacks import get_openai_callback
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7, openai_api_key = api_key)

# with get_openai_callback() as cb:
#     results = llm.invoke("Tell me a poem about my girlfriend")
#     print(cb)



# #  Few-Shot Learning
# from langchain_core.prompts.few_shot import FewShotPromptTemplate
# from langchain_core.prompts.prompt import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.7, openai_api_key = api_key)

# examples = [
#     {
#         "query":"What is the weather like today?",
#         "answer":"It is likely to rain cats and dogs, better carry an umbrella"
#     },
#     {
#         "query":"What's your age?",
#         "answer":"Age is just a number, but I am timeless"
#     }
# ]
# example_template = """
# User:{query}
# AI:{answer}
# """

# example_prompt = PromptTemplate(
#     input_variables=["query", "answer"],
#     template = example_template
# )
# prefix = """
# The following are excerpts from conversations with an AI
# assistant. The assistant is known for its humor and wit, providing
# entertaining and amusing responses to users' questions. Here are some
# examples:
# """
# suffix = """
# User:{query}
# AI:
# """
# few_shot_prompt_template = FewShotPromptTemplate(
#     examples = examples,
#     example_prompt = example_prompt,
#     prefix = prefix,
#     suffix = suffix,
#     input_variables = ["query"],
#     example_separator = "\n\n"
# )
# chain = few_shot_prompt_template | llm
# print(chain.invoke("Some sweet words for my girlfriend?"))


# #  Movie Summarizer
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate
# )
# from dotenv import load_dotenv
# import os


# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key = api_key)

# template  = "You are a helpful movie summarizer, giving attention to details"
# system_prompt = SystemMessagePromptTemplate.from_template(template)

# human_template = "Generate information about {movie_title}"
# human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# chat_prompt = ChatPromptTemplate([system_prompt, human_prompt])

# response = llm.invoke(chat_prompt.format_prompt(movie_title="god's must be crazy").to_messages())

# print(response.content)



# # News Article Summarizer

# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI
# from newspaper import Article
# from dotenv import load_dotenv

# import os, requests, json


# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(model='gpt-4', temperature=0.7, openai_api_key = api_key)

# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
# }

# article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

# session = requests.Session()

# try:
#     response = session.get(article_url,headers=headers, timeout=100)

#     # if response.status_code == 200:
#     #     article=Article(article_url)
#     #     article.download()
#     #     article.parse()

#     response.raise_for_status()

#     article = Article(article_url)
#     article.set_html(response.text)
#     article.parse()

#     article_title = article.title
#     article_text = article.text

#     print(f"Title:{article.title}")
#     print(f"Text:{article.text}")

#     template = """You are an Intelligent Online Article Summarizer
#     Summarize the following article.

#     Title:{article_title}

#     {article_text}

#     Generate the previous article
#     """

#     prompt = template(article_title=article.title, article_text=article.text)
#     messages = [HumanMessage(content=prompt)]

#     summary = llm.messages
#     print(summary.content)

#     # else:
#     #     print(f"Failed to summarize {article_url}")
# except Exception as e:
#     print(f"An error occured when summarizing {article_url}:{e}")



# # Character Text Splitter

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

# loader = PyPDFLoader("FEES STATEMENT.pdf")
# pages = loader.load_and_split()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# texts = text_splitter.split_documents(pages)

# print(texts[0])

# # print(f"You have {len(texts)} documents")
# # print("Preview:")
# # print(texts[0].page_content)


# Recursive Chatracter Text Splitter

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key = api_key)

loader = PyPDFLoader("FEES STATEMENT.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = len,
)

docs = text_splitter.split_documents(pages)
for doc in docs:
    print(doc)
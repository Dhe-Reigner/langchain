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



#  Few-Shot Learning
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.7, openai_api_key = api_key)

examples = [
    {
        "query":"What is the weather like today?",
        "answer":"It is likely to rain cats and dogs, better carry an umbrella"
    },
    {
        "query":"What's your age?",
        "answer":"Age is just a number, but I am timeless"
    }
]
example_template = """
User:{query}
AI:{answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template = example_template
)
prefix = """
The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
suffix = """
User:{query}
AI:
"""
few_shot_prompt_template = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    prefix = prefix,
    suffix = suffix,
    input_variables = ["query"],
    example_separator = "\n\n"
)
chain = few_shot_prompt_template | llm
print(chain.invoke("Some sweet words for my girlfriend?"))

from langchain.agents import load_tools, Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

llm = OpenAI(temperature=0.7)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader

base_loader = TextLoader('base.txt')
base_documents = base_loader.load()
default_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = default_splitter.split_documents(base_documents)


import json
# load config.json
with open('config.json') as f:
    config = json.load(f)

for source in config['sources']:
    # source is a file name
    suffix = source.split('.')[-1]
    if suffix == 'txt' or suffix == 'org':
        loader = TextLoader(source)
    elif suffix == 'pdf':
        loader = UnstructuredPDFLoader(source)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts.extend(text_splitter.split_documents(documents))

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())


def search_notes(query):
    return qa.run(query)

from langchain.tools import DuckDuckGoSearchTool
ddg = DuckDuckGoSearchTool()
from langchain.utilities import PythonREPL
repl = PythonREPL()

from stats import list_tools


from pydantic import BaseModel, Field

class BetaCalculatorInputSchema(BaseModel):
    x_bar: float = Field(..., example=0.5)
    mu_0: float = Field(..., example=0.5)
    n: int = Field(..., example=100)
    s: float = Field(..., example=0.1)
tools = [
    Tool(
        name="stats_qa",
        description="Find extra information about statistical concepts.",
        func=qa.run
    ),
    Tool(
        name="python",
        description="Run python code in the REPL. Or execute some calculation. Do not search any information here.",
        func=repl.run
    ),
    *list_tools()
]


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)

template = '''
You are a data analyst. It is your job to solve problems and answer questions.
Problem to solve:
{problem}
---
Questions || goals:
{need_to_know}
---
When using the Python tool, always print all the variables you want to see. For example, if you want to see the value of x, you should print(x) instead of just x.
Here are approaches you should take to solve different kinds of problems.
Hypothesis Testing:
    1. State the null and alternative hypothesis
    2. Choose a significance level (alpha)
    3. Calculate the test statistic
    4. Determine the critical value
    5. Compare test statistic with critical value
    6. State the conclusion
'''

import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True
import json

@app.route('/api/v1/ask', methods=['POST'])
def api_id():
    # implement the solve task
    context = request.json['context']
    goals = request.json['goals']
    inall = template.format(problem=context, need_to_know=goals)
    response = agent({inall})
    return jsonify({'response': response['output'], 'intermediate_steps': response['intermediate_steps']})

# add route for the search notes method
@app.route('/api/v1/search', methods=['POST'])
def api_search():
    query = request.json['query']
    response = search_notes(query)
    return jsonify({'response': response})



app.run()

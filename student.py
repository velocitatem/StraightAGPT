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

stats_tool = Tool(
    name="stats",
    description="Search notes on statistics and how to use python to calculate them. You can also ask what to do when solving some problem.",
    func=qa.run
)

def search_notes(query):
    return qa.run(query)

from langchain.tools import DuckDuckGoSearchTool
ddg = DuckDuckGoSearchTool()
from langchain.utilities import PythonREPL
repl = PythonREPL()
tools = []
tools.append(stats_tool)
#tools.append(ddg)
python_calc_tool = Tool(
    name="python",
    description="Run python code in the REPL. Or execute some calculation. Do not search any information here.",
    func=repl.run
)
tools.append(python_calc_tool)


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)

template = '''
Context:
{problem}
---
Given the above problem, solve it and select one of the following options. Return only the number of the correct option.
{need_to_know}
---
When using the Python tool, always print all the variables you want to see. For example, if you want to see the value of x, you should print(x) instead of just x. You must only respond with the number of the correct option, no other text is allowed.
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

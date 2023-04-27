from langchain.agents import load_tools, Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

llm = OpenAI(temperature=0.3)

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

import sys
mode = sys.argv[1]
if mode in config['modes'].keys():
    config = config['modes'][mode]
else:
    raise ValueError(f'Unknown mode {mode}')

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

# docsearch = Chroma.from_documents(texts, embeddings)

# qa = RetrievalQA.from_chain_type(llm=OpenAI(),
#                                  chain_type="stuff",
#                                  retriever=docsearch.as_retriever())

def qa_from_filenames(filenames):
    base_loader = TextLoader('base.txt')
    base_documents = base_loader.load()
    default_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = default_splitter.split_documents(base_documents)

    for source in filenames:
        # source is a file name
        suffix = source.split('.')[-1]
        if suffix in ['txt', 'org', 'md']:
            loader = TextLoader(source)
        elif suffix == 'pdf':
            loader = UnstructuredPDFLoader(source)
        else:
            raise ValueError(f'Unknown file type {suffix}')

        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts.extend(text_splitter.split_documents(documents))

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_documents(texts, embeddings, persist_directory='db')


    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                     chain_type="stuff",
                                     retriever=docsearch.as_retriever())
    return qa

#nextStepsQA = qa_from_filenames(['./next_steps.org'])


from langchain.tools import DuckDuckGoSearchTool
ddg = DuckDuckGoSearchTool()
from langchain.utilities import PythonREPL
repl = PythonREPL()

toolkit = __import__(mode)


tools = [
    Tool(
        name="python",
        description="Run python code in the REPL. Or execute some calculation. Do not search any information here.",
        func=repl.run
    ),
    *toolkit.list_tools() # extracts all the tools from the stats.py file and adds them to the list
]


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)

with open(f'{mode}.prompt') as f:
    template = f.read()

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
    print(response)
    res = response['output']
    try:
        return jsonify({'response': res, 'intermediate_steps': response['intermediate_steps']})
    except:
        return jsonify({'response': res})

# add route for the search notes method
@app.route('/api/v1/search', methods=['POST'])
def api_search():
    query = request.json['query']
    response = search_notes(query)
    print(response)
    return jsonify(response)



app.run()

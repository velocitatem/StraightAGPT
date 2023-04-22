from langchain.agents import load_tools, Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
#loader = TextLoader('/mnt/s/Documents/Collage/university-study-notes/content/20221231174507-fundamentals_of_data_analysis_class_notes.org')
from langchain.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader('/home/velocitatem/Downloads/Hypothesis Testing One Sample-3.pdf')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
from langchain.vectorstores import Chroma
docsearch = Chroma.from_documents(texts, embeddings)

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

stats_tool = Tool(
    name="stats",
    description="Search notes on statistics and how to use python to calculate them. You can also ask what to do when solving some problem.",
    func=qa.run
)
from langchain.tools import DuckDuckGoSearchTool
ddg = DuckDuckGoSearchTool()
from langchain.utilities import PythonREPL
repl = PythonREPL()
tools = []
tools.append(stats_tool)
tools.append(ddg)
python_calc_tool = Tool(
    name="python",
    description="Run python code in the REPL. Or execute some calculation. Do not search any information here.",
    func=repl.run
)
tools.append(python_calc_tool)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)

template = '''
:State the null and alternative hypotheses;
:Identify the test statistic;
if (Population standard deviation is known?) then (yes)
  if (Sample size is large?) then (yes)
    :Use the z-test;
  else (no)
    :Use the t-test;
  endif
else (no)
  :Use the t-test;
endif
:Find the p-value;
if (p <= alpha) then (yes)
  :Reject H0;
else (no)
  :Fail to reject H0;
endif
if (p <= alpha) then (yes)
  :The result is statistically significant;
else (no)
  :The result is not statistically significant;
endif

You will not always have to use the above methodology, only use it if you should conduct an entire hypothesis test. Solve the following problem:
{problem}
We also need to know the following intermediate steps:
{need_to_know}
---
When using the Python tool, always print all the variables you want to see. For example, if you want to see the value of x, you should print(x) instead of just x.
'''

while True:
    task = input("Task: ")
    if task == "exit":
        break
    if task == 'solve':
        prompt = input("Context of Question: ")
        prompt_2 = input("Goals: ")
        response = agent.run(template.format(problem=prompt, need_to_know=prompt_2))
        print("A: ", response)
    else:
        response = agent.run(task)
        print("A: ", response)

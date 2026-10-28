from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool
import os

from crewai.tools import tool
from duckduckgo_search import DDGS

os.environ["SERPER_API_KEY"] = "<YOUR SERPER API KEY HERE>" # d11eed318aa5478ab36b1345b8583cbcd90743b4

# Указываем модель через префикс ollama/
# CrewAI под капотом обратится к локальному серверу Ollama (http://localhost:11434)
local_llm = "ollama/qwen2.5:14b"

# Tools
search = SerperDevTool()

@tool("Free DuckDuckGo Search")
def free_ddg_search(query: str) -> str:
  """Searches the web using DuckDuckGo free API."""
  with DDGS() as ddgs:
    results = [r["body"] for r in ddgs.text(query, max_results=3)]
    return "\n".join(results) if results else "No results found."

# Create the agent
researcher = Agent(
    llm=local_llm,
    role="Senior AI Researcher",
    goal="Find promising research in the field of quantum computing.",
    backstory="You are a veteran quantum computing researcher with a background in modern physics.",
    allow_delegation=False,
    tools=[free_ddg_search], # search, free_ddg_search],
    verbose=True,
)

# Create a task
task1 = Task(
    description=(
        "Search the internet and find 5 examples of promising AI research."
    ),
    expected_output=(
        "A detailed bullet point summary on each of the topics. Each bullet"
        " point should cover the topic, background and why the innovation is"
        " useful."
    ),
    output_file="task1output.txt",
    agent=researcher,
)

# Create the second agent
writer = Agent(
    llm=local_llm,
    role="Senior Speech Writer",
    goal="Write engaging and witty keynote speeches from provided research.",
    backstory=(
        "You are a veteran quantum computing writer with a background in"
        " modern physics."
    ),
    allow_delegation=False,
    verbose=True,
)

# Create a task
task2 = Task(
    description="Write an engaging keynote speech on quantum computing.",
    expected_output=(
        "A detailed keynote speech with an intro, body and conclusion."
    ),
    output_file="task2output.txt",
    agent=writer,
)

# Put all together with the crew
crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=True)
print(crew.kickoff())

from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
from langchain_community.llms import Ollama
ollama_openhermes = Ollama(model="ollama/llama3.2:latest")

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://tradingeconomics.com/vietnam/stock-market"
)


local_expert = Agent(
      role='The Best Financial Analyst',
      goal="""Impress all customers with your financial data 
      and market trends analysis""",
      backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
      verbose=True,
      llm=ollama_openhermes, # Ollama model passed here
      tools=[
        docs_scrape_tool
      ]
    )

# Create a task named "edit" for "editor" agent.
edit = Task(
    description="Analyse the Financial Analyst's report and correct it",
    expected_output="A well-written report in markdown format, "
                    "ready for publication.",
    agent=local_expert
)
# Create a crew with the three agents
crew = Crew(
    name="Content Creation Crew",
    agents=[local_expert],
    tasks=[edit],
    verbose=True,
    chat_llm=ollama_openhermes
)


result = crew.kickoff(inputs={"topic": "search for the best stocks to invest in 2025 in VietNam"})

print(result)
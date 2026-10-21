import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch
from tavily import TavilyClient

class SearchResult(BaseModel):
    """
    Model representing a single search result.
    Attributes:
        title (str): The title of the search result.
        url (str): The URL of the search result.
        snippet (str): A brief snippet or summary of the search result.
    """
    title: str = Field(..., description="The title of the search result")
    url: str = Field(..., description="The URL of the search result")
    snippet: str = Field(..., description="A brief snippet or summary of the search result")

class SearchResults(BaseModel):
    """Container model for a list of search results."""
    results: list[SearchResult] = Field(..., description="List of search results")

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches the web for information related to the query using the Tavily Search API.
    Args: 
        query (str): The search query.
    Returns:
        str: A string containing the search results.
    """
    print(f"Searching for: {query}")
    results = tavily.search(query)
    return results

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="qwen/Qwen3.8-27B",
    temperature=0.7
)

#tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=SearchResults)

def main() -> None:
    print("Starting the LangChain course example...")
    query = "Search for 3 jobs on LinkedIn for an AI architect role in Noida"
    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(response)

if __name__ == "__main__":
    main()

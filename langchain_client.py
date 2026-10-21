from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage


load_dotenv()

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="qwen/Qwen3.8-27B",
    temperature=0.7
)

stdio_server_params = StdioServerParameters(
    command="python", args=["servers/math_server.py"], env=os.environ.copy())

async def main():
    print("Starting the LangChain client...")
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read,write_stream=write) as session:
            print("Connected to the Math server via stdio.")
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_agent(llm,tools)
            result = await agent.ainvoke({"messages":[HumanMessage(content="what is 8*2+2")]})
            print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10


@tool
def get_product_price(product_name: str) -> float:
    """
    Tool that retrieves the price of a product from an e-commerce website.
    Args:
        product_name (str): The name of the product to search for.
    Returns:
        float: The price of the product as a float.
    """
    print(f"Fetching price for product: {product_name}")
    prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
        "smartwatch": 249.99,
        "tablet": 399.99,
    }
    return prices.get(product_name.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Tool that applies a discount percentage based on the selected tier.
    Args:
        price (float): The original price of the product.
        discount_tier (str): The discount tier to apply.
    Returns:
        float: The final price after applying the discount.
    """
    print(f"Fetching discount for product with price: {price}")
    discounts_percentage = {"gold": 23, "silver": 12, "bronze": 5}
    final_price = round(price * (discounts_percentage.get(discount_tier.lower(), 0) / 100), 2)
    print(f"Final price for {discount_tier} tier is: {final_price}")
    return final_price


@traceable(name="Lang chain agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_map = {tool.name: tool for tool in tools}
    llm = init_chat_model(model="qwen/Qwen3.8-27B", model_provider="groq", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"question: {question}")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that can answer questions about e-commerce products. "
                "You have access to the following tools: "
                "1. get_product_price(product_name: str) -> float: Retrieves the price of a product. "
                "2. apply_discount(price: float, discount_tier: str) -> float: Applies a discount to a product price based on the discount tier. "
                "STRICT RULES: You must use the tools to answer questions about product prices and discounts. "
                "Never make up prices or discounts. If you don't know the answer, use the tools to find it."
            )
        ),
        HumanMessage(content=question),
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {i}: Invoking LLM with messages: {[message.content for message in messages]}")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            print("Final answer:", response.content)
            break

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call.get("args", {})
            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:
                print(f"Tool {tool_name} not found.")
                break

            tool_output = selected_tool.invoke(tool_input)
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )


if __name__ == "__main__":
    print("Starting the LangChain course example...")
    question = "What is the price of a laptop for a gold member?"
    run_agent(question)

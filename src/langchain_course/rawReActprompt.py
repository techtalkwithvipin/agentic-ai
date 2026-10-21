import re
import json

from dotenv import load_dotenv
from langchain_core import messages
from langsmith import traceable
import inspect
import groq
from langchain_groq import ChatGroq 

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10


def parse_tool_input(raw_input: str):
    """Support both JSON-style and key=value action inputs from the LLM."""
    stripped = raw_input.strip()

    if not stripped:
        return {}

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    kwargs = {}
    for item in [part.strip() for part in stripped.split(",") if part.strip()]:
        if "=" in item:
            key, value = item.split("=", 1)
            kwargs[key.strip().strip("'\"")] = value.strip().strip("'\"")
    return kwargs


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


tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}


def get_tool_descriptions(tools_dict):
    tool_descriptions = []
    for tool_name, tool_func in tools_dict.items():
        callable_target = getattr(tool_func, "func", tool_func)
        original_tool_func = getattr(callable_target, "__wrapped__", callable_target)
        signature = inspect.signature(original_tool_func)
        doc_string = inspect.getdoc(tool_func) or inspect.getdoc(callable_target) or ""
        tool_descriptions.append(f"{tool_name}{signature} - {doc_string} ")
    return "\n".join(tool_descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES: You must follow these exactly.
1. Never guess or assume any price. You must call get_product_price to get the price of a product.
2. Never guess or assume any discount. You must call apply_discount to get the discounted price of a product. Pass the exact price return by get_product_price to apply_discount.
3. Never caclate the final price yourself. You must call apply_discount to get the final price of a product.
4. if user doen't specify the pricing tier, you must ask the user to specify the pricing tier before calling apply_discount. do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="groq chat", run_type="llm")
def groq_chat_traced(model, messages, **options):
    """
    A traced function that interacts with the Groq chat model.
    Args:
        model: The chat model to use for generating responses.
        messages: The input messages to send to the model.
        options: Additional options for the model invocation.
    Returns:
        The response from the chat model.
    """
    return model.invoke(messages, **options)


@traceable(name="Lang chain agent loop")
def run_agent(question: str):
    print(f"Running agent for question: {question}")
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    groq_chat = ChatGroq(model="qwen/Qwen3.8-27B", temperature=0)

    for i in range(1, MAX_ITERATIONS + 1):
        fullprompt = prompt + scratchpad
        print(f"Iteration {i}: Invoking LLM with prompt length {len(fullprompt)}")
        response = groq_chat_traced(
            groq_chat,
            [{"role": "user", "content": fullprompt}],
            stop=["\nObservation"],
            temperature=0,
        )

        output = response.content
        print(f"LLM output: {output}")
        final_answer_match = re.search(r"Final Answer:\s*(.*)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("Final answer:", final_answer)
            return final_answer

        action_match = re.search(r"Action:\s*(\w+)", output)
        action_input_match = re.search(r"Action Input:\s*(.*)", output)

        if not action_match or not action_input_match:
            print("No action or action input found in the output.")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        print(f"Tool name: {tool_name}, Tool input raw: {tool_input_raw}")

        tool_kwargs = parse_tool_input(tool_input_raw)
        print(f"Tool executing with args: {tool_kwargs}")

        if tool_name not in tools:
            observation = f"Error: Tool {tool_name} not found. Available tools: {list(str(tools.keys()))}"
            break

        tool_obj = tools[tool_name]
        callable_target = getattr(tool_obj, "func", tool_obj)
        if tool_kwargs:
            observation = str(callable_target(**tool_kwargs))
        else:
            observation = str(callable_target())

        print(f"Observation: {observation}")
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("Max iterations reached without finding a final answer.")
    return None


if __name__ == "__main__":
    print("Starting the LangChain course example...")
    question = "What is the price of a laptop for a gold member?"
    run_agent(question)

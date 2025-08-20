import asyncio
import mlflow
import os
import os
from openai import AsyncAzureOpenAI
import logfire
from langfuse import get_client, Langfuse
from src.utils.functions import*
import gradio as gr
from gradio.components.chatbot import ChatMessage
from dotenv import load_dotenv
load_dotenv()

from src.utils.pretty_printing import pretty_print

from src.utils.azure_openai.client import get_openai_client
from src.utils.tools.mongodb.atlas_mongo_util import MongoManager
from agents import Agent, Runner, ModelSettings, function_tool, OpenAIChatCompletionsModel, trace
from agents import set_default_openai_client,set_default_openai_api,set_tracing_disabled
from pydantic import BaseModel
from src.utils.gradio.messages import oai_agent_stream_to_gradio_messages

mongo = MongoManager()
openai_client = get_openai_client()

# Set the default OpenAI client for the Agents SDK at the global level once
set_default_openai_client(openai_client)
set_default_openai_api ("chat_completions")
set_tracing_disabled(True)

class AgentOutput(BaseModel):
    reasoning: str
    sourceUrl: list[str]
    productID: list[str]


# def structured_output(reasoning: str, source_url: list[str], product_id: list[str]) -> AgentOutput:
#     """Structure the output from the agent into a AgentOutput model."""
#     return AgentOutput(reasoning=reasoning, sourceUrl=source_url, productID=product_id).model_dump_json()

# Enable automatic tracing for your framework
mlflow.openai.autolog()  # For OpenAI

# Creates local mlruns directory for experiments
mlflow.set_experiment("ask_and_answer_experiment_1:03pm")

executor_agent = Agent(
    name="ProductSupportAgent",
    instructions=(
        "You are a product support assistant with access to a manufacturer's product manuals. Given a search query, use the perform_vector_search tool to retrieve the relevant information.\
         Do NOT return raw search results."
    ),
    tools=[
        function_tool(mongo.perform_vector_search),
    ],
    # model=OpenAIChatCompletionsModel(
    #     model="gpt-4o-2024-08-06", openai_client=openai_client
    # ),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model_settings=ModelSettings(temperature=0),
)

planner_instructions = read_instructions("instructions.md")

# Main Agent: Orchestrator
main_agent = Agent(
    name="MainAgent",
    instructions=planner_instructions,
    tools=[
        executor_agent.as_tool(
            tool_name="ProductSupportAgent",
            tool_description="Perform search for a query and return a concise summary.",
        ), 
        # function_tool(structured_output)

    ],
    # model=OpenAIChatCompletionsModel(
    #     model="gpt-4o-2024-08-06", openai_client=openai_client
    # ),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model_settings=ModelSettings(temperature=0),
    output_type=AgentOutput
)


# Simple robust Gradio chat handler
async def chat_handler(messages: list[ChatMessage], state=None):
    # Ensure messages is a list
    if isinstance(messages, str):
        messages = [ {"role": "user", "content": messages} ]
    elif not isinstance(messages, list):
        messages = []

    # Extract the latest user message
    if messages and hasattr(messages[-1], "content"):
        user_message = messages[-1].content
    elif messages and isinstance(messages[-1], dict) and "content" in messages[-1]:
        user_message = messages[-1]["content"]
    else:
        user_message = ""

    # Call the agent
    result_main = Runner.run_streamed(main_agent, input=user_message)
    chat_history = messages.copy() if isinstance(messages, list) else list(messages)
    async for _item in result_main.stream_events():
        chat_history += oai_agent_stream_to_gradio_messages(_item)
        # pretty_print(chat_history)
        yield chat_history


with gr.Blocks() as demo:
    gr.Image(os.getenv("CANADIAN_TIRE_LOGO_URL"), show_label=False, width=150)
    gr.ChatInterface(
        chat_handler,
        title="Customer Support",
        type="messages",
        examples=[
            "Can you recommend a paderno kettle that has a capacity more than 1.5L?",
            "What is the warranty period for Breville espresso machines?",
            "Show me kettles with temperature control.",
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True)
    # asyncio.run(main())
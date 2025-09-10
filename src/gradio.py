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
# from agents.events import RunItemStreamEvent
from src.utils.azure_openai.client import get_openai_client
from src.utils.tools.mongodb.atlas_mongo_util import MongoManager
from agents import Agent, Runner, ModelSettings, function_tool, OpenAIChatCompletionsModel, trace
from agents import set_default_openai_client,set_default_openai_api,set_tracing_disabled
from pydantic import BaseModel
# from openai.types.response import RawResponsesStreamEvent, ResponseOutputMessage
from src.utils.gradio.messages import oai_agent_stream_to_gradio_messages

mongo = MongoManager()
openai_client = get_openai_client()

# Set the default OpenAI client for the Agents SDK at the global level once
set_default_openai_client(openai_client)
set_default_openai_api ("chat_completions")
set_tracing_disabled(True)

# def extract_final_and_details(agent_output):
#     if (
#         isinstance(agent_output, RawResponsesStreamEvent)
#         and hasattr(agent_output.data, "item")
#         and isinstance(agent_output.data.item, ResponseOutputMessage)
#         and getattr(agent_output.data.item, "role", None) == "assistant"
#     ):
#         # This is the final agent output!
#         for content in agent_output.data.item.content:
#             if hasattr(content, "text"):
#                 return(content.text)  # This is your answer as a JSON string
#     return ""

class AgentOutput(BaseModel):
    reasoning: str
    sourceUrl: list[str]
    productID: list[str]
class FormatterOutput(BaseModel):
    final_output: str


# def structured_output(reasoning: str, source_url: list[str], product_id: list[str]) -> AgentOutput:
#     """Structure the output from the agent into a AgentOutput model."""
#     return AgentOutput(reasoning=reasoning, sourceUrl=source_url, productID=product_id).model_dump_json()

# Enable automatic tracing for your framework
mlflow.openai.autolog()  # For OpenAI

# Creates local mlruns directory for experiments
mlflow.set_experiment("ask_and_answer_experiment_1:03pm")


async def _generate_final_answer(formatter_agent, input_data: list[str], current_query: str) -> AgentOutput:
    response = await Runner.run(formatter_agent, input = input_data)
    return response



async def _main(question: str, gr_messages: list[ChatMessage]):
    chat_history = []
    for msg in gr_messages:
        if hasattr(msg, "role") and hasattr(msg, "content"):
            chat_history.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, dict) and "role" in msg and "content" in msg:
            chat_history.append({"role": msg["role"], "content": msg["content"]})

    formatter_agent = Agent(
        name = "FormatterAgent",
        instructions = (
            "Given the following conversation history and the latest user question, "
            "generate a concise, standalone question suitable for retrieving relevant product information. "
            "Do not include chit-chat or irrelevant context."),
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model_settings = ModelSettings(temperature=0),
    output_type = FormatterOutput
    )

    formatter_input = f"history: {chat_history} \n question: {question}"
    formatted_question = await _generate_final_answer(formatter_agent, formatter_input, question)
    print ("Formatted question:", formatted_question)
    print ("chat history:", chat_history)
    if not gr_messages or len(gr_messages) == 0:
        focused_question = question
    else:
        if hasattr(formatted_question, "final_output"):
            print ("**** Formatted question is an object with final_output:", str(formatted_question.final_output))
            focused_question = str(formatted_question.final_output)
        elif isinstance(formatted_question, str):
            print ("Formatted question is a string:", formatted_question)
            focused_question = formatted_question.final_output
        else:
            print ("Unknown formatted question type:", type(formatted_question))
            focused_question = str(formatted_question.final_output)
    
    
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
    model_settings=ModelSettings(temperature=0,tool_choice="required")
    # model_settings=ModelSettings()
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
    print ("===============================")
    print ("Focused question:", focused_question)
    print ("question:", question)
    print ("formatted_question:", formatted_question)
    print ("===============================")
    result_main = Runner.run_streamed(main_agent, input=focused_question)
    
    # print (result_main)
    # print ("===============================")
    async for _item in result_main.stream_events():
        # try:
        #     if _item.name == "tool_output":
        #     # if (type(_item) == "run_item_stream_event"):
        #         # print (_item)
        #         # print (_item.item)
        #         gr_messages.append(
        #         gr.ChatMessage(
        #             content=_item.item.output,
        #             role="assistant"
        #         )
        #     )
        # except Exception as e:
        #     continue
        gr_messages += oai_agent_stream_to_gradio_messages(_item)
        yield gr_messages


# Simple robust Gradio chat handler
# async def chat_handler(messages: list[ChatMessage], state=None):
#     # Ensure messages is a list
#     if isinstance(messages, str):
#         messages = [ {"role": "user", "content": messages}]
#     elif not isinstance(messages, list):
#         messages = []

#     # Extract the latest user message
#     if messages and hasattr(messages[-1], "content"):
#         user_message = messages[-1].content
#     elif messages and isinstance(messages[-1], dict) and "content" in messages[-1]:
#         user_message = messages[-1]["content"]
#     else:
#         user_message = ""
#     print ("User message:", user_message)
#     print ("Full messages:", messages)
#     user_message = messages

#     # Call the agent
#     result_main = Runner.run_streamed(main_agent, input=user_message)
#     print ("===============================")
#     print (result_main)
#     print ("===============================")
#     chat_history = messages.copy() if isinstance(messages, list) else list(messages)
#     all_outputs = []




#     async for  _item in (result_main.stream_events()):
#         all_outputs.append(_item)
#         # print (_item)
#         # print ("-----------------------")
#         # print (_item.type)
#         # print ("================================")
#         # try:
#         #     print (_item.data)
#         # except Exception as e:
#             # continue
#         try:
#             print ("-----------------------")
#             # print (_item)
#             print ("-----------------------")
#         except Exception as e:
#             continue
#         try:
#             if _item.name == "tool_output":
#                 print ("FOUND TOOL CALLED")
#                 # print (type(_item.data))
#             # and _item.data.name == "tool_output":
#             # import pdb; pdb.set_trace()
#                 try:
#                     chat_history += oai_agent_stream_to_gradio_messages(_item)
#                 except Exception as e:
#                     print ("Exception found", e)
#                 # print (chat_history)
#         except Exception as e:
#             continue
        
#         # chat_history += oai_agent_stream_to_gradio_messages(_item)
#         # pretty_print(chat_history)
#         # final_answer, details_markdown = extract_final_and_details(_item)
#         # Only show the final answer in the chat, but attach details as metadata
#         # chat_history.append(
#         #     gr.ChatMessage(
#         #         content=final_answer,
#         #         role="assistant",
#         #         metadata={"details": details_markdown}
#         #     )
#         # )
#         yield chat_history
#     # print (all_outputs)


with gr.Blocks(theme=gr.themes.Soft(), css=".footer {text-align:center; font-size:0.9em; color:gray;}") as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=160):
            gr.Image(os.getenv("CANADIAN_TIRE_LOGO_URL"), show_label=False, width=120)
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # 🛠️ Canadian Tire Product Support Chat
                Welcome! Ask any product-related question.  
                The assistant uses manufacturer manuals and smart search to help you.
                """
            )
    gr.ChatInterface(
        _main,
        title="Customer Support",
        type="messages",
        examples=[
            "Can you recommend a paderno kettle that has a capacity more than 1.5L?",
            "What is the warranty period for Breville espresso machines?",
            "Show me kettles with temperature control.",
        ]
    )
    gr.Markdown(
        "<div class='footer'>© 2025 Canadian Tire Product Support. Powered by Azure OpenAI & MongoDB.</div>"
    )


if __name__ == "__main__":
    demo.launch(share=True)
    # asyncio.run(main())

#Changes:
# -  Remove the question, tool output and keep the final answer only.
# - Pass context to ensure conversation   
# - 
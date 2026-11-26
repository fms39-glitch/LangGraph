from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages #This is a 'reducer function' we won't append but rather use this add_message directly   
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] #annotate(datatype, keyword)  eg. Annotate(str, "Shoud be in the email format")


# creating the first tool

@tool
def add(a: int, b: int):
    """This is an addition funtion that adds two numbers"""

    return a+b
# When you put @tool above def add(a: int, b: int):, LangChain inspects your function and converts its name, parameter types (a: int, b: int), and docstring into a JSON schema that gets sent to OpenAI's API.

tools = [add]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    #response = model.invoke("You are my AI assistant, please answer my query to the best of your ability")
    # or
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    
    response = model.invoke([system_prompt] + state["messages"]) # The input message will be stored and invoked to the AI with the system prompt. 
# Your system prompt and your message both go to an AI provider, and it generates the response in the following manner:
# 
# response = AIMessage(
#    content="",  # Empty! No text answer yet
#    tool_calls=[
#        {
#            "name": "add",
#            "args": {"a": 40, "b": 12},
#            "id": "call_abc123"
#        }
#    ]
#   )
   
    return {"messages": [response]} #The alternative is apending but since we are using the 'reducer function' it will appennd automatically


def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"



graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))




#WorkFlow:
# • Now state["messages"] holds 3 messages:
#	1. HumanMessage("Add 40 + 12 and then multiply the result by 6...")
#	2. AIMessage(content="", tool_calls=[add(40, 12)])
#	3. ToolMessage(content="52", tool_call_id="call_abc123")
#• model.invoke([system_prompt] + state["messages"]) sends all of this back to GPT-4o. Seeing 52 in the ToolMessage, GPT-4o multiplies 52 * 6 = 312, writes a joke, and returns a final AIMessage with normal .content and an empty tool_calls=[].
#• should_continue runs again, sees not last_message.tool_calls is True, returns "end", and stops the graph at END.

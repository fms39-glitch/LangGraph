import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) 
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"]) 
# us waqt graph ke state mein jitne bhi messages save hain,  print 

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

### Work FLow:
# 1. Sabse pehle user apna input dega aur woh `conversation_history` list mein add ho jayega.
# 2. Phir `agent.invoke` is poori list ko graph ke `process` node ke paas bhejega.
# 3. `process` node mein LLM poori history padh kar apna reply generate karega aur use list mein append kar dega.
# 4. Graph apna kaam khatam karke updated state wapas `result` dictionary mein return karega.
# 5. Last mein `result["messages"]` se updated chat wapas `conversation_history` mein save ho jayegi taaki agle turn mein AI ko sab yaad rahe.


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")

# The above piece of code saves your conversation history in a text file. 
# `isinstance(obj, class)` checks whether the following object is an instance of the class. 
#  Since we are using `union` and we have two objects within one list, 
#  what we need is an instance to differentiate which is 'you' and which is 'AI'. 
# Example: 
#Your Conversation Log:
#You: hi
#AI: Hello! How can I help you today?

#You: what is 2+2?
#AI: 2 + 2 is 4.

#End of Conversation
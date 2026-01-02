import os
import json
import asyncio
from typing import Annotated, TypedDict, List

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from langsmith import traceable

from langchain_mcp_adapters.client import MultiServerMCPClient


#############################################
# 1. ENV + HR POLICY RAG SETUP
#############################################

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHROMA_DIR = "chroma_hr_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})


#############################################
# 2. MCP CLIENT (REMOTE HRMS)
#############################################

SERVERS = {
    "hrms": {
        "transport": "streamable_http",
        "url": "https://first-mcp.fastmcp.app/mcp"
    }
}

_mcp_client = None
_mcp_tools = None


async def init_mcp():
    global _mcp_client, _mcp_tools
    _mcp_client = MultiServerMCPClient(SERVERS)
    _mcp_tools = await _mcp_client.get_tools()


#############################################
# 3. TOOLS
#############################################

@tool
@traceable(name="HR Policy RAG Tool")
def HR_chatbot(query: str) -> str:
    """
    Answer HR policy questions using RAG.
    """
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


@tool
@traceable(name="HRMS Employee Tool")
def get_employee_hrms_tool(employee_id: str) -> str:
    """
    Fetch HRMS details for a given employee_id from remote MCP server.
    """

    hrms_tool = None
    for t in _mcp_tools:
        if t.name == "get_employee_hrms":
            hrms_tool = t
            break

    if hrms_tool is None:
        return "HRMS service unavailable."

    result = asyncio.run(
        hrms_tool.ainvoke({"employee_id": employee_id})
    )

    return json.dumps(result, indent=2)


@tool
@traceable(name="Escalation HITL Tool")
def Escalation_tool(issue: str) -> str:
    """
    Human approval before escalation.
    """

    email_preview = f"""
Subject: Escalation Request

Issue reported:
-------------------------
{issue}

Regards,
Employee
"""

    decision = interrupt(
        {
            "message": f"Approve escalation for issue '{issue}'?",
            "preview_email": email_preview
        }
    )

    if isinstance(decision, str) and decision.lower() == "yes":
        os.makedirs("Escalation_folder", exist_ok=True)
        path = os.path.join("Escalation_folder", "escalation_mail.txt")
        with open(path, "w") as f:
            f.write(email_preview)
        return f"Escalation completed and saved to {path}"
    else:
        return "Escalation cancelled."


tools = [HR_chatbot, get_employee_hrms_tool, Escalation_tool]


#############################################
# 4. MODEL
#############################################

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


#############################################
# 5. STATE
#############################################

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


#############################################
# 6. CHAT NODE
#############################################

@traceable(name="HR Chat Node")
def chat_node(state: ChatState):

    instructions = SystemMessage(content=(
        "You are an HR Assistant named px_assistant.\n"
        "Rules:\n"
        "1. If the user greets, reply: 'Hi employee, welcome to the px_assistant'.\n"
        "2. For HR policy questions, use HR_chatbot.\n"
        "3. If the user asks about employee-specific details "
        "(name, department, salary, leave balance) and use get_employee_hrms_tool.\n"
        "4. Use tool output as authoritative truth.\n"
        "5. If the user is unhappy or complains, use Escalation_tool.\n"
        "6. Do NOT hallucinate.\n"
        "7. Be concise and professional."
    ))

    messages = [instructions] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


#############################################
# 7. GRAPH
#############################################

memory = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

app = graph.compile(checkpointer=memory)


#############################################
# 8. CLI + HITL LOOP
#############################################

if __name__ == "__main__":

    print("\n🚀 HR HITL + MCP Assistant Ready\n")

    asyncio.run(init_mcp())
    emp_id='EMP003'

    thread_id = emp_id
    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    while True:
        user_info=f'The employee id of the employee is {emp_id}. Here is the user question: '
        user = input("\nYou: ")
        if user.lower() in {"exit", "quit"}:
            break
        user = user_info + user
        #print('final_prompt: ',user)
        

        print("\nAssistant: ", end="", flush=True)

        for msg, _ in app.stream(
            {"messages": [HumanMessage(content=user)]},
            config=CONFIG,
            stream_mode="messages"
        ):
            if isinstance(msg, AIMessage) and msg.content:
                print(msg.content, end="", flush=True)
        print()

        snapshot = app.get_state(CONFIG)

        if snapshot.next and snapshot.tasks:
            for task in snapshot.tasks:
                if task.interrupts:
                    payload = task.interrupts[0].value
                    print("\n⚠ HUMAN APPROVAL REQUIRED ⚠\n")
                    print(payload["preview_email"])
                    decision = input("\nApprove escalation? (yes/no): ")

                    print("\nResuming...\n")
                    for msg, _ in app.stream(
                        Command(resume=decision),
                        config=CONFIG,
                        stream_mode="messages"
                    ):
                        if msg.content:
                            print(msg.content, end="", flush=True)
                    print()

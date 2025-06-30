from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable

from .agent_state import AgentState
from .nodes import processing_node, recommendation_node

def build_graph(llm_client: ChatOpenAI) -> Runnable:
    """
    Builds and compiles the LangGraph agent.

    This function defines the workflow, connecting the nodes in the correct
    sequence to create a runnable agent application.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("processor", processing_node)
    workflow.add_node("recommender", lambda state: recommendation_node(state, llm_client))
    
    workflow.set_entry_point("processor")
    workflow.add_edge("processor", "recommender")
    workflow.add_edge("recommender", END)
    
    print("✅ Agent graph built successfully.")
    return workflow.compile()

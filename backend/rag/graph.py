from langgraph.graph import StateGraph, END
from rag.state import RAGState
from langgraph.checkpoint.memory import MemorySaver
from rag.nodes import router_node, rag_node

def build_rag_graph():
    graph = StateGraph(RAGState) #This creates a new graph and tells langgraph what state looks like 

    ## Add Nodes   
    graph.add_node("router", router_node) #Registers router_node under the name "router"
    graph.add_node("rag", rag_node) #Registers rag_node under the name "rag"

    # Define Flow 
    graph.set_entry_point("router") #Setting the entry point of the router node
    graph.add_edge("router", "rag") #Creating eddges between the router and rag node 
    graph.add_edge("rag", END) # And then ending the rag node 

    return graph.compile(checkpointer=MemorySaver())

rag_graph = build_rag_graph()    





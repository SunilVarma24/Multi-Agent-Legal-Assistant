# main.py
from IPython.display import Image, display, Markdown
from langchain_core.messages import HumanMessage
from src.graph import get_agent_workflow

def main():
    # Visualize the graph
    agent_workflow = get_agent_workflow()
    display(Image(agent_workflow.get_graph().draw_mermaid_png()))
    
    # Test the Multi-Agent Legal Assistant
    test_prompt = "What are the steps involved in filing a lawsuit against government authority in India?"
    initial_state = {
        "messages": [HumanMessage(content=test_prompt)]
    }
    response = agent_workflow.invoke(initial_state)
    
    # Display the final output (summary)
    final_summary = response["messages"][-1].content.replace("SUMMARY COMPLETE: ", "", 1)
    from IPython.display import display, Markdown
    display(Markdown(final_summary))

if __name__ == "__main__":
    main()

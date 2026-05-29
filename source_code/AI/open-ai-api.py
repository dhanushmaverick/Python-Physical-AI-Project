""" 
    This should be the new workflow : 
    read the roboDK context from simulation 
    read the world coordinates of the blocks from the vision system
    pass the list of commands that can be used 
    pass through previously generated code 
    generate a new motion plan based on the current context and the user query it should be of the format:
        def run_motion(ctx):
        ...

"""




# Run Command: python -m source_code.AI.open-ai-api

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from source_code.AI.tools import get_world_coords_string, read_roboDK_function_list ,save_to_py
from langgraph.checkpoint.memory import InMemorySaver  

load_dotenv() # loads the API key

# OUTPUT TEMPLATE Definition
class ResearchResponse(BaseModel): # defines how output must be
    # topic: str # type hinting in python
    # summary: str
    # sources: list[str]
    # tools_used: list[str]
    code_generation_status: str
    tokens_used: str

llm = ChatOpenAI(model="gpt-5.4-nano") 



# CREATING AN AGENT
tools=[get_world_coords_string, read_roboDK_function_list ,save_to_py]
agent = create_agent(
    model = llm,
    system_prompt = """
    You are an expert Programmer and your role is to take in the following world 
    coordinates of 3 Blocks(red, blue, green) passed in with every query and read the 
    robodk python functions in order to generate a working python-robodk script for an
    existing workspace in order to stack them in the order specified by the user.
    The stacking will be done by a ur5e Robotic Arm in Robodk with a pre setup workspace.

    Return as code within the file source_code/AI/main_motion_plan.py and nothing else.
    """,

    tools=tools,
    checkpointer=InMemorySaver(),
)

while True:
    query = input("What's your query:  (exit/q to exit) ")
    if query.lower() == "exit" or query.lower() == "q":
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query, 
                },
            ],
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        } 
    
    )
    assistant_message = response["messages"][-1].content
    usage = response["messages"][-1].usage_metadata

    print("AI:", assistant_message)
    print("Input Tokens:", usage["input_tokens"])
    print("Output Tokens:", usage["output_tokens"])
    print("Total Tokens:", usage["total_tokens"])
# random
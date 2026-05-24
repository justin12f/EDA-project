from agents.context_creator import ContextCreatorAgent
import polars as pl 
from model_tools.create_data_context import CreateContext
from model_tools.create_data_context import AnalyzeContextPolars

def main():

    data = "shopping_trends.csv"
    
    agent = ContextCreatorAgent(data)

    context = agent.create_context()

    print(context)

if __name__ == "__main__":
    main()
    
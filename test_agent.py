from agents.context_creator import ContextCreatorAgent
import sys

def main():
    file_path = "GBM - Acciones.csv"
    print(f"Testing ContextCreatorAgent with {file_path}...")
    try:
        agent = ContextCreatorAgent(file_path)
        context = agent.create_context()
        print("Agent execution successful!")
        print("Context output:")
        print(context)
    except Exception as e:
        print(f"Agent execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

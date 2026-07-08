from agents.context_creator import ContextCreatorAgent
import polars as pl
from model_tools.create_data_context import CreateContext
from model_tools.create_data_context import AnalyzeContextPolars


def main():
    data_path = "shopping_trends.csv"
    print(f"--- Iniciando Análisis de Contexto sobre {data_path} ---")

    agent = ContextCreatorAgent(data_path)
    print("\nGenerando contexto estadístico local...")
    # Generate statistics context
    try:
        # Generate statistics context
        result = agent.create_context()
        print("\n--- Resultado Final ---")
        print(result)
    except Exception as e:
        print(f"\n--- ERROR CRÍTICO ---")
        print(f"El agente falló con el siguiente error: {e}")


if __name__ == "__main__":
    main()

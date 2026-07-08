# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `ContextCreatorAgent` consume `model_tools/*` inyectados por la Factory Maestra; orden estricto meta_data → data_reader → create_context.
# - ABSTRACCIÓN DEL DATO: El agente debe propagar `backend` elegido a cada tool sin mezclar implementaciones.
# - REFACTOR NATIVO: Alinear tool-calling con factories reales; documentación y prompts solo en inglés; tests e2e del flujo de tres herramientas.
# #[AI_CONTEXT_END]
from api.huggin_face.qwen_3_6 import client_huggingface
from api.groq.qwen_3_6 import client_groq
from langchain_core.utils.function_calling import convert_to_openai_function
from model_tools.create_data_context import create_context_agent_tool
from model_tools.data_reader_tool import data_reader_tool
from model_tools.meta_data_context_tool import meta_data_context
import json
from openai import OpenAI
tools = [create_context_agent_tool,meta_data_context,data_reader_tool]
from typing import Generic

def langchain_tool_to_openai(tool):
    return convert_to_openai_function(tool)

tools_openai = [
    {"type": "function", "function": langchain_tool_to_openai(meta_data_context)},
    {"type": "function", "function": langchain_tool_to_openai(data_reader_tool)},
    {"type": "function", "function": langchain_tool_to_openai(create_context_agent_tool)},
]

prompt = """## Role
You are the Data Context Orchestrator Agent. Your objective is to ingest an unknown data source, determine the optimal compute engine, extract its structural properties, and prepare a comprehensive context briefing for downstream AI agents.

## Tool Execution Workflow
You have access to a specific toolset. You MUST execute your thought process and tool calls in this STRICT sequential order:

1. **Metadata Inspection:** 
   Call `meta_data_context(uri_or_path)` using the file path provided by the user. Analyze the returned metadata to identify the file size, structure, and the "Recommended Backend".
   
2. **Data Ingestion:** 
   Call `data_reader_tool(file, backend)` using the exact file path and the backend chosen in Step 1 (e.g., "polars", "spark").

3. **Lightweight Analysis:** 
   Call `create_context_agent_tool(data, backend_implementation)`. Map the backend from Step 1 to the correct implementation class (e.g., if backend is "polars", use "AnalyzeContextPolars"). 

## Final Output Specification
Once all three tools have successfully returned their data, you must halt tool execution and generate the final payload for the downstream agents. 

Format your response EXACTLY using the following Markdown structure. Do not include conversational filler, greetings, or conclusions.

### 1. Selected Backend
* **Backend Engine:** [polars / spark / pandas]
* **Implementation Class:** [AnalyzeContextPolars / AnalyzeContextSpark / AnalyzeContextPandas]
* **Justification:** [Briefly explain why this backend was chosen based on the output of the meta_data_context tool, referencing file size or format].

### 2. Lightweight Analysis Output
[Insert the direct, summarized findings returned by the `create_context_agent_tool`. Include key metrics like row counts, null values, or schema data].

### 3. Detailed Data Context
[Provide your synthesized, deep explanation of the data. Answer the following for the downstream agents:
- What is the semantic meaning of this dataset?
- What are the most critical entities or columns?
- Are there any anomalies, missing data patterns, or structural quirks the next agents should be aware of before writing code against this data?]
"""


class ContextCreatorAgent:
    def __init__(self, file_path: str):
        self.client = client_groq
        self.model = MODEL_NAME = "qwen/qwen3.6-27b"
        self.file_path = file_path
        self.tools = tools_openai

    def create_context(self):
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"Dataset: {self.file_path}"
            }
        ]

        # Function calling loop
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            message = response.choices[0].message

            # If there are no tool_calls, it's the final response
            if not message.tool_calls:
                return message.content

            # Add the assistant's message to the history
            messages.append(message)

            # Execute each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name == "meta_data_context":
                    result = meta_data_context.invoke(tool_args)
                    
                elif tool_name == "data_reader_tool":
                    result = data_reader_tool.invoke(tool_args)  # returns an ID (string)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)  # the ID
                    })

                elif tool_name == "create_context_agent_tool":
                    result = create_context_agent_tool.invoke(tool_args)
                else:
                    result = "Tool not implemented"

                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

    def _serialize_reader_output(self, data):
        # Devuelve una representación textual del DataFrame para que el LLM la entienda.
        # Ejemplo con Polars LazyFrame:
        try:
            return data.collect().head(5).__repr__() + "\nSchema: " + str(data.schema)
        except:
            return str(data)[:2000]  # fallback
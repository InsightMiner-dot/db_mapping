import pandas as pd
import sqlite3
import os
import datetime
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from the .env file securely
load_dotenv()

# ==========================================
# 1. DEFINE THE GRAPH STATE
# ==========================================
class VarianceState(TypedDict):
    dimensions: dict       
    metrics: dict          
    history: str           
    draft_comment: str     
    final_comment: str     

# ==========================================
# 2. DEFINE THE GRAPH NODES
# ==========================================
def retrieve_history_node(state: VarianceState) -> dict:
    """Node 1: Dynamically queries SQLite based on the dimensions."""
    db_path = "master_historical_db.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        where_clauses = " AND ".join([f"{col} = ?" for col in state["dimensions"].keys()])
        values = tuple(state["dimensions"].values())
        query = f"SELECT historical_comment FROM historical_variances WHERE {where_clauses} ORDER BY date DESC LIMIT 3"
        
        cursor.execute(query, values)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {"history": "No specific historical data found for this exact combination."}
            
        context = "\n".join([f"{i+1}. Past Explanation: {row[0]}" for i, row in enumerate(results)])
        return {"history": context}
        
    except sqlite3.OperationalError:
        conn.close()
        return {"history": "Database initialized, but no historical data exists yet."}

def generate_draft_node(state: VarianceState) -> dict:
    """Node 2: Passes the data to Azure OpenAI to generate a draft."""
    
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.2
    )
    
    prompt = PromptTemplate.from_template("""
        You are a Senior FP&A Analyst. Explain the current financial variance concisely.
        
        Current Financial Data:
        Dimensions: {dimensions}
        Metrics: {metrics}
        
        Historical Context:
        {history}
        
        Draft a 2-3 sentence business explanation for this variance. Keep the tone professional.
    """)
    
    chain = prompt | llm
    
    dim_str = " | ".join([f"{k}: {v}" for k, v in state["dimensions"].items()])
    met_str = " | ".join([f"{k}: {v:,.2f}" for k, v in state["metrics"].items()])
    
    response = chain.invoke({
        "dimensions": dim_str,
        "metrics": met_str,
        "history": state["history"]
    })
    
    return {"draft_comment": response.content.strip()}

def human_approval_node(state: VarianceState) -> dict:
    """Node 3: Dummy node to interrupt the graph for human input."""
    return {}

def save_memory_node(state: VarianceState) -> dict:
    """Node 4: Saves the finalized comment back to SQLite."""
    if not state.get("final_comment"):
        return {}

    db_path = "master_historical_db.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    insert_data = state["dimensions"].copy()
    insert_data["historical_comment"] = state["final_comment"]
    insert_data["date"] = datetime.datetime.now().strftime("%Y-%m-%d")
    
    columns_str = ", ".join(insert_data.keys())
    placeholders = ", ".join(["?"] * len(insert_data))
    values = tuple(insert_data.values())
    
    try:
        create_table_sql = f"CREATE TABLE IF NOT EXISTS historical_variances ({', '.join([f'{col} TEXT' for col in insert_data.keys()])})"
        cursor.execute(create_table_sql)
        
        query = f"INSERT INTO historical_variances ({columns_str}) VALUES ({placeholders})"
        cursor.execute(query, values)
        conn.commit()
        print(" -> [Memory Saved to Database]")
    except Exception as e:
        print(f" -> [Failed to save memory: {e}]")
    finally:
        conn.close()
        
    return {}

# ==========================================
# 3. COMPILE THE LANGGRAPH WORKFLOW
# ==========================================
workflow = StateGraph(VarianceState)

workflow.add_node("retrieve", retrieve_history_node)
workflow.add_node("draft", generate_draft_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("save_memory", save_memory_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "draft")
workflow.add_edge("draft", "human_approval")
workflow.add_edge("human_approval", "save_memory") 
workflow.add_edge("save_memory", END)              

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
def process_variances(input_excel, output_excel):
    print(f"Loading {input_excel}...\n")
    try:
        df = pd.read_excel(input_excel)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_excel}'. Please check your file path.")
        return

    if 'Final_Comment' not in df.columns:
        df['Final_Comment'] = ""

    dimension_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if 'Final_Comment' in dimension_cols: dimension_cols.remove('Final_Comment')
    metric_cols = df.select_dtypes(include=['number']).columns.tolist()

    print("="*60)
    print("STARTING AZURE AI AGENT & HUMAN REVIEW")
    print("="*60)

    for index, row in df.iterrows():
        row_dimensions = {col: row[col] for col in dimension_cols}
        row_metrics = {col: row[col] for col in metric_cols}
        
        print(f"\nProcessing Row {index + 1} -> {row_dimensions}")
        
        initial_state = {
            "dimensions": row_dimensions,
            "metrics": row_metrics,
            "history": "",
            "draft_comment": "",
            "final_comment": ""
        }
        config = {"configurable": {"thread_id": f"row_{index}"}}

        # Run the graph until the human approval breakpoint
        for event in app.stream(initial_state, config):
            pass 
        
        current_state = app.get_state(config).values
        ai_draft = current_state.get("draft_comment", "Error: No draft generated.")
        
        print(f"Historical Context Found:\n{current_state.get('history', 'None').strip()}")
        print("\n--- AI DRAFT ---")
        print(ai_draft)
        print("----------------")
        
        print("\nPress [ENTER] to approve, or type your edited comment below:")
        user_input = input("> ")
        
        final_comment = ai_draft if user_input.strip() == "" else user_input.strip()
        print("Status: APPROVED" if user_input.strip() == "" else "Status: OVERRIDDEN")
        
        # Update state and resume the graph to trigger the save_memory node
        app.update_state(config, {"final_comment": final_comment})
        for event in app.stream(None, config):
            pass
            
        df.at[index, 'Final_Comment'] = final_comment

    print("\n" + "="*60)
    print(f"Review complete. Exporting final data to {output_excel}...")
    try:
        df.to_excel(output_excel, index=False)
        print("Export successful!")
    except Exception as e:
        print(f"Error exporting file: {e}. (Make sure the Excel file isn't open in another program).")

# ==========================================
# 5. HARDCODED EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Provide the exact path to your Tagetik export here
    my_input_file = "C:/Users/YourName/Downloads/may_tagetik_export.xlsx"
    
    # Provide the path where you want the final AI report saved
    my_output_file = "C:/Users/YourName/Documents/may_final_variances.xlsx"
    
    process_variances(input_excel=my_input_file, output_excel=my_output_file)

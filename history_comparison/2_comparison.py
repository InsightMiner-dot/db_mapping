import pandas as pd
import sqlite3
import os
import json
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables securely
load_dotenv()

# ==========================================
# 1. DEFINE THE GRAPH STATE
# ==========================================
class VarianceState(TypedDict):
    row_data: dict         # All columns from the current Excel row
    history: str           # Retrieved historical context
    draft_comment: str     # AI generated Comment
    draft_reason: str      # AI generated Reason_for_variance
    final_comment: str     # Human approved Comment
    final_reason: str      # Human approved Reason

# ==========================================
# 2. DEFINE THE GRAPH NODES
# ==========================================
def retrieve_history_node(state: VarianceState) -> dict:
    """Queries SQLite using all structural columns and Scenarios to build a trend history."""
    db_path = "master_historical_db.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # EXACT columns requested for the perfect historical match
    standard_columns = [
        "Scenario-A",
        "Scenario-B",
        "Region", 
        "Market", 
        "OH_LC", 
        "Division_Desc", 
        "Function_desc",      
        "Department_descc",   
        "Entity_desc",        
        "CostCat description"
    ]
    
    match_keys = []
    values = []
    
    # Extract only these standard columns from the current row
    for col in standard_columns:
        if col in state["row_data"]:
            match_keys.append(col)
            values.append(state["row_data"][col])
            
    try:
        # Check if table exists
        cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='historical_variances'")
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {"history": "Master database initialized, but no historical data exists yet."}

        # Abort if the file is missing the structural columns
        if not match_keys:
             return {"history": "No standard structural dimensions found in this row."}

        # Safely wrap column names in double quotes to handle spaces/hyphens
        where_clauses = " AND ".join([f'"{col}" = ?' for col in match_keys])
        
        # Pull up to the last 6 months of data to establish a solid trend
        query = f"""
            SELECT Year, Month, "Variancce Amount ", Comments, Reason_for_variance 
            FROM historical_variances 
            WHERE {where_clauses} 
            ORDER BY Year DESC, Month DESC LIMIT 6
        """
        
        cursor.execute(query, tuple(values))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {"history": "No historical data found for this exact scenario and line item."}
            
        # Format the history so the AI can easily read the trend
        history_lines = ["--- HISTORICAL TREND ---"]
        for row in results:
            history_lines.append(
                f"- {row[1]} {row[0]}: Variance = {row[2]} | Comment: '{row[3]}' | Reason: '{row[4]}'"
            )
        
        return {"history": "\n".join(history_lines)}
        
    except sqlite3.OperationalError as e:
        conn.close()
        return {"history": f"Database lookup failed: {e}"}

def generate_draft_node(state: VarianceState) -> dict:
    """Passes the data to Azure OpenAI to analyze the trend and generate drafts."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    prompt = PromptTemplate.from_template("""
        You are a Senior FP&A Analyst. 
        
        CURRENT MONTH DATA:
        Year/Month: {year}-{month}
        Variance Amount: {current_variance}
        Line Item Details: {details}
        
        HISTORICAL VARIANCES FOR THIS EXACT LINE ITEM:
        {history}
        
        TASK:
        1. Compare the current variance to the historical trend.
        2. Generate a "Comments" field (1 sentence). If there is history, explicitly mention the previous month/year trend (e.g., "Continuing the trend from November 2025, this variance is driven by...").
        3. Generate a "Reason_for_variance" field explaining the likely business driver based on the historical context.
        
        Respond ONLY with a valid JSON object using exactly these keys: 
        "Comments", "Reason_for_variance"
    """)
    
    chain = prompt | llm
    
    details = {
        "Region": state["row_data"].get("Region", "Not Provided"),
        "Department": state["row_data"].get("Department_descc", "Not Provided"),
        "Cost Category": state["row_data"].get("CostCat description", "Not Provided")
    }
    
    response = chain.invoke({
        "year": state["row_data"].get("Year", "Unknown"),
        "month": state["row_data"].get("Month", "Unknown"),
        "current_variance": state["row_data"].get("Variancce Amount ", 0),
        "details": json.dumps(details),
        "history": state["history"]
    })
    
    try:
        ai_output = json.loads(response.content)
        return {
            "draft_comment": ai_output.get("Comments", "N/A"),
            "draft_reason": ai_output.get("Reason_for_variance", "N/A")
        }
    except json.JSONDecodeError:
        return {
            "draft_comment": "Error parsing JSON.",
            "draft_reason": "Raw Response: " + response.content 
        }

def human_approval_node(state: VarianceState) -> dict:
    """Dummy node to interrupt the graph for human input."""
    return {}

def save_memory_node(state: VarianceState) -> dict:
    """Saves the approved output into the DB under the standard names for next month."""
    if not state.get("final_comment") and not state.get("final_reason"):
        return {}

    db_path = "master_historical_db.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    insert_data = state["row_data"].copy()
    
    # Map the finalized output to the standard DB column names
    insert_data["Comments"] = state["final_comment"]
    insert_data["Reason_for_variance"] = state["final_reason"]
    
    # Prevent the new Excel-only columns from polluting the SQL database
    insert_data.pop("New_AI_Comments", None)
    insert_data.pop("New_AI_Reason", None)
    
    columns_str = ", ".join([f'"{col}"' for col in insert_data.keys()])
    placeholders = ", ".join(["?"] * len(insert_data))
    values = tuple(insert_data.values())
    
    try:
        schema_defs = ", ".join([f'"{col}" TEXT' for col in insert_data.keys()])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS historical_variances ({schema_defs})"
        cursor.execute(create_table_sql)
        
        query = f"INSERT INTO historical_variances ({columns_str}) VALUES ({placeholders})"
        cursor.execute(query, values)
        conn.commit()
        print(" -> [Master Database Updated]")
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
        print(f"Error: Could not find '{input_excel}'.")
        return

    # CREATE NEW COLUMNS SO WE DON'T OVERWRITE ORIGINAL ONES
    if 'New_AI_Comments' not in df.columns:
        df['New_AI_Comments'] = ""
    if 'New_AI_Reason' not in df.columns:
        df['New_AI_Reason'] = ""

    print("="*60)
    print("STARTING AI REVIEW & HUMAN OVERSIGHT")
    print("="*60)

    for index, row in df.iterrows():
        row_data = row.fillna("").to_dict()
        
        region = row_data.get('Region', 'Unknown Region')
        dept = row_data.get('Department_descc', 'Unknown Dept')
        cost_cat = row_data.get('CostCat description', 'Unknown Category')
        variance_amt = row_data.get('Variancce Amount ', 0) 
        
        print(f"\n[{index + 1}/{len(df)}] Analyzing: {region} | {dept} | {cost_cat}")
        print(f"Variance Amount: {variance_amt}")
        
        initial_state = {
            "row_data": row_data,
            "history": "",
            "draft_comment": "",
            "draft_reason": "",
            "final_comment": "",
            "final_reason": ""
        }
        
        config = {"configurable": {"thread_id": f"batch_run_row_{index}"}}

        # 1. Run until human approval
        for event in app.stream(initial_state, config):
            pass 
        
        current_state = app.get_state(config).values
        
        print(f"\n{current_state.get('history', 'None').strip()}")
        
        # 2. Human Review Process
        print("\n--- AI DRAFTS ---")
        print(f"Comment : {current_state.get('draft_comment')}")
        print(f"Reason  : {current_state.get('draft_reason')}")
        
        print("\n[Press ENTER to approve the draft, or type your edit]")
        
        user_comment = input("Edit Comment > ")
        final_comment = current_state['draft_comment'] if user_comment.strip() == "" else user_comment.strip()
        
        user_reason = input("Edit Reason  > ")
        final_reason = current_state['draft_reason'] if user_reason.strip() == "" else user_reason.strip()
        
        # 3. Update state, resume graph, and save to DB
        app.update_state(config, {"final_comment": final_comment, "final_reason": final_reason})
        for event in app.stream(None, config):
            pass
            
        # 4. WRITE THE APPROVED OUTPUT TO THE BRAND NEW COLUMNS
        df.at[index, 'New_AI_Comments'] = final_comment
        df.at[index, 'New_AI_Reason'] = final_reason

    # Final Export
    print("\n" + "="*60)
    print(f"Review complete. Exporting final data to {output_excel}...")
    try:
        df.to_excel(output_excel, index=False)
        print("Export successful!")
    except Exception as e:
        print(f"Error exporting file: {e}. (Ensure the file isn't open).")

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure your one-time CSV database is loaded before running!
    my_input_file = "current_month_variances.xlsx"
    my_output_file = "final_reviewed_variances.xlsx"
    
    process_variances(input_excel=my_input_file, output_excel=my_output_file)

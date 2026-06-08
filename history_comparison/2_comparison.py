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

# Load environment variables
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
    """Dynamically queries SQLite using whatever dimension columns exist in the data."""
    db_path = "master_historical_db.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # The ideal list of dimensions we want to match on
    ideal_keys = [
        "Region", "Market", "OH_LC", "Division_Desc", 
        "Function_Desc", "Department_Desc", "Entity_Desc", "CostCat description"
    ]
    
    # DYNAMIC MATCH: Only use keys that actually exist in the current Excel row AND have a value
    match_keys = [
        key for key in ideal_keys 
        if key in state["row_data"] and state["row_data"][key] != ""
    ]
    
    try:
        # Check if table exists
        cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='historical_variances'")
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {"history": "Master database initialized, but no historical data exists yet."}

        # Build query dynamically based ONLY on the available columns
        if not match_keys:
             return {"history": "No valid structural dimensions found in this row to query history."}

        # Safely wrap column names in double quotes to handle spaces
        where_clauses = " AND ".join([f'"{col}" = ?' for col in match_keys])
        values = tuple(state["row_data"][col] for col in match_keys)
        
        query = f"""
            SELECT Year, Month, "variance Amount", Comments, Reason_for_variance 
            FROM historical_variances 
            WHERE {where_clauses} 
            ORDER BY Year DESC, Month DESC LIMIT 3
        """
        
        cursor.execute(query, values)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {"history": "No historical data found for this specific combination."}
            
        history_lines = [
            f"- {row[0]}-{row[1]} | Variance: {row[2]:,.2f} | Comment: {row[3]} | Reason: {row[4]}"
            for row in results
        ]
        return {"history": "\n".join(history_lines)}
        
    except sqlite3.OperationalError as e:
        conn.close()
        return {"history": f"Database lookup failed: {e}"}

def generate_draft_node(state: VarianceState) -> dict:
    """Passes the data to Azure OpenAI to compare current vs. previous variance."""
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
        
        HISTORICAL VARIANCES FOR THIS LINE ITEM:
        {history}
        
        TASK:
        1. Compare the current variance amount to the historical variance amounts.
        2. Generate a concise "Comments" summarizing the trend (1 sentence).
        3. Generate a "Reason_for_variance" explaining the likely business driver based on historical context (1-2 sentences).
        
        Respond ONLY with a valid JSON object using exactly these keys: 
        "Comments", "Reason_for_variance"
    """)
    
    chain = prompt | llm
    
    # Use .get() to prevent KeyErrors. If the column doesn't exist, it defaults to "Not Provided"
    details = {
        "Region": state["row_data"].get("Region", "Not Provided"),
        "Division": state["row_data"].get("Division_Desc", state["row_data"].get("Division", "Not Provided")),
        "Department": state["row_data"].get("Department_Desc", state["row_data"].get("Department", "Not Provided")),
        "Cost Category": state["row_data"].get("CostCat description", state["row_data"].get("CostCat", "Not Provided"))
    }
    
    # Dump the remaining non-empty string columns to give the LLM maximum context
    extra_context = {k: v for k, v in state["row_data"].items() if isinstance(v, str) and v != "" and k not in details}
    details["Additional_Context"] = extra_context
    
    response = chain.invoke({
        "year": state["row_data"].get("Year", "Unknown"),
        "month": state["row_data"].get("Month", "Unknown"),
        "current_variance": state["row_data"].get("variance Amount", 0),
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
    """Saves the finalized comment, reason, and full row data back to SQLite."""
    if not state.get("final_comment") and not state.get("final_reason"):
        return {}

    db_path = "master_historical_db.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Prepare the data to insert based on the exact required schema
    insert_data = state["row_data"].copy()
    insert_data["Comments"] = state["final_comment"]
    insert_data["Reason_for_variance"] = state["final_reason"]
    
    # Wrap column names in double quotes to handle spaces/hyphens dynamically
    columns_str = ", ".join([f'"{col}"' for col in insert_data.keys()])
    placeholders = ", ".join(["?"] * len(insert_data))
    values = tuple(insert_data.values())
    
    try:
        # Create table dynamically based on the current row's structure
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

    # Ensure output columns exist
    if 'Comments' not in df.columns:
        df['Comments'] = ""
    if 'Reason_for_variance' not in df.columns:
        df['Reason_for_variance'] = ""

    print("="*60)
    print("STARTING AI REVIEW & HUMAN OVERSIGHT")
    print("="*60)

    for index, row in df.iterrows():
        # Convert row to dict, replacing NaNs with empty strings for JSON/SQL compatibility
        row_data = row.fillna("").to_dict()
        
        # Safely get descriptive names for the console printout
        region = row_data.get('Region', 'Unknown Region')
        dept = row_data.get('Department_Desc', row_data.get('Department', 'Unknown Dept'))
        cost_cat = row_data.get('CostCat description', row_data.get('CostCat', 'Unknown Category'))
        variance_amt = row_data.get('variance Amount', 0)
        
        print(f"\n[{index + 1}/{len(df)}] Analyzing: {region} | {dept} | {cost_cat}")
        print(f"Variance Amount: {variance_amt:,.2f}")
        
        initial_state = {
            "row_data": row_data,
            "history": "",
            "draft_comment": "",
            "draft_reason": "",
            "final_comment": "",
            "final_reason": ""
        }
        # Unique thread ID for each run to prevent state overlap
        config = {"configurable": {"thread_id": f"batch_run_row_{index}"}}

        # 1. Run until human approval
        for event in app.stream(initial_state, config):
            pass 
        
        current_state = app.get_state(config).values
        
        print(f"\n--- HISTORICAL CONTEXT ---\n{current_state.get('history', 'None').strip()}")
        
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
            
        # 4. Update the Excel DataFrame
        df.at[index, 'Comments'] = final_comment
        df.at[index, 'Reason_for_variance'] = final_reason

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
    my_input_file = "current_month_variances.xlsx"
    my_output_file = "final_reviewed_variances.xlsx"
    
    process_variances(input_excel=my_input_file, output_excel=my_output_file)

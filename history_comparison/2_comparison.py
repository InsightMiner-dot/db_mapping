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
# 1. HELPER LOGIC FOR DATES
# ==========================================
def get_yyyymm(year, month_str):
    """Converts a Year and Month string into a sortable integer (e.g., 2025 'December' -> 202512)."""
    months = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
        'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
        'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    try:
        m = months[str(month_str).strip().lower()]
        return int(year) * 100 + m
    except (KeyError, ValueError):
        return 0  # Fallback if date is missing or invalid

# ==========================================
# 2. DEFINE THE GRAPH STATE
# ==========================================
class VarianceState(TypedDict):
    row_data: dict         
    history: str           
    recent_historical_comment: str  
    recent_historical_reason: str   
    draft_comment: str     
    draft_reason: str      
    final_comment: str     
    final_reason: str      

# ==========================================
# 3. DEFINE THE GRAPH NODES
# ==========================================
def retrieve_history_node(state: VarianceState) -> dict:
    """Queries SQLite and filters chronologically for PREVIOUS months only."""
    db_path = "master_historical_db.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Excluded Scenario and Dates so we can look back into previous timelines.
    standard_columns = [
        "Region", 
        "Market", 
        "OH_LC", 
        "Division_Desc", 
        "Function_desc",      
        "Department_desc",    # Exact spelling matched 
        "Entity_desc",        
        "CostCat description"
    ]
    
    match_keys = []
    values = []
    
    for col in standard_columns:
        if col in state["row_data"]:
            match_keys.append(col)
            val = str(state["row_data"][col]).strip() 
            values.append(val)
            
    try:
        cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='historical_variances'")
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {"history": "Master database initialized, but no historical data exists yet.", "recent_historical_comment": "N/A", "recent_historical_reason": "N/A"}

        if not match_keys:
             return {"history": "No standard structural dimensions found in this row.", "recent_historical_comment": "N/A", "recent_historical_reason": "N/A"}

        where_clauses = " AND ".join([f'"{col}" = ?' for col in match_keys])
        
        # We pull everything matching the structure, chronological filtering happens in Python
        query = f"""
            SELECT Year, Month, "Variancce Amount ", Comments, Reason_for_variance 
            FROM historical_variances 
            WHERE {where_clauses} 
        """

        cursor.execute(query, tuple(values))
        results = cursor.fetchall()
        conn.close()
        
        # --- CHRONOLOGICAL FILTERING: strictly look for PREVIOUS months ---
        curr_date_num = get_yyyymm(state["row_data"].get("Year"), state["row_data"].get("Month"))
        valid_history = []
        
        for row in results:
            h_year, h_month, h_var, h_com, h_rea = row
            h_date_num = get_yyyymm(h_year, h_month)
            
            # ONLY keep records that are chronologically BEFORE the current input month
            if h_date_num > 0 and h_date_num < curr_date_num:
                valid_history.append({
                    'date_num': h_date_num, 'year': h_year, 'month': h_month, 
                    'var': h_var, 'com': h_com, 'rea': h_rea
                })
        
        if not valid_history:
            return {
                "history": "No valid historical trend found strictly prior to this month.",
                "recent_historical_comment": "No History Found",
                "recent_historical_reason": "No History Found"
            }
            
        # Sort chronologically (Newest to Oldest) and take the top 6 for the trend
        valid_history.sort(key=lambda x: x['date_num'], reverse=True)
        valid_history = valid_history[:6]
        
        # Extract the single most recent comment/reason for the Excel Output
        most_recent_comment = valid_history[0]['com']
        most_recent_reason = valid_history[0]['rea']
            
        # Format the history string for the AI Agent
        history_lines = ["--- HISTORICAL TREND ---"]
        for item in valid_history:
            history_lines.append(
                f"- {item['month']} {item['year']}: Variance = {item['var']} | Comment: '{item['com']}' | Reason: '{item['rea']}'"
            )
        
        return {
            "history": "\n".join(history_lines),
            "recent_historical_comment": most_recent_comment,
            "recent_historical_reason": most_recent_reason
        }
        
    except sqlite3.OperationalError as e:
        conn.close()
        return {"history": f"Database lookup failed: {e}", "recent_historical_comment": "Error", "recent_historical_reason": "Error"}

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
        "Department": state["row_data"].get("Department_desc", "Not Provided"),
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
        return {"draft_comment": "Error parsing JSON.", "draft_reason": "Raw Response: " + response.content}

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
    
    insert_data["Comments"] = state["final_comment"]
    insert_data["Reason_for_variance"] = state["final_reason"]
    
    # Clean output-only Excel columns so they don't break the SQL database schema
    keys_to_remove = ["New_AI_Comments", "New_AI_Reason", "Historical_Comment", "Historical_Reason"]
    for k in keys_to_remove:
        insert_data.pop(k, None)
    
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
# 4. COMPILE THE LANGGRAPH WORKFLOW
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
# 5. EXECUTION LOOP
# ==========================================
def process_variances(input_excel, output_excel):
    print(f"Loading {input_excel}...\n")
    try:
        df = pd.read_excel(input_excel)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_excel}'.")
        return

    # Create new columns for the Excel Output
    for new_col in ['Historical_Comment', 'Historical_Reason', 'New_AI_Comments', 'New_AI_Reason']:
        if new_col not in df.columns:
            df[new_col] = ""

    print("="*60)
    print("STARTING AI REVIEW & HUMAN OVERSIGHT")
    print("="*60)

    for index, row in df.iterrows():
        row_data = row.fillna("").to_dict()
        
        region = row_data.get('Region', 'Unknown Region')
        dept = row_data.get('Department_desc', 'Unknown Dept')
        cost_cat = row_data.get('CostCat description', 'Unknown Category')
        variance_amt = row_data.get('Variancce Amount ', 0) 
        
        print(f"\n[{index + 1}/{len(df)}] Analyzing: {region} | {dept} | {cost_cat}")
        print(f"Variance Amount: {variance_amt}")
        
        initial_state = {
            "row_data": row_data,
            "history": "",
            "recent_historical_comment": "",
            "recent_historical_reason": "",
            "draft_comment": "",
            "draft_reason": "",
            "final_comment": "",
            "final_reason": ""
        }
        
        config = {"configurable": {"thread_id": f"batch_run_row_{index}"}}

        for event in app.stream(initial_state, config):
            pass 
        
        current_state = app.get_state(config).values
        
        print(f"\n{current_state.get('history', 'None').strip()}")
        
        print("\n--- AI DRAFTS ---")
        print(f"Comment : {current_state.get('draft_comment')}")
        print(f"Reason  : {current_state.get('draft_reason')}")
        
        print("\n[Press ENTER to approve the draft, or type your edit]")
        
        user_comment = input("Edit Comment > ")
        final_comment = current_state['draft_comment'] if user_comment.strip() == "" else user_comment.strip()
        
        user_reason = input("Edit Reason  > ")
        final_reason = current_state['draft_reason'] if user_reason.strip() == "" else user_reason.strip()
        
        app.update_state(config, {"final_comment": final_comment, "final_reason": final_reason})
        for event in app.stream(None, config):
            pass
            
        # Write everything to the Excel dataframe
        df.at[index, 'Historical_Comment'] = current_state.get('recent_historical_comment', 'N/A')
        df.at[index, 'Historical_Reason'] = current_state.get('recent_historical_reason', 'N/A')
        df.at[index, 'New_AI_Comments'] = final_comment
        df.at[index, 'New_AI_Reason'] = final_reason

    print("\n" + "="*60)
    print(f"Review complete. Exporting final data to {output_excel}...")
    try:
        df.to_excel(output_excel, index=False)
        print("Export successful!")
    except Exception as e:
        print(f"Error exporting file: {e}. (Ensure the file isn't open).")

# ==========================================
# 6. RUN
# ==========================================
if __name__ == "__main__":
    my_input_file = "current_month_variances.xlsx"
    my_output_file = "final_reviewed_variances.xlsx"
    
    process_variances(input_excel=my_input_file, output_excel=my_output_file)

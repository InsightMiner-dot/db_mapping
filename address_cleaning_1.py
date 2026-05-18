import streamlit as st
import pandas as pd
import io
import re
from rapidfuzz import process, fuzz

# --- Processing Functions ---

def clean_whitespace(df, columns):
    """Strips leading/trailing spaces and condenses internal multiple spaces."""
    cleaned_df = df.copy()
    for col in columns:
        if cleaned_df[col].dtype == 'object' or pd.api.types.is_string_dtype(cleaned_df[col]):
            mask = cleaned_df[col].notna()
            cleaned_df.loc[mask, col] = (
                cleaned_df.loc[mask, col]
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
            )
    return cleaned_df

def prep_for_matching(text):
    """Removes noise to improve fuzzy matching accuracy."""
    if pd.isna(text):
        return ""
    text = str(text).upper()
    text = re.sub(r'[^A-Z0-9,\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_column(df, columns, known_list, threshold=82):
    """Uses RapidFuzz to snap messy data to the clean master list."""
    normalized_df = df.copy()
    
    clean_known_list = [prep_for_matching(item) for item in known_list if item.strip()]
    master_mapping = {prep_for_matching(item): item for item in known_list if item.strip()}

    for col in columns:
        if normalized_df[col].dtype == 'object' or pd.api.types.is_string_dtype(normalized_df[col]):
            
            def match_logic(messy_val):
                cleaned_messy = prep_for_matching(messy_val)
                if not cleaned_messy:
                    return messy_val
                
                match = process.extractOne(cleaned_messy, clean_known_list, scorer=fuzz.WRatio)
                
                if match:
                    best_match_key, score, _ = match
                    if score >= threshold:
                        return master_mapping[best_match_key]
                        
                return messy_val 

            normalized_df[col] = normalized_df[col].apply(match_logic)
            
    return normalized_df

def generate_change_log(original_df, processed_df, columns):
    """Compares original and processed dataframes to log row-by-row changes."""
    log_records = []
    
    for col in columns:
        for idx in original_df.index:
            orig_val = original_df.at[idx, col]
            new_val = processed_df.at[idx, col]
            
            # Safely handle NaN comparisons
            if pd.isna(orig_val) and pd.isna(new_val):
                continue
                
            # If the value changed, log it
            if str(orig_val) != str(new_val):
                log_records.append({
                    'Excel Row': idx + 2, # +2 to match Excel row numbering (1-based + header)
                    'Column': col,
                    'Original Value': orig_val,
                    'Cleaned Value': new_val
                })
                
    return pd.DataFrame(log_records)

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Data Cleaner Pipeline", layout="wide")
    st.title("Excel Data Cleaner & Normalizer")
    
    # Sidebar for Master List Configuration
    with st.sidebar:
        st.header("Normalization Settings")
        st.markdown("Paste your master list of clean addresses here (one per line).")
        
        default_master = (
            "ExxonMobil Chemical, BAYWAY DRIVE, BAYTOWN, TX 77520\n"
            "GARCO, INC., ASHEBORO, NC 27203\n"
            "Dow Chemical Company, 123 MAIN ST, HOUSTON, TX 77002"
        )
        
        master_list_input = st.text_area("Master Records", value=default_master, height=300)
        match_threshold = st.slider("Match Confidence Threshold", 50, 100, 82)

    # 1. Main Upload UI
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # 2. Select Sheet
            xls = pd.ExcelFile(uploaded_file)
            selected_sheet = st.selectbox("Select Sheet", options=xls.sheet_names)
            
            # Keep a pristine copy of the original data for the change log
            original_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            
            st.subheader("Data Preview")
            st.dataframe(original_df.head(), use_container_width=True)

            # 3. Column Selection
            st.markdown("### 1. Select Columns to Clean")
            target_fields = ["Remit To", "Shipper", "Bill To", "Origin", "Destination", "Supplier Address"]
            default_cols = [col for col in target_fields if col in original_df.columns]
            
            selected_columns = st.multiselect(
                "Columns to target:",
                options=original_df.columns.tolist(),
                default=default_cols
            )

            # 4. Cleaning Checkboxes
            st.markdown("### 2. Select Cleaning Actions")
            col1, col2 = st.columns(2)
            with col1:
                do_whitespace = st.checkbox("🧹 Strip Extra Whitespaces", value=True)
            with col2:
                do_normalize = st.checkbox("🔗 Apply Fuzzy Normalization", value=False)

            # 5. Process Execution
            if st.button("Process Data", type="primary"):
                if not selected_columns:
                    st.warning("Please select at least one column.")
                    return
                if not do_whitespace and not do_normalize:
                    st.warning("Please select at least one cleaning action.")
                    return

                with st.spinner("Processing data..."):
                    # Create a working copy to apply changes to
                    processed_df = original_df.copy()
                    
                    # Apply actions
                    if do_whitespace:
                        processed_df = clean_whitespace(processed_df, selected_columns)
                        
                    if do_normalize:
                        master_list = master_list_input.split('\n')
                        processed_df = normalize_column(processed_df, selected_columns, master_list, threshold=match_threshold)
                        
                    # Generate the Change Log
                    change_log_df = generate_change_log(original_df, processed_df, selected_columns)

                st.success("Data processed successfully!")
                
                # Show results
                st.markdown("### Cleaned Data Preview")
                st.dataframe(processed_df.head(), use_container_width=True)

                if not change_log_df.empty:
                    st.markdown(f"### Change Log ({len(change_log_df)} edits made)")
                    st.dataframe(change_log_df.head(10), use_container_width=True)
                else:
                    st.info("No changes were needed. The data is already clean based on your settings.")

                # 6. Prepare Downloads
                st.markdown("### 3. Download Files")
                dl_col1, dl_col2 = st.columns(2)
                
                # Excel Download Buffer
                excel_output = io.BytesIO()
                with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
                    processed_df.to_excel(writer, index=False, sheet_name=selected_sheet[:31])
                
                with dl_col1:
                    st.download_button(
                        label="⬇️ Download Cleaned Excel",
                        data=excel_output.getvalue(),
                        file_name="cleaned_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # Log Download Buffer (as CSV for ease)
                if not change_log_df.empty:
                    csv_log = change_log_df.to_csv(index=False).encode('utf-8')
                    with dl_col2:
                        st.download_button(
                            label="⬇️ Download Change Log (CSV)",
                            data=csv_log,
                            file_name="cleaning_log.csv",
                            mime="text/csv"
                        )
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

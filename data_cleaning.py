import streamlit as st
import pandas as pd
import io

def clean_whitespace(df, columns):
    """
    Cleans leading/trailing whitespaces and condenses multiple internal 
    spaces into a single space for the specified columns.
    """
    cleaned_df = df.copy()
    for col in columns:
        # Ensure we only apply string methods to object/string columns
        if cleaned_df[col].dtype == 'object' or pd.api.types.is_string_dtype(cleaned_df[col]):
            # Fill NaNs with empty string temporarily to avoid converting them to the string "nan"
            # Alternatively, apply only to non-null rows
            mask = cleaned_df[col].notna()
            cleaned_df.loc[mask, col] = (
                cleaned_df.loc[mask, col]
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True) # Replaces \t, \n, and multiple spaces with one space
            )
    return cleaned_df

def main():
    st.set_page_config(page_title="Data Cleaner Pipeline", layout="wide")
    st.title("Data Cleaner: Whitespace Removal")
    st.markdown("Upload an Excel file to clean target columns of excessive whitespace.")

    # 1. File Upload
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # 2. Sheet Selection
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            
            selected_sheet = st.selectbox("Select Sheet", options=sheet_names)
            
            # Load the selected sheet into a DataFrame
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            
            st.subheader("Original Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # 3. Column Selection
            # Define your target fields to auto-select if they exist in the sheet
            target_fields = [
                "Remit To", "Shipper", "Bill To", 
                "Origin", "Destination", "Supplier Address"
            ]
            
            # Find the intersection to set as default values
            default_cols = [col for col in target_fields if col in df.columns]
            
            selected_columns = st.multiselect(
                "Select columns to clean",
                options=df.columns.tolist(),
                default=default_cols,
                help="By default, target routing and address fields are selected if found."
            )

            # 4. Trigger Cleaning
            if st.button("Clean Selected Columns", type="primary"):
                if not selected_columns:
                    st.warning("Please select at least one column to clean.")
                else:
                    with st.spinner("Processing data..."):
                        cleaned_df = clean_whitespace(df, selected_columns)
                        
                    st.success("Data cleaned successfully!")
                    
                    st.subheader("Cleaned Data Preview")
                    st.dataframe(cleaned_df.head(), use_container_width=True)

                    # 5. Download Processed File
                    # Write to a buffer so we don't save to the local disk
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        cleaned_df.to_excel(writer, index=False, sheet_name=selected_sheet[:31]) # Excel sheet names max 31 chars
                    
                    processed_data = output.getvalue()

                    st.download_button(
                        label="⬇️ Download Cleaned Excel",
                        data=processed_data,
                        file_name="cleaned_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()

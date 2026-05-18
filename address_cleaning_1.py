"""
Address Cleaner — Streamlit App
Handles: whitespace, comma spacing, concatenated street suffixes,
         punctuation noise, abbreviation normalization, fuzzy duplicate flagging.
"""

import re
import io
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Address Cleaner", layout="wide")

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_FIELDS = ["Remit To", "Shipper", "Bill To", "Origin", "Destination", "Supplier Address"]

# Long-form suffixes only (avoid splitting short abbreviations like ST, DR mid-word)
STREET_SUFFIXES_LONG = [
    "STREET", "AVENUE", "BOULEVARD", "DRIVE", "ROAD", "LANE", "COURT",
    "CIRCLE", "PLACE", "HIGHWAY", "PARKWAY", "TERRACE", "TRAIL", "SUITE", "FLOOR",
]

# Common abbreviation expansions (optional — applied only if user enables it)
ABBREV_MAP = {
    r"\bST\b":   "STREET",
    r"\bAVE\b":  "AVENUE",
    r"\bBLVD\b": "BOULEVARD",
    r"\bDR\b":   "DRIVE",
    r"\bRD\b":   "ROAD",
    r"\bLN\b":   "LANE",
    r"\bCT\b":   "COURT",
    r"\bCIR\b":  "CIRCLE",
    r"\bPL\b":   "PLACE",
    r"\bHWY\b":  "HIGHWAY",
    r"\bPKWY\b": "PARKWAY",
    r"\bTER\b":  "TERRACE",
    r"\bTRL\b":  "TRAIL",
    r"\bSTE\b":  "SUITE",
}


# ── Core cleaning functions ────────────────────────────────────────────────────

def fix_concatenated_suffix(text: str) -> str:
    """
    Split street suffixes that are glued to the next word.
    e.g.  DRIVEBAYTOWN  →  DRIVE BAYTOWN
          BLVDNORTH     →  BLVD NORTH
    Only fires when the suffix appears at the start of a token (after space or comma).
    """
    for suffix in sorted(STREET_SUFFIXES_LONG, key=len, reverse=True):
        text = re.sub(rf"(?<=[\s,])({suffix})([A-Z][a-zA-Z])", rf"\1 \2", text)
    return text


def normalize_address(raw: str, expand_abbrev: bool = False) -> str:
    """
    Full normalization pipeline for a single address string.

    Steps
    -----
    1. Uppercase + strip
    2. Strip trailing periods from abbreviations  INC. → INC
    3. Fix comma spacing                          GARCO,INC → GARCO, INC
    4. Collapse internal whitespace
    5. Split concatenated street suffixes         DRIVEBAYTOWN → DRIVE BAYTOWN
    6. Remove redundant punctuation               double commas, trailing commas
    7. (Optional) Expand abbreviations            DR → DRIVE
    8. Final whitespace collapse
    """
    if not raw or not isinstance(raw, str):
        return raw

    s = raw.upper().strip()

    # Step 2 — remove trailing period from abbreviations (INC. CO. LTD.)
    s = re.sub(r"\b([A-Z]{1,6})\.", r"\1", s)

    # Step 3 — normalize comma spacing
    s = re.sub(r",\s*", ", ", s)

    # Step 4 — collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # Step 5 — split glued suffixes (pad with space so lookahead works at start)
    s = " " + s
    s = fix_concatenated_suffix(s)
    s = s.strip()

    # Step 6 — clean up punctuation artifacts
    s = re.sub(r",\s*,", ",", s)   # double comma
    s = re.sub(r"\s+,", ",", s)    # space before comma
    s = s.strip(" ,")

    # Step 7 — optional abbreviation expansion
    if expand_abbrev:
        for pattern, replacement in ABBREV_MAP.items():
            s = re.sub(pattern, replacement, s)

    # Step 8 — final whitespace pass
    s = re.sub(r"\s+", " ", s).strip()

    return s


def clean_whitespace_only(value: str) -> str:
    """Basic whitespace clean (strip + collapse) for non-address columns."""
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value.strip())


def process_dataframe(
    df: pd.DataFrame,
    address_cols: list,
    whitespace_cols: list,
    expand_abbrev: bool,
) -> pd.DataFrame:
    """Apply appropriate cleaning to each selected column."""
    cleaned = df.copy()

    for col in address_cols:
        mask = cleaned[col].notna()
        cleaned.loc[mask, col] = cleaned.loc[mask, col].astype(str).apply(
            lambda v: normalize_address(v, expand_abbrev)
        )

    for col in whitespace_cols:
        if col not in address_cols:
            mask = cleaned[col].notna()
            cleaned.loc[mask, col] = cleaned.loc[mask, col].astype(str).apply(
                clean_whitespace_only
            )

    return cleaned


def count_changes(orig: pd.DataFrame, cleaned: pd.DataFrame, cols: list) -> dict:
    stats = {}
    for col in cols:
        if col in orig.columns:
            diff = (orig[col].fillna("").astype(str) != cleaned[col].fillna("").astype(str))
            stats[col] = int(diff.sum())
    stats["_total"] = sum(v for k, v in stats.items() if not k.startswith("_"))
    return stats


def find_fuzzy_duplicates(df: pd.DataFrame, col: str, threshold: int = 85) -> pd.DataFrame:
    """
    Identify rows in `col` whose cleaned values are near-duplicates (similar but not identical).
    Returns a DataFrame of pairs with their similarity score.
    """
    vals = df[col].dropna().astype(str).unique().tolist()
    pairs = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            score = fuzz.token_sort_ratio(vals[i], vals[j])
            if threshold <= score < 100:
                pairs.append({"Value A": vals[i], "Value B": vals[j], "Similarity %": round(score, 1)})
    return pd.DataFrame(pairs).sort_values("Similarity %", ascending=False) if pairs else pd.DataFrame()


def build_diff_records(orig: pd.DataFrame, cleaned: pd.DataFrame, cols: list, max_rows: int = 200) -> pd.DataFrame:
    """Build a flat DataFrame of changed cells for display."""
    records = []
    for col in cols:
        o_series = orig[col].fillna("").astype(str)
        c_series = cleaned[col].fillna("").astype(str)
        changed_idx = o_series[o_series != c_series].index
        for idx in changed_idx[:max_rows]:
            records.append({
                "Row": idx + 2,           # Excel row (1-indexed + header)
                "Column": col,
                "Before": orig.at[idx, col],
                "After":  cleaned.at[idx, col],
            })
    return pd.DataFrame(records)


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🧹 Address & Whitespace Cleaner")
st.caption("Normalize addresses, fix punctuation noise, collapse whitespace, and flag near-duplicates.")

st.divider()

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.subheader("1 · Upload Excel File")
uploaded_file = st.file_uploader("Upload .xlsx or .xls", type=["xlsx", "xls"])

if not uploaded_file:
    st.info("Upload an Excel file to begin.")
    st.stop()

# ── Step 2: Sheet ──────────────────────────────────────────────────────────────
st.subheader("2 · Select Sheet")
try:
    xls = pd.ExcelFile(uploaded_file)
except Exception as e:
    st.error(f"Cannot read file: {e}")
    st.stop()

selected_sheet = st.selectbox("Sheet", xls.sheet_names)

try:
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, dtype=str)
except Exception as e:
    st.error(f"Cannot load sheet: {e}")
    st.stop()

st.caption(f"Loaded **{df.shape[0]:,} rows × {df.shape[1]} columns** from `{selected_sheet}`")

with st.expander("Preview raw data (first 5 rows)"):
    st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ── Step 3: Select columns ─────────────────────────────────────────────────────
st.subheader("3 · Select Columns")

str_cols  = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
auto_cols = [c for c in TARGET_FIELDS if c in str_cols]

selected_cols = st.multiselect(
    "Choose columns to clean",
    options=str_cols,
    default=auto_cols,
    help="Only text columns are listed. Known address fields are pre-selected.",
)

if not selected_cols:
    st.warning("Select at least one column to continue.")
    st.stop()

st.divider()

# ── Step 4: Cleaning mode ──────────────────────────────────────────────────────
st.subheader("4 · Cleaning Mode")
st.caption("Choose what to apply to the selected columns.")

col_ck1, col_ck2 = st.columns(2)

with col_ck1:
    do_whitespace = st.checkbox(
        "✂️  Whitespace cleaning",
        value=True,
        help="Strips leading/trailing spaces and collapses multiple spaces into one.",
    )
    do_normalize = st.checkbox(
        "🔧  Address normalization",
        value=True,
        help=(
            "Full address fix: comma spacing, concatenated suffixes "
            "(DRIVEBAYTOWN → DRIVE BAYTOWN), punctuation cleanup, uppercase."
        ),
    )

with col_ck2:
    # Sub-option — only shown when normalization is on
    if do_normalize:
        expand_abbrev = st.checkbox(
            "📖  Expand abbreviations (DR → DRIVE, AVE → AVENUE)",
            value=False,
            help="Expands common street abbreviations. Applied after normalization.",
        )
    else:
        expand_abbrev = False

    run_fuzzy = st.checkbox(
        "🔍  Flag near-duplicate addresses",
        value=False,
        help="Detects values that look like the same address written differently (≥85% similarity).",
    )
    if run_fuzzy:
        fuzzy_threshold = st.slider("Similarity threshold (%)", 70, 99, 85)
    else:
        fuzzy_threshold = 85

if not do_whitespace and not do_normalize:
    st.warning("Enable at least one cleaning mode to continue.")
    st.stop()

# Show a plain-English summary of what will run
mode_parts = []
if do_whitespace and not do_normalize:
    mode_parts.append("**Whitespace only** — strip & collapse spaces")
if do_normalize and not do_whitespace:
    mode_parts.append("**Normalization only** — address fixes (whitespace included as part of normalization)")
if do_whitespace and do_normalize:
    mode_parts.append("**Whitespace + Normalization** — full pipeline")
if expand_abbrev:
    mode_parts.append("+ abbreviation expansion")
if run_fuzzy:
    mode_parts.append(f"+ near-duplicate scan at {fuzzy_threshold}%")

st.info("  ·  ".join(mode_parts))

st.divider()

# ── Step 5: Run ────────────────────────────────────────────────────────────────
st.subheader("5 · Run")

if st.button("▶  Run Cleaning", type="primary"):

    # Decide which columns go into which pipeline
    # If only whitespace: all selected → whitespace pipeline
    # If only normalize: all selected → address pipeline
    # If both: all selected → address pipeline (which includes whitespace steps)
    if do_normalize:
        addr_cols_run = selected_cols
        ws_cols_run   = []          # normalization already collapses whitespace
    else:
        addr_cols_run = []
        ws_cols_run   = selected_cols

    with st.spinner("Processing…"):
        cleaned_df = process_dataframe(df, addr_cols_run, ws_cols_run, expand_abbrev)
        all_cols_run = list(set(addr_cols_run + ws_cols_run))
        stats      = count_changes(df, cleaned_df, all_cols_run)
        diff_df    = build_diff_records(df, cleaned_df, all_cols_run)

    total_changed = stats["_total"]
    cols_touched  = sum(1 for k, v in stats.items() if not k.startswith("_") and v > 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows",              f"{df.shape[0]:,}")
    m2.metric("Columns processed", len(all_cols_run))
    m3.metric("Cells modified",    total_changed)
    m4.metric("Columns changed",   cols_touched)

    st.divider()

    # Per-column breakdown
    with st.expander("Per-column change counts", expanded=True):
        bd = pd.DataFrame([
            {
                "Column": k,
                "Cells changed": v,
                "Mode": "Normalize" if k in addr_cols_run else "Whitespace",
            }
            for k, v in stats.items() if not k.startswith("_")
        ]).sort_values("Cells changed", ascending=False)
        st.dataframe(bd, use_container_width=True, hide_index=True)

    # Cell diff
    if not diff_df.empty:
        with st.expander(f"Cell-level diff — {len(diff_df)} changes (showing up to 200)", expanded=True):
            st.dataframe(diff_df, use_container_width=True, hide_index=True)
    else:
        st.success("No changes detected — data was already clean.")

    # Fuzzy duplicates
    if run_fuzzy and addr_cols_run:
        st.divider()
        st.subheader("Near-duplicate Detection")
        for col in addr_cols_run:
            with st.spinner(f"Scanning '{col}'…"):
                dup_df = find_fuzzy_duplicates(cleaned_df, col, threshold=fuzzy_threshold)
            if dup_df.empty:
                st.success(f"**{col}**: No near-duplicates above {fuzzy_threshold}%.")
            else:
                st.warning(f"**{col}**: {len(dup_df)} near-duplicate pair(s) found.")
                st.dataframe(dup_df, use_container_width=True, hide_index=True)

    st.divider()

    # Download
    st.subheader("Download")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        cleaned_df.to_excel(writer, index=False, sheet_name=selected_sheet[:31])
        if not diff_df.empty:
            diff_df.to_excel(writer, index=False, sheet_name="Changes Log")
    output.seek(0)

    original_name = uploaded_file.name.rsplit(".", 1)[0]
    st.download_button(
        label="⬇️  Download Cleaned Excel (+ Changes Log sheet)",
        data=output.getvalue(),
        file_name=f"{original_name}_cleaned.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("The downloaded file includes a **Changes Log** sheet with every before/after cell.")

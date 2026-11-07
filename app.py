
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from google import genai


# ==========================================
# 1. GEMINI API CONNECTION
# ==========================================

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None


# ==========================================
# 2. ICE SILVER + DEEP AUBERGINE + MARIGOLD
# ==========================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --aubergine: #24152F;
    --aubergine-light: #32203F;
    --aubergine-card: #2C1D38;
    --silver: #E8E9ED;
    --silver-muted: #B8B9C5;
    --marigold: #F4B942;
    --border: #493653;
}

body, .gradio-container {
    background: var(--aubergine) !important;
    color: var(--silver) !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
    padding: 24px !important;
}

#hero {
    background: linear-gradient(120deg, #392344, #24152F 65%, #49304D);
    border: 1px solid #654B6B;
    border-radius: 22px;
    padding: 32px !important;
    margin-bottom: 24px;
}

#hero h1 {
    color: var(--silver) !important;
    font-size: 36px !important;
    font-weight: 800 !important;
    margin-bottom: 8px !important;
}

#hero p {
    color: #D2C7D8 !important;
}

.hero-icon {
    width: 52px;
    height: 52px;
    background: var(--marigold);
    color: #24152F;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 15px;
    font-size: 27px;
    font-weight: 800;
}

.hero-tag {
    display: inline-block;
    background: #493653;
    color: #F4B942;
    padding: 7px 13px;
    border-radius: 20px;
    margin: 5px 5px 0 0;
    font-size: 12px;
    font-weight: 600;
}

.panel {
    background: var(--aubergine-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 17px !important;
    padding: 22px !important;
    margin-bottom: 16px !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    color: var(--silver) !important;
}

.gradio-container p,
.gradio-container label {
    color: var(--silver-muted) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #21142B !important;
    color: var(--silver) !important;
    border: 1px solid #604A69 !important;
    border-radius: 10px !important;
}

.gradio-container button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.primary-btn button {
    background: var(--marigold) !important;
    color: #24152F !important;
    border: none !important;
    min-height: 46px !important;
    font-weight: 800 !important;
}

.primary-btn button:hover {
    background: #FFD16B !important;
}

.secondary-btn button {
    background: #493653 !important;
    color: var(--silver) !important;
    border: 1px solid #80658A !important;
}

.tab-nav {
    background: #32203F !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 6px !important;
}

.tab-nav button {
    color: var(--silver-muted) !important;
    border-radius: 9px !important;
}

.tab-nav button.selected {
    background: var(--marigold) !important;
    color: #24152F !important;
}

.prose, .markdown {
    color: var(--silver) !important;
}

.prose h2, .markdown h2,
.prose h3, .markdown h3 {
    color: var(--marigold) !important;
}

.prose strong {
    color: var(--silver) !important;
}

.prose table {
    color: var(--silver) !important;
}

.prose th {
    background: #493653 !important;
}

.prose td, .prose th {
    border-color: #604A69 !important;
}

footer {
    display: none !important;
}

#footer-note {
    text-align: center;
    color: #A99AAD;
    font-size: 12px;
    padding: 22px;
}

/* Improved text visibility */

.gradio-container,
.gradio-container .prose,
.gradio-container .markdown {
    color: #E8E9ED !important;
}

.gradio-container label,
.gradio-container p,
.gradio-container span {
    color: #E8E9ED !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    color: #E8E9ED !important;
    background: #21142B !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #B8B9C5 !important;
    opacity: 1 !important;
}

.gradio-container .prose table td {
    color: #E8E9ED !important;
}

.gradio-container .prose table th {
    color: #F4B942 !important;
}

"""


# ==========================================
# 3. DATA ANALYSIS
# ==========================================

def analyze_dataset(file):

    if file is None:
        raise gr.Error("Please upload a dataset first.")

    try:
        if file.lower().endswith(".csv"):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)

        if data.empty or len(data.columns) == 0:
            raise gr.Error("The uploaded dataset is empty.")

        rows = len(data)
        columns = len(data.columns)

        missing = int(data.isnull().sum().sum())
        total_cells = data.size
        available = total_cells - missing

        percentage = round(
            (missing / total_cells) * 100, 2
        ) if total_cells else 0

        duplicates = int(data.duplicated().sum())

        missing_report = pd.DataFrame({
            "Column": data.columns.astype(str),
            "Missing Values": data.isnull().sum().values,
            "Missing Percentage": (
                data.isnull().mean() * 100
            ).round(2).values
        })

        missing_report = missing_report[
            missing_report["Missing Values"] > 0
        ].sort_values(
            by="Missing Percentage",
            ascending=False
        )

        # Chart theme
        plt.rcParams.update({
            "figure.facecolor": "#2C1D38",
            "axes.facecolor": "#2C1D38",
            "axes.labelcolor": "#E8E9ED",
            "text.color": "#E8E9ED",
            "xtick.color": "#B8B9C5",
            "ytick.color": "#B8B9C5",
            "axes.edgecolor": "#604A69",
            "font.size": 10
        })

        # Missing values chart
        fig1, ax1 = plt.subplots(figsize=(10, 5))

        if not missing_report.empty:
            ax1.barh(
                missing_report["Column"],
                missing_report["Missing Values"],
                color="#F4B942"
            )

            ax1.set_title(
                "Missing Values by Column",
                color="#F4B942",
                fontsize=14,
                pad=15
            )
            ax1.invert_yaxis()
            ax1.grid(axis="x", alpha=0.15)

        else:
            ax1.text(
                0.5, 0.5,
                "No Missing Values Found",
                ha="center",
                va="center",
                color="#F4B942",
                fontsize=14
            )
            ax1.set_axis_off()

        fig1.tight_layout()

        # Missing vs available chart
        fig2, ax2 = plt.subplots(figsize=(6, 5))

        if total_cells > 0:
            ax2.pie(
                [missing, available],
                labels=["Missing Data", "Available Data"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#F4B942", "#B8B9C5"],
                textprops={"color": "#24152F"},
                wedgeprops={
                    "width": 0.42,
                    "edgecolor": "#2C1D38"
                }
            )

        ax2.set_title(
            "Data Availability",
            color="#F4B942",
            fontsize=14,
            pad=15
        )

        fig2.tight_layout()

        # Gemini AI insights
        if client is None:
            raise gr.Error(
                "Gemini API key is not configured on the server."
            )

        prompt = f"""
        You are Data Narrate AI, an intelligent data analysis assistant.

        Explain this dataset in simple English.

        Rows: {rows}
        Columns: {columns}
        Missing Cells: {missing}
        Missing Percentage: {percentage}%
        Duplicate Rows: {duplicates}

        Column-wise missing values:
        {missing_report.to_string(index=False)}

        Explain:
        1. Dataset overview.
        2. Major data quality problems.
        3. Columns needing attention.
        4. Practical cleaning recommendations.

        Do not assume why data is missing without evidence.
        Use clear headings and simple language.
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        ai_insights = (
            response.text
            if response.text
            else "AI insights could not be generated."
        )

        # Downloadable report
        report = f"""
DATA NARRATE AI - DATA QUALITY REPORT

Total Rows: {rows}
Total Columns: {columns}
Missing Cells: {missing}
Missing Percentage: {percentage}%
Duplicate Rows: {duplicates}

COLUMN-WISE MISSING VALUES

{missing_report.to_string(index=False)}

AI-GENERATED INSIGHTS

{ai_insights}
"""

        report_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8"
        )

        report_file.write(report)
        report_file.close()

        summary = f"""
## ✦ Analysis Completed Successfully

### Dataset Overview

| Metric | Result |
|---|---:|
| Total Rows | {rows:,} |
| Total Columns | {columns} |
| Missing Cells | {missing:,} |
| Missing Percentage | {percentage}% |
| Duplicate Rows | {duplicates:,} |

### Column-wise Missing Values

{missing_report.to_string(index=False) if not missing_report.empty else "No missing values found in this dataset."}

### ✦ AI-Generated Insights

{ai_insights}
"""

        return (
            summary,
            fig1,
            fig2,
            report_file.name,
           gr.update(
    choices=data.columns.astype(str).tolist(),
    value=str(data.columns[0]),
    label="Select Column"
),
            data,
            data.head(10)
        )

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(f"Analysis failed: {str(e)}")


# ==========================================
# 4. DATA CLEANING
# ==========================================

def clean_dataset(data, column, method):

    if data is None:
        raise gr.Error(
            "Please upload and analyze a dataset first."
        )

    if not column or column not in data.columns:
        raise gr.Error("Please select a valid column.")

    cleaned = data.copy()

    if method == "Fill with Mean":

        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            raise gr.Error(
                "Mean is only available for numerical columns."
            )

        cleaned[column] = cleaned[column].fillna(
            cleaned[column].mean()
        )

    elif method == "Fill with Median":

        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            raise gr.Error(
                "Median is only available for numerical columns."
            )

        cleaned[column] = cleaned[column].fillna(
            cleaned[column].median()
        )

    elif method == "Fill with Mode":

        mode_values = cleaned[column].mode(dropna=True)

        if len(mode_values) > 0:
            cleaned[column] = cleaned[column].fillna(
                mode_values.iloc[0]
            )

    elif method == "Drop Rows with Missing Values":

        cleaned = cleaned.dropna(subset=[column]).copy()

    # CSV download
    csv_file = tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False
    )
    csv_file.close()

    cleaned.to_csv(csv_file.name, index=False)

    # Excel download
    excel_file = tempfile.NamedTemporaryFile(
        suffix=".xlsx",
        delete=False
    )
    excel_file.close()

    cleaned.to_excel(excel_file.name, index=False)

    status = f"""
## ✓ Cleaning Completed

**Selected Column:** {column}

**Cleaning Method:** {method}

**Original Rows:** {len(data)}

**Remaining Rows:** {len(cleaned)}

**Remaining Columns:** {len(cleaned.columns)}

**Remaining Missing Cells:** {int(cleaned.isnull().sum().sum())}
"""

    return (
        cleaned,
        status,
        cleaned.head(10),
        csv_file.name,
        excel_file.name
    )


# ==========================================
# 5. CREATE WEBSITE
# ==========================================

with gr.Blocks(
    title="Data Narrate AI",
    theme=gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="purple",
        neutral_hue="slate"
    ),
    css=CUSTOM_CSS
) as app:

    # Header
    gr.HTML("""
    <div id="hero">
        <div style="display:flex;align-items:center;gap:16px;">
            <div class="hero-icon">✦</div>
            <div>
                <h1>Data Narrate AI</h1>
                <p>Intelligent Data Analysis & Cleaning Workspace</p>
            </div>
        </div>

        <div style="margin-top:20px;">
            <span class="hero-tag">✦ AI Powered</span>
            <span class="hero-tag">◈ Smart Analytics</span>
            <span class="hero-tag">✓ Data Cleaning</span>
        </div>
    </div>
    """)

    gr.Markdown("""
    ## Welcome to your Data Workspace

    Upload your dataset, discover data quality issues,
    understand AI insights and prepare cleaner data.
    """)

    with gr.Tabs():

        # ----------------------------------
        # DATA ANALYSIS
        # ----------------------------------

        with gr.Tab("📊  Data Analysis"):

            with gr.Row():

                with gr.Column(
                    scale=1,
                    elem_classes=["panel"]
                ):

                    gr.Markdown("### Upload Dataset")

                    gr.Markdown(
                        "Choose a CSV or Excel file to begin."
                    )

                    upload_file = gr.File(
                        label="Select your dataset",
                        file_types=[".csv", ".xlsx"],
                        type="filepath"
                    )

                    analyze_button = gr.Button(
                        "✦  Analyze Dataset",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )

                    gr.Markdown(
                        "Supported formats: CSV and XLSX"
                    )

                with gr.Column(
                    scale=2,
                    elem_classes=["panel"]
                ):

                    gr.Markdown("### What you will discover")

                    gr.Markdown("""
                    **01 — Dataset Overview**

                    Understand rows, columns and data structure.

                    **02 — Data Quality**

                    Identify missing values and duplicate records.

                    **03 — Visual Analytics**

                    Explore data quality through charts.

                    **04 — AI Insights**

                    Get easy-to-understand explanations and recommendations.
                    """)

            with gr.Column(elem_classes=["panel"]):

                gr.Markdown("## Analysis Results")

                result = gr.Markdown(
                    "Your analysis report will appear here after uploading a dataset."
                )

            with gr.Row():

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Missing Values Analysis")
                    chart1 = gr.Plot(
                        label="Missing Values by Column"
                    )

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Data Availability")
                    chart2 = gr.Plot(
                        label="Missing vs Available Data"
                    )

            with gr.Column(elem_classes=["panel"]):

                gr.Markdown("### Download Analysis Report")

                download_report = gr.File(
                    label="Download AI Data Quality Report"
                )

        # ----------------------------------
        # DATA CLEANING STUDIO
        # ----------------------------------

        with gr.Tab("🧹  Data Cleaning Studio"):

            gr.Markdown("""
            ## Data Cleaning Studio

            Choose a column and apply a suitable missing-value
            treatment to prepare your dataset.
            """)

            with gr.Row():

                with gr.Column(
                    scale=1,
                    elem_classes=["panel"]
                ):

                    gr.Markdown("### Cleaning Configuration")

                    column_dropdown = gr.Dropdown(
                        label="Select Column",
                        choices=[]
                    )

                    strategy_dropdown = gr.Dropdown(
                        label="Select Cleaning Method",
                        choices=[
                            "Fill with Mean",
                            "Fill with Median",
                            "Fill with Mode",
                            "Drop Rows with Missing Values"
                        ],
                        value="Fill with Median"
                    )

                    clean_button = gr.Button(
                        "✦  Apply Cleaning",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )

                    cleaning_status = gr.Markdown(
                        "Your cleaning result will appear here."
                    )

                with gr.Column(
                    scale=2,
                    elem_classes=["panel"]
                ):

                    gr.Markdown("### Cleaning Methods")

                    gr.Markdown("""
                    **Mean** — Replaces missing numerical values with the average.

                    **Median** — Replaces missing numerical values with the middle value.

                    **Mode** — Replaces missing values with the most frequent value.

                    **Drop Rows** — Removes rows that contain missing values in the selected column.
                    """)

            with gr.Column(elem_classes=["panel"]):

                gr.Markdown("### Cleaned Dataset Preview")

                cleaned_preview = gr.Dataframe(
                    label="Cleaned Dataset",
                    interactive=False
                )

            with gr.Row():

                with gr.Column(elem_classes=["panel"]):

                    gr.Markdown("### Download Cleaned CSV")

                    download_csv = gr.File(
                        label="CSV File"
                    )

                with gr.Column(elem_classes=["panel"]):

                    gr.Markdown("### Download Cleaned Excel")

                    download_excel = gr.File(
                        label="Excel File"
                    )

    # Shared dataset state
    dataset_state = gr.State()

    # Analysis button connection
    analyze_button.click(
        fn=analyze_dataset,
        inputs=upload_file,
        outputs=[
            result,
            chart1,
            chart2,
            download_report,
            column_dropdown,
            dataset_state,
            cleaned_preview
        ]
    )

    # Cleaning button connection
    clean_button.click(
        fn=clean_dataset,
        inputs=[
            dataset_state,
            column_dropdown,
            strategy_dropdown
        ],
        outputs=[
            dataset_state,
            cleaning_status,
            cleaned_preview,
            download_csv,
            download_excel
        ]
    )

    # Footer
    gr.HTML("""
    <div id="footer-note">
        DATA NARRATE AI · Intelligent Data Analysis Workspace
        <br><br>
        Built with Python · Pandas · Gradio · Gemini AI
    </div>
    """)


# ==========================================
# 6. LAUNCH WEBSITE
# ==========================================

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )

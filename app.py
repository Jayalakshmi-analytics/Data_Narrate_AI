
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from google import genai

# Gemini API connection
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None


# DATA ANALYSIS
def analyze_dataset(file):

    if file is None:
        raise gr.Error("Please upload a dataset.")

    try:
        if file.endswith(".csv"):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)

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
            "Column": data.columns,
            "Missing Values": data.isnull().sum().values,
            "Missing Percentage": (
                data.isnull().mean() * 100
            ).round(2).values
        })

        missing_report = missing_report[
            missing_report["Missing Values"] > 0
        ].sort_values(
            by="Missing Percentage", ascending=False
        )

        # Missing values chart
        fig1, ax1 = plt.subplots(figsize=(10, 5))

        if not missing_report.empty:
            ax1.barh(
                missing_report["Column"],
                missing_report["Missing Values"]
            )
            ax1.set_title("Missing Values by Column")
            ax1.invert_yaxis()
        else:
            ax1.text(
                0.5, 0.5, "No Missing Values",
                ha="center", va="center"
            )
            ax1.set_axis_off()

        fig1.tight_layout()

        # Missing vs available chart
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        if total_cells > 0:
            ax2.pie(
                [missing, available],
                labels=["Missing Data", "Available Data"],
                autopct="%1.1f%%",
                startangle=90
            )

        ax2.set_title("Missing vs Available Data")
        fig2.tight_layout()

        # AI report
        if client is None:
            raise gr.Error(
                "Gemini API key is not configured."
            )

        prompt = f"""
        You are Data Narrate AI.
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
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

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

{response.text}
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
## DATA QUALITY REPORT

| Metric | Result |
|---|---:|
| Total Rows | {rows:,} |
| Total Columns | {columns} |
| Missing Cells | {missing:,} |
| Missing Percentage | {percentage}% |
| Duplicate Rows | {duplicates} |

## AI-GENERATED INSIGHTS

{response.text}
"""

        return (
            summary,
            fig1,
            fig2,
            report_file.name,
            gr.Dropdown(
                choices=list(data.columns),
                value=data.columns[0],
                label="Select Column"
            ),
            data,
            data.head(10)
        )

    except Exception as e:
        raise gr.Error(str(e))


# DATA CLEANING
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
            raise gr.Error("Mean is only for numerical columns.")
        cleaned[column] = cleaned[column].fillna(
            cleaned[column].mean()
        )

    elif method == "Fill with Median":
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            raise gr.Error("Median is only for numerical columns.")
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
        suffix=".csv", delete=False
    )
    csv_file.close()
    cleaned.to_csv(csv_file.name, index=False)

    # Excel download
    excel_file = tempfile.NamedTemporaryFile(
        suffix=".xlsx", delete=False
    )
    excel_file.close()
    cleaned.to_excel(excel_file.name, index=False)

    status = f"""
Cleaning completed successfully!

Selected Column: {column}
Cleaning Method: {method}
Remaining Rows: {len(cleaned)}
Remaining Columns: {len(cleaned.columns)}
Remaining Missing Cells: {cleaned.isnull().sum().sum()}
"""

    return (
        cleaned,
        status,
        cleaned.head(10),
        csv_file.name,
        excel_file.name
    )


# CREATE WEBSITE
with gr.Blocks(title="Data Narrate AI") as app:

    gr.Markdown("# 📊 Data Narrate AI")
    gr.Markdown("### Your AI-Powered Data Analysis Assistant")

    with gr.Tab("Data Analysis"):

        upload_file = gr.File(
            label="Upload CSV or Excel Dataset",
            file_types=[".csv", ".xlsx"],
            type="filepath"
        )

        analyze_button = gr.Button(
            "Analyze Dataset", variant="primary"
        )

        result = gr.Markdown()
        chart1 = gr.Plot(label="Missing Values by Column")
        chart2 = gr.Plot(label="Missing vs Available Data")

        download_report = gr.File(
            label="Download AI Report"
        )

    with gr.Tab("Data Cleaning Studio"):

        gr.Markdown("### 🧹 Clean Your Dataset")

        column_dropdown = gr.Dropdown(
            label="Select Column", choices=[]
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
            "Apply Cleaning", variant="primary"
        )

        cleaning_status = gr.Textbox(
            label="Cleaning Result", lines=5
        )

        cleaned_preview = gr.Dataframe(
            label="Cleaned Dataset Preview"
        )

        download_csv = gr.File(
            label="Download Cleaned CSV"
        )

        download_excel = gr.File(
            label="Download Cleaned Excel"
        )

    dataset_state = gr.State()

    analyze_button.click(
        fn=analyze_dataset,
        inputs=upload_file,
        outputs=[
            result, chart1, chart2, download_report,
            column_dropdown, dataset_state, cleaned_preview
        ]
    )

    clean_button.click(
        fn=clean_dataset,
        inputs=[
            dataset_state, column_dropdown, strategy_dropdown
        ],
        outputs=[
            dataset_state, cleaning_status, cleaned_preview,
            download_csv, download_excel
        ]
    )


if __name__ == "__main__":
    app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)

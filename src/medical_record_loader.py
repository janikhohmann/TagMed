"""
TagMed Medical Record Loader - Clinical report and medical data integration

This module provides comprehensive medical record loading and processing capabilities
for the TagMed medical annotation application. It handles the integration of clinical
reports, medical findings, and patient documentation into the annotation workflow.

Key Features:
- CSV-based medical report loading with configurable data sources
- Structured medical report formatting with clinical sections
- Multi-exam support for longitudinal patient studies
- Integration with TagMed configuration management
- Standardized medical terminology and report structure
- Error handling for missing or corrupted medical data

# The loader supports standard medical report sections including indication,
# findings, evaluation, and recommendations, formatted for optimal readability
# in medical annotation workflows.

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import pandas as pd
import os

from config_handler import ConfigHandler

class MedicalRecordLoader:
    """
    Handles loading and processing of medical reports from different table formats.

    Features:
    - Automatically detects CSV separators.
    - Supports CSV and Excel files (.csv, .xlsx, .xls).
    - Allows alternative column names for flexibility.
    - Handles missing columns gracefully without crashing.
    """

    def __init__(self):

        # get the path to currentyl selected medical report file
        self.config = ConfigHandler()
        self.selected_medical_report_file = self.config.get("selected_medical_report_file", "")

        # ==== Define expected columns in the medical report table ====
        self.column_aliases = {
            "exam_ID": ["exam_ID", "exam_id", "ExamID", "examid","Exam_Id", "exam_Id","ID", "id"],
        }

    def _find_column_name(self, df, possible_names):
        """
        Finds the first matching column in the DataFrame from a list of possible names.
        Returns None if no match is found.
        """
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    

    def _load_medical_report(self, file_path):
        """
        Loads a table from a CSV or Excel file.
        Automatically detects CSV separator.
        """
        if not os.path.exists(file_path):
            return None

        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext in [".csv", ".txt"]:
                # Auto-detect separator for CSV
                return pd.read_csv(file_path, sep=None, engine="python")
            elif file_ext in [".xlsx", ".xls"]:
                return pd.read_excel(file_path)
            else:
                print(f"Unsupported file format: {file_ext}")
                return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None




    def fill_description(self, exam_ID):
        """
        Returns a structured text description for a given exam ID.
        Searches for matching records and formats them into sections.

        If columns are missing, that section is skipped.
        If no match is found, returns 'No annotation available.'

        """

        reports = self._load_medical_report(self.selected_medical_report_file)
        if reports is None:
            return "No annotation available."
        

        # Find the actual name of the exam_ID column
        exam_id_col = self._find_column_name(reports, self.column_aliases["exam_ID"])
        if not exam_id_col:
            print("Error: No valid exam_ID column found in the report. Please check the file format in medical_record_loader.py.")
            return "No annotation available."

        text_rows = reports[reports[exam_id_col] == exam_ID]
        if text_rows.empty:
            return "No annotation available."

        available_columns = text_rows.columns.tolist()

        description = ""

        for col in available_columns:
            description += f"\n==== {col.upper()} ====\n"
            internal_exam_number = 1
            for index, row in text_rows.iterrows():
                description += f"Exam: {internal_exam_number}\n"
                description += f"{row[col]}\n\n"
                internal_exam_number += 1

        return description.strip() if len(description)>1 else "No annotation available."

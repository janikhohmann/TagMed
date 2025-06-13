
import pandas as pd
import tkinter as tk
import os

from config_handler import ConfigHandler

class MedicalRecordLoader:
    """
    Class to handle loading and processing of medical records from CSV files.
    """

    def __init__(self):

        self.config = ConfigHandler()
        self.selected_medical_report_file = self.config.get("selected_medical_report_file", "")


    def fill_description(self, exam_ID):
        """
        gets Description for each exam from the text table.
        OUTPUT: String
        """


        de = ""
        text_table = pd.read_csv(self.selected_medical_report_file, sep=";")

        text_row = text_table[text_table['exam_ID'] == exam_ID]

        if not text_row.empty:
            de = de + "**INDICATION**\n"
            for exam in range(len(text_row)):
                de = de + "\t" + "Exam: " + str(exam + 1 ) + "\n"
                de = de + str(text_row.iloc[exam]['indication']) + "\n"
            de = de + "\n**FINDING_LIVER**\n"
            for exam in range(len(text_row)):
                de = de + "\t" + "Exam: " + str(exam + 1 ) + "\n"
                de = de + str(text_row.iloc[exam]['finding_liver']) + "\n"
            de = de + "\n**EVALUATION**\n"
            for exam in range(len(text_row)):
                de = de + "\t" + "Exam: " + str(exam + 1 ) + "\n"
                de = de + str(text_row.iloc[exam]['evaluation']) + "\n"
            de = de + "\n**RECOMMENDATION**\n"
            for exam in range(len(text_row)):
                de = de + "\t" + "Exam: " + str(exam + 1 ) + "\n"
                de = de + str(text_row.iloc[exam]['recommendation']) + "\n"      

        else: 
            de = "No annotation available."

        return de
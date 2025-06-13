'''
AnnotationLoader
This Module is responsible for loading and saving of all annotations. 
'''

import os
import pandas as pd
import ast
from tkinter import messagebox

from config_handler import ConfigHandler

class AnnotationLoader():
    def __init__(self):
        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.selected_anno_table_file = config.get("selected_anno_table_file")

    def available_patients(self):
        """Returns sorted list of available patient folders."""
        return sorted(
            [d for d in os.listdir(self.selected_image_folder) if os.path.isdir(os.path.join(self.selected_image_folder, d))]
        )

    def count_exams_and_files(self, patient_id):
        """Counts exams (folders) and files for a given patient."""
        exams = 0
        total_files = 0
        path = os.path.join(self.selected_image_folder, patient_id)
        for root, dirs, files in os.walk(path):
            exams += len(dirs)
            total_files += len(files)
        return exams, total_files
    
    def get_exams(self, patient_id):
        """
        function gives back Folder names inside the patient folder
        INPUT: patient_id, data directory
        OUTPUT: list of folders
        """
        path = os.path.join(self.selected_image_folder, patient_id)
        exams = [
            name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ]
        return exams

    def count_annotated_images(self, patient_id):
        """Counts annotated images for a given patient based on the annotation table."""
        try : 
            anno_table = pd.read_csv(self.selected_anno_table_file, sep=";")
            filtered = anno_table[
                (anno_table["pat_ID"] == patient_id) & (anno_table["x"] != "NN")
            ]
        except FileNotFoundError:
            print(f"[ERROR] Annotation table file not found: {self.selected_anno_table_file}")
            #messagebox.showinfo("ERROR", f"Annotation table file not found: {self.selected_anno_table_file}")
            return 0
        return len(filtered)
    
       

    def load_annotations_from_annotable(self):
        '''
        Reads annotation table and returns all annotations as panda dataframe.
        '''
        anno_table = pd.read_csv(self.selected_anno_table_file, sep=";")
        annotated_images = anno_table[anno_table["x"].apply(lambda x: isinstance(x, str) and x.startswith("["))]
        return annotated_images

    def load_annotations_in_internal_list(self):
        '''
        This function loads all annotations already done into the internal list.
        The List will be passed to the Annotation Handler. 
        '''

        try:
            all_annotated_data = self.load_annotations_from_annotable()
            print(f"[DEBUG] Read {len(all_annotated_data)} total rows from {self.selected_anno_table_file}")

            patient_id_col = "pat_ID"
            exam_id_col = "exam_ID"
            img_id_col = "img_ID"
            class_col = "class"
            x_col, y_col, w_col, h_col = "x", "y", "w", "h"

            required_csv_cols = [patient_id_col, exam_id_col, img_id_col, class_col, x_col, y_col, w_col, h_col]
            if not all(col in all_annotated_data.columns for col in required_csv_cols):
                missing_cols = [col for col in required_csv_cols if col not in all_annotated_data.columns]
                print(f"[ERROR] Missing required columns in CSV: {missing_cols}")
                return

            # convert pandas df into list of dicts
            #self.all_internal_annotation_data = all_annotated_data.to_dict('records')
            self.all_annotated_data = all_annotated_data
            print(f"[DEBUG] Found {len(all_annotated_data)} rows for your dataset.")

        except KeyError as e:
            print(e)
        
        return all_annotated_data

        
    def save_annotations_to_anno_table(self):
        """
        Synchronizes self.all_internal_annotation_data with the CSV table (list structure)
        """

        try:
            if not self.all_internal_annotation_data:
                print("No annotations to save")
                return

            # Lade bestehende Tabelle
            anno_table = pd.read_csv(self.selected_anno_table_file, sep=";")

            # Konvertiere gespeicherte String-Listen in echte Listen
            for col in ['x', 'y', 'w', 'h', 'class']:
                anno_table[col] = anno_table[col].apply(
                    lambda val: ast.literal_eval(val) if isinstance(val, str) and val.startswith('[') else []
                )

            # Mapping von img_ID auf Zeilenindex
            table_map = {row['img_ID']: idx for idx, row in anno_table.iterrows()}

            # Gruppiere Annotationen nach img_ID
            df_annotations = pd.DataFrame(self.all_internal_annotation_data)
            grouped = df_annotations.groupby('img_ID')

            new_count = 0
            updated_count = 0

            processed_img_ids = set()

            for img_id, group in grouped:
                processed_img_ids.add(img_id)
                ann_data = group.to_dict(orient='list')

                if img_id in table_map:
                    idx = table_map[img_id]
                    for col in ['x', 'y', 'w', 'h', 'class']:
                        values = ann_data.get(col, [])
                        anno_table.at[idx, col] = values if values else ["NN"]
                    updated_count += 1
                else:
                    # Bild nicht vorhanden → Neue Zeile
                    pat_id = getattr(self.gui, 'patient_id', 'NN')
                    exam_id = getattr(self.gui, 'selected_exam', 'NN')
                    img_num = getattr(self.gui, 'image_number', 'NN')
                    new_path = getattr(self.gui, 'image_path', 'NN')

                    new_row = {
                        'pat_ID': pat_id,
                        'exam_ID': exam_id,
                        'img_num': img_num,
                        'img_ID': img_id,
                        'exam_mode': 'NN',
                        'organ': 'NN',
                        'x': ann_data.get('x', ["NN"]),
                        'y': ann_data.get('y', ["NN"]),
                        'w': ann_data.get('w', ["NN"]),
                        'h': ann_data.get('h', ["NN"]),
                        'class': ann_data.get('class', ["NN"]),
                        'file_type': 'img',
                        'new_path': new_path
                    }

                    anno_table = pd.concat([anno_table, pd.DataFrame([new_row])], ignore_index=True)
                    new_count += 1

            # Optional: Für Bilder ohne Annotationen, aber die schon im Table existieren → auf "NN" setzen
            for idx, row in anno_table.iterrows():
                img_id = row['img_ID']
                if img_id not in processed_img_ids:
                    for col in ['x', 'y', 'w', 'h', 'class']:
                        anno_table.at[idx, col] = ["NN"]

            # Speichern
            anno_table.to_csv(self.selected_anno_table_file, sep=";", index=False)
            print(f"Saved {new_count} new annotations and updated {updated_count} existing annotations")

        except Exception as e:
            print(f"Error saving annotations to table: {e}")


    def filter_annotations_for_patient(self, all_annotated_data, patient_id):
        '''
        Filters the annotation list for the current patient and the current exam.
        The result is saved in self.annotations_filtered.
        '''

        # filter for specific patient and exam
        try:
            filtered = all_annotated_data[
                (all_annotated_data["pat_ID"].astype(str) == str(patient_id))
            ]

            self.annotations_filtered = filtered
            return filtered

        except Exception as e:
            print(f"Error filtering annotations for {patient_id}: {e}")

        

    def sync_all_annotations_w_pat_annotations(self):
        '''
        Synchronizes self.annotations_filtered back to self.all_internal_annotation_data.
        Replaces all annotations of the current patient/exam in the large list.
        '''
        try:
            selected_pat = str(self.patient_id)
            selected_exam = str(self.selected_exam)

            # delete old annotations for patient and exam in self.all_internal_annotation_data
            self.all_internal_annotation_data = [
                ann for ann in self.all_internal_annotation_data
                if not (str(ann.get("pat_ID")) == selected_pat and str(ann.get("exam_ID")) == selected_exam)
            ]
        
            # replace deleted annotations with new ones
            self.all_internal_annotation_data.extend(self.annotations_filtered)


        except Exception as e:
            print(f"Error syncing annotations for {selected_pat} in {selected_exam}: {e}")


    def create_default_anno_table(self):
        '''
        This function creates an default annotation table by searching for available file in the selected iamge folder
        '''
        image_folder = self.selected_image_folder
        if not image_folder or not os.path.exists(image_folder):
            print("No valid image folder set - no Anno table created.")
            return

        entries = []

        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue  # only image files
                rel_path = os.path.relpath(root, image_folder)
                parts = rel_path.split(os.sep)
                pat_id = parts[0] if len(parts) > 0 else "NN"
                exam_id = parts[1].split("_")[-1] if len(parts) > 1 else "NN"
                img_num = int(os.path.splitext(file)[0].split("_")[2])
                img_id = os.path.splitext(file)[0]

                entries.append({
                    "pat_ID": pat_id,
                    "exam_ID": exam_id,
                    "img_num": img_num,
                    "img_ID": img_id,
                    "class": "NN",
                    "x": "NN",
                    "y": "NN",
                    "w": "NN",
                    "h": "NN",
                    "exam_mode": "NN",
                    "organ": "NN",
                    "file_type": "img",
                    "new_path": os.path.join(root, file)
                })

        df = pd.DataFrame(entries)
        default_path = os.path.join("auto_generated_anno_table.csv")
        df.to_csv(default_path, sep=";", index=False)
        print(f"Annotationstable created: {default_path}")

        # set auto generated file in config
        config = ConfigHandler()
        config.set("selected_anno_table_file", default_path)
        config.save()






'''
AnnotationLoader
This Module is responsible for loading and saving of all annotations. 
'''

import os
import pandas as pd
import ast
import cv2
from pathlib import Path
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
                (anno_table["pat_ID"] == patient_id) &
                (
                    (anno_table["class"] != "NN") | (anno_table["class_polygon"] != "NN")
                )
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
        # load all possible annotations, even when they are empty
        annotated_images = anno_table[anno_table["x"].apply(lambda x: isinstance(x, str))]

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
        Synchronizes self.all_annotated_data with the CSV table (list structure)
        """
        try:
            self.all_annotated_data.to_csv(self.selected_anno_table_file, sep=";", index=False)
            annotations = self.all_annotated_data["x"] != "NN"
            print(f"[DEBUG] Saved {len(annotations)} rows to {self.selected_anno_table_file}")

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

    def _get_video_frame_count(self, video_path, frame_interval=1):
        """
        Ermittelt die Anzahl der extrahierbaren Frames aus einem Video.
        
        Args:
            video_path (str): Pfad zur Videodatei
            frame_interval (int): Extrahiere jeden N-ten Frame (Standard: 1)
            
        Returns:
            int: Anzahl der extrahierbaren Frames
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[WARNING] Kann Video nicht öffnen: {video_path}")
                return 0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Berechne Anzahl der Frames basierend auf Intervall
            extractable_frames = total_frames // frame_interval
            print(f"[INFO] Video {os.path.basename(video_path)}: {total_frames} total Frames → {extractable_frames} extrahierbare Frames (Intervall: {frame_interval})")
            
            return extractable_frames
            
        except Exception as e:
            print(f"[ERROR] Fehler beim Lesen der Video-Eigenschaften von {video_path}: {e}")
            return 0

    def _video_already_framed(self, video_base_name, root_dir):
        """
        Prüft ob für ein Video bereits Frames vorhanden sind.
        
        Args:
            video_base_name (str): Basis-Name des Videos (ohne Extension)
            root_dir (str): Verzeichnis in dem nach Frames gesucht wird
            
        Returns:
            bool: True wenn bereits Frames vorhanden sind
        """
        try:
            files_in_dir = os.listdir(root_dir)
            
            # Suche nach Dateien die dem Video-Namen entsprechen und "frame" enthalten
            frame_files = [
                f for f in files_in_dir 
                if video_base_name in f and "frame" in f.lower()
                and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            
            if frame_files:
                print(f"[INFO] Video '{video_base_name}' bereits geframed - überspringe ({len(frame_files)} Frames gefunden)")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Fehler beim Prüfen auf bestehende Frames: {e}")
            return False

    def _is_video_file(self, file_path):
        """Prüft ob Datei ein unterstütztes Videoformat hat."""
        supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return Path(file_path).suffix.lower() in supported_video_formats

        

    # def sync_all_annotations_w_pat_annotations(self):
    #     '''
    #     Synchronizes self.annotations_filtered back to self.all_internal_annotation_data.
    #     Replaces all annotations of the current patient/exam in the large list.
    #     '''
    #     try:
    #         selected_pat = str(self.patient_id)
    #         selected_exam = str(self.selected_exam)

    #         # delete old annotations for patient and exam in self.all_internal_annotation_data
    #         self.all_internal_annotation_data = [
    #             ann for ann in self.all_internal_annotation_data
    #             if not (str(ann.get("pat_ID")) == selected_pat and str(ann.get("exam_ID")) == selected_exam)
    #         ]
        
    #         # replace deleted annotations with new ones
    #         self.all_internal_annotation_data.extend(self.annotations_filtered)


    #     except Exception as e:
    #         print(f"Error syncing annotations for {selected_pat} in {selected_exam}: {e}")


    def create_default_anno_table(self):
        '''
        This function creates an default annotation table by searching for available file in the selected image folder.
        Erweitert um Video-Support: Erstellt Einträge für alle extrahierbaren Frames aus Videos.
        '''
        image_folder = self.selected_image_folder
        if not image_folder or not os.path.exists(image_folder):
            print("No valid image folder set - no Anno table created.")
            return

        entries = []
        frame_interval = 1  # Jeden 30. Frame extrahieren (konfigurierbar)

        for root, dirs, files in os.walk(image_folder):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, image_folder)
                parts = rel_path.split(os.sep)
                pat_id = parts[0] if len(parts) > 0 else "NN"
                exam_id = parts[1].split("_")[-1] if len(parts) > 1 else "NN"
                
                # === Behandlung von Bilddateien ===
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    try:
                        img_num = int(os.path.splitext(file)[0].split("_")[2])
                    except (IndexError, ValueError):
                        img_num = 0
                    
                    img_id = os.path.splitext(file)[0]
                    file_type = "frame" if "frame" in file else "img"

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
                        "class_polygon": "NN",
                        "polygon": "NN",
                        "exam_mode": "NN",
                        "organ": "NN",
                        "file_type": file_type,
                        "bb_annotype": "NN",
                        "polygon_annotype": "NN",
                        "masks": "NN",
                        "new_path": file_path
                    })
                    
                # === Behandlung von Videodateien ===
                elif self._is_video_file(file_path):
                    video_base_name = os.path.splitext(file)[0]
                    
                    # Prüfe ob Video bereits geframed ist
                    if self._video_already_framed(video_base_name, root):
                        continue  # Überspringe dieses Video
                    
                    print(f"[INFO] Verarbeite Video: {file}")
                    
                    # Ermittle Anzahl der extrahierbaren Frames
                    frame_count = self._get_video_frame_count(file_path, frame_interval)
                    
                    if frame_count > 0:
                        # Erstelle Einträge für alle extrahierbaren Frames (beginne mit Frame 1)
                        for frame_idx in range(frame_count):
                            frame_number = (frame_idx * frame_interval) + 1  # Beginne mit Frame 1
                            frame_id = f"{video_base_name}_frame_{frame_number:06d}"
                            
                            # img_num bleibt konstant (nicht hochzählen)
                            try:
                                base_img_num = int(video_base_name.split("_")[-1]) if "_" in video_base_name else 1
                            except (ValueError, IndexError):
                                base_img_num = 1
                            
                            entries.append({
                                "pat_ID": pat_id,
                                "exam_ID": exam_id,
                                "img_num": base_img_num,  # Konstant für alle Frames dieses Videos
                                "img_ID": frame_id,
                                "class": "NN",
                                "x": "NN",
                                "y": "NN",
                                "w": "NN",
                                "h": "NN",
                                "class_polygon": "NN",
                                "polygon": "NN",
                                "exam_mode": "NN",
                                "organ": "NN",
                                "file_type": "frame",
                                "bb_annotype": "NN",
                                "polygon_annotype": "NN",
                                "masks": "NN",
                                "new_path": file_path  # Zeigt auf das ursprüngliche Video
                            })
                            
                        print(f"[INFO] {frame_count} Frame-Einträge für Video '{file}' erstellt (Frame 1-{frame_count})")
                    else:
                        print(f"[WARNING] Keine Frames für Video '{file}' extrahierbar")

        df = pd.DataFrame(entries)
        default_path = os.path.join("auto_generated_anno_table.csv")
        df.to_csv(default_path, sep=";", index=False)
        
        total_entries = len(entries)
        video_frames = len([e for e in entries if e["file_type"] == "frame" and "_frame_" in e["img_ID"]])
        image_entries = total_entries - video_frames
        
        print(f"[INFO] Annotationstabelle erstellt: {default_path}")
        print(f"[INFO] Gesamt: {total_entries} Einträge ({image_entries} Bilder, {video_frames} Video-Frames)")

        # set auto generated file in config
        config = ConfigHandler()
        config.set("selected_anno_table_file", default_path)
        config.save()






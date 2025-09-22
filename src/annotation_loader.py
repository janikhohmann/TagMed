"""
AnnotationLoader - Data Management for Medical Annotations

This module handles the loading, saving, and management of medical image annotations
for the TagMed application. It provides functionality for reading annotation tables,
counting annotated images, and managing patient data structures.

Features:
- Loading and saving annotations from/to CSV files
- Patient and exam statistics calculation
- Annotation table creation and validation
- Support for both image and video frame annotations
- Automatic handling of missing annotation files

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import os
import pandas as pd
import cv2
from pathlib import Path
from tkinter import messagebox
import ast

from config_handler import ConfigHandler

class AnnotationLoader():
    """
    Manages loading and saving of medical image annotations.
    
    This class handles all aspects of annotation data management including
    reading from CSV files, counting statistics, and maintaining the
    annotation database for medical images and video frames.
    
    Attributes:
        selected_image_folder (str): Path to folder containing patient data
        selected_anno_table_file (str): Path to CSV annotation database
    """
    
    def __init__(self):
        """
        Initialize AnnotationLoader with configuration settings.
        
        Loads folder paths and annotation table path from ConfigHandler.
        """
        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.selected_anno_table_file = config.get("selected_anno_table_file")

    def available_patients(self):
        """
        Returns sorted list of available patient folders.
        
        Scans the configured image folder for subdirectories representing
        patient IDs and returns them in sorted order.
        
        Returns:
            list: Sorted list of patient ID strings
        """
        return sorted(
            [d for d in os.listdir(self.selected_image_folder) if os.path.isdir(os.path.join(self.selected_image_folder, d))]
        )

    def count_exams_and_files(self, patient_id):
        """
        Counts exams (folders) and files for a given patient.
        
        Recursively walks through the patient's directory structure
        to count the number of exam folders and total files.
        
        Args:
            patient_id (str): Patient identifier
            
        Returns:
            tuple: (number of exams, total number of files)
        """
        exams = 0
        total_files = 0
        path = os.path.join(self.selected_image_folder, patient_id)
        for root, dirs, files in os.walk(path):
            exams += len(dirs)
            total_files += len(files)

            # Include frames from videos in the count
            for file_path in files:
                if self._is_video_file(file_path): # check if it's a video file
                    video_base_name = os.path.splitext(file_path)[0]
                    if not self._video_already_framed(video_base_name, root): # check if already framed
                        video_path = os.path.join(root, file_path)
                        frames = self._get_video_frame_count(video_path, frame_interval=1) # count all frames
                        total_files += frames
                        total_files -= 1 # subtract the original video file
                    
        return exams, total_files
    
    def get_exams(self, patient_id):
        """
        Retrieve folder names (exams) inside a patient's directory. 
        
        Scans the patient folder for subdirectories representing
        different medical examinations or study sessions.
        
        Args:
            patient_id (str): Patient identifier
            
        Returns:
            list: List of exam folder names
        """
        path = os.path.join(self.selected_image_folder, patient_id)
        exams = [
            name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ]
        return exams

    def count_annotated_images(self, patient_id):
        """
        Counts annotated images for a given patient based on the annotation table.
        
        Reads the annotation CSV file and counts entries for the specified patient
        that have actual annotations (non-"NN" values for class or class_polygon).
        
        Args:
            patient_id (str): Patient identifier
            
        Returns:
            int: Number of annotated images for this patient
        """
        try : 
            anno_table = pd.read_csv(self.selected_anno_table_file, sep=";")
            filtered = anno_table[
                (anno_table["pat_ID"] == patient_id) &
                (
                    (anno_table["class"].notna() | (anno_table["class_polygon"].notna()))
                )
            ]
        except FileNotFoundError:
            return 0
        return len(filtered)
    
       

    def load_annotations_from_annotable(self):
        """
        Reads annotation table and returns all annotations as pandas DataFrame.
        
        Loads the complete annotation CSV file and filters for entries that
        contain string values in the 'x' column, indicating actual annotations.
        
        Returns:
            pd.DataFrame: DataFrame containing all valid annotation entries
        """
        anno_table = pd.read_csv(self.selected_anno_table_file, sep=";")
        # Load all possible annotations, including empty ones (None/NaN values)
        # Return all rows instead of filtering for string values in 'x' column
        annotated_images = anno_table  # Return all rows, not just those with string x values

        return annotated_images

    def load_annotations_in_internal_list(self):
        """
        Loads all existing annotations into an internal list structure.
        
        This function processes the annotation table and creates an internal
        representation that can be efficiently used by the annotation handlers.
        Essential for maintaining annotation state during editing sessions.
        
        Returns:
            list: Internal list structure containing all annotations
        """
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
            self.all_annotated_data = all_annotated_data
            print(f"[DEBUG] Found {len(all_annotated_data)} rows for your dataset.")

        except KeyError as e:
            print(e)
        #print(all_annotated_data)

        return all_annotated_data

        
    def save_annotations_to_anno_table(self):
        """
        Synchronizes annotation data with the CSV table.
        
        Saves the current annotation data structure (self.all_annotated_data)
        to the configured annotation table CSV file.
        """
        try:
            self.all_annotated_data.to_csv(self.selected_anno_table_file, sep=";", index=False)
            annotations = self.all_annotated_data["x"] != None
            print(f"[DEBUG] Saved {len(annotations)} rows to {self.selected_anno_table_file}")

        except Exception as e:
             print(f"Error saving annotations to table: {e}")


    def _get_video_frame_count(self, video_path, frame_interval=1):
        """
        Determines the number of extractable frames from a video file.
        
        Uses OpenCV to read video properties and calculate how many frames
        can be extracted based on the specified frame interval.
        
        Args:
            video_path (str): Path to the video file
            frame_interval (int): Extract every N-th frame (default: 1)
            
        Returns:
            int: Number of extractable frames
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[WARNING] Cannot open video: {video_path}")
                return 0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Calculate number of frames based on interval
            extractable_frames = total_frames // frame_interval
            print(f"[INFO] Video {os.path.basename(video_path)}: {total_frames} total frames → {extractable_frames} extractable frames (interval: {frame_interval})")
            
            return extractable_frames
            
        except Exception as e:
            print(f"[ERROR] Error reading video properties from {video_path}: {e}")
            return 0

    def _video_already_framed(self, video_base_name, root_dir):
        """
        Checks if frames already exist for a video file.
        
        Searches the directory for existing frame files that correspond to
        the given video name. Helps avoid duplicate frame extraction.
        
        Args:
            video_base_name (str): Base name of the video (without extension)
            root_dir (str): Directory to search for existing frames
            
        Returns:
            bool: True if frames already exist for this video
        """
        try:
            files_in_dir = os.listdir(root_dir)
            
            # Search for files that match the video name and contain "frame"
            frame_files = [
                f for f in files_in_dir 
                if video_base_name in f and "frame" in f.lower()
                and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            
            if frame_files:
                print(f"[INFO] Video '{video_base_name}' already framed - skipping ({len(frame_files)} frames found)")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Error checking for existing frames: {e}")
            return False

    def _is_video_file(self, file_path):
        """
        Checks if a file is a supported video format.
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            bool: True if the file is a supported video format
        """
        supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return Path(file_path).suffix.lower() in supported_video_formats



    def create_default_anno_table(self):
        """
        Creates a default annotation table by scanning for available files.
        
        This function walks through the selected image folder and creates
        annotation table entries for all found images and video frames.
        Extended with video support: Creates entries for all extractable
        frames from video files.
        
        The generated table includes:
        - Image file entries with metadata
        - Video frame entries for trackable content
        - Default None values for empty annotations
        """
        image_folder = self.selected_image_folder
        # check if image folder is set
        if not image_folder or not os.path.exists(image_folder):
            #print(f"[ERROR] No image folder selected.")
            messagebox.showinfo("ERROR", f"No valid image folder selected.\n\nPlease select an image folder in the File menu.")
            return


        entries = []
        frame_interval = 1  # Extract every frame (configurable)

        for root, dirs, files in os.walk(image_folder):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, image_folder)
                parts = rel_path.split(os.sep)
                pat_id = parts[0] if len(parts) > 0 else None
                exam_id = parts[1].split("_")[-1] if len(parts) > 1 else None

                # === Image Files ===
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
                        "class": None,
                        "x": None,
                        "y": None,
                        "w": None,
                        "h": None,
                        "class_polygon": None,
                        "polygon": None,
                        "exam_mode": None,
                        "organ": None,
                        "file_type": file_type,
                        "bb_annotype": None,
                        "polygon_annotype": None,
                        "masks": None,
                        "file_path": file_path
                    })

                # === Video Files ===
                elif self._is_video_file(file_path):
                    video_base_name = os.path.splitext(file)[0]

                    # Check if video is already framed
                    if self._video_already_framed(video_base_name, root):
                        continue  # Skip this video

                    print(f"[INFO] Processing video: {file}")

                    # Determine the number of extractable frames
                    frame_count = self._get_video_frame_count(file_path, frame_interval)
                    
                    if frame_count > 0:
                        # Create entries for all extractable frames (starting with frame 1)
                        for frame_idx in range(frame_count):
                            frame_number = (frame_idx * frame_interval) + 1  # Start with frame 1
                            frame_id = f"{video_base_name}_frame_{frame_number:06d}"

                            # img_num remains constant (do not increment)
                            try:
                                base_img_num = int(video_base_name.split("_")[-1]) if "_" in video_base_name else 1
                            except (ValueError, IndexError):
                                base_img_num = 1
                            
                            entries.append({
                                "pat_ID": pat_id,
                                "exam_ID": exam_id,
                                "img_num": base_img_num, 
                                "img_ID": frame_id,
                                "class": None,
                                "x": None,
                                "y": None,
                                "w": None,
                                "h": None,
                                "class_polygon": None,
                                "polygon": None,
                                "exam_mode": None,
                                "organ": None,
                                "file_type": "frame",
                                "bb_annotype": None,
                                "polygon_annotype": None,
                                "masks": None,
                                "path": file_path  # Points to the original video
                            })

                        print(f"[INFO] {frame_count} frame entries created for video '{file}' (Frame 1-{frame_count})")
                    else:
                        print(f"[WARNING] No frames extractable for video '{file}'")

        df = pd.DataFrame(entries)
        default_path = os.path.join("..", "auto_generated_anno_table.csv")
        df.to_csv(default_path, sep=";", index=False)
        
        total_entries = len(entries)
        video_frames = len([e for e in entries if e["file_type"] == "frame" and "_frame_" in e["img_ID"]])
        image_entries = total_entries - video_frames

        print(f"[INFO] Annotation table created: {default_path}")
        print(f"[INFO] Total: {total_entries} entries ({image_entries} images, {video_frames} video frames)")

        # set auto generated file in config
        config = ConfigHandler()
        config.set("selected_anno_table_file", default_path)
        config.save()

        return default_path
    




# ==== function is not used anymore ====

    # def filter_annotations_for_patient(self, all_annotated_data, patient_id):
    #     """
    #     Filters the annotation list for a specific patient.
        
    #     Takes the complete annotation dataset and returns only the annotations
    #     belonging to the specified patient. The filtered result is also stored
    #     in self.annotations_filtered for later use.
        
    #     Args:
    #         all_annotated_data (pd.DataFrame): Complete annotation dataset
    #         patient_id (str): Patient identifier to filter for
            
    #     Returns:
    #         pd.DataFrame: Filtered annotations for the specified patient
    #     """
    #     # Filter for specific patient - optimized for performance
    #     try:
    #         # Direct equality comparison
    #         filtered = all_annotated_data[all_annotated_data["pat_ID"] == patient_id]

    #         self.annotations_filtered = filtered
    #         return filtered

    #     except Exception as e:
    #         print(f"Error filtering annotations for {patient_id}: {e}")
import os
import tkinter as tk

from config_handler import ConfigHandler

class DataLoader:
    def __init__(self):

        self.config = ConfigHandler()
        self.selected_image_folder = self.config.get("selected_image_folder", "")



    def load_images_in_dir(self, patient_id, selected_exam):
        """
        This function loads images from a specified directory based on the patient ID and selected exam.
        """
        # Form the path to the selected exam folder
        image_folder = os.path.join(self.selected_image_folder, patient_id, selected_exam)
        
        # Check if folder exists and load images
        if os.path.exists(image_folder):
            images = [i for i in os.listdir(image_folder) if i.endswith(('.png', '.jpg', '.jpeg')) and "frame" not in i]

            return images
        
        else:
            print(f"Folder for selected exam '{selected_exam}' not found.")


    def load_videos_in_dir(self, patient_id, selected_exam):
        """
        This function loads videos from a specified directory based on the patient ID and selected exam.
        """
        # Form the path to the selected exam folder
        video_folder = os.path.join(self.selected_image_folder, patient_id, selected_exam)
        
        # Check if folder exists and load images
        if os.path.exists(video_folder):
            videos = [i for i in os.listdir(video_folder) if i.endswith(('.mov'))]

            return videos

        else:
            print(f"Folder for selected exam '{selected_exam}' not found.")
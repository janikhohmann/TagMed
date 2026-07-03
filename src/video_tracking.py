import os
import ast
import tkinter as tk
import pandas as pd
import requests
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2
import numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
import matplotlib.pyplot as plt



from config_handler import ConfigHandler
from video_annotation_handler import VideoAnnotationHandler
from mask_handler import MaskHandler
from sam2_tracking import SAM2Tracking
from sam3_tracking import SAM3Tracking
from medsam2_tracking import MedSAM2Tracking

class  VideoTracking:
    def __init__(self, gui):
        self.gui = gui
        self.video_annotation_handler = VideoAnnotationHandler(gui)
        self.sam2_tracking = SAM2Tracking(gui)
        self.sam3_tracking = SAM3Tracking(gui)
        self.medsam2_tracking = MedSAM2Tracking(gui)

        self.mask_handler = MaskHandler(gui)
        
        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.image_size = config.get("image_size")




    def tracking_starter(self):

        selected_annotation = self.gui.video_annotation_listbox.curselection()
        if not selected_annotation:
            self.gui.select_annotation_before_tracking_gui()
            return


        tracking_type = self.gui.tracking_type.get()
        
        if tracking_type == "Simple":
            self.simple_tracking_method()
            
        if tracking_type == "SAM3":
            available = self.sam3_tracking.check_if_sam_3_is_available() # option to download different models

            if available:
                self.gui.wait_for_tracking_gui(on_complete=lambda: self.sam3_tracking.sam3_tracking_method())
            else:
                return
            

        if tracking_type in ["SAM2 large", "SAM2 tiny", "SAM2 US Liver finetuned"]:
            available = self.sam2_tracking.check_if_sam_2_is_available() # option to download different models

            if available:
                self.gui.wait_for_tracking_gui(on_complete=lambda: self.sam2_tracking.sam2_tracking_method())
            else:
                return

        if tracking_type in ["MedSAM2", "MedSAM2 US Heart", "MedSAM2 MRI Liver Lesion"]:
            available = self.medsam2_tracking.check_if_medsam2_is_available() # option to download MedSAM2 model

            if available:
                self.gui.wait_for_tracking_gui(on_complete=lambda: self.medsam2_tracking.medsam2_tracking_method())
            else:
                return
            
        # if tracking_type == "MedSAM2 US Heart":
        #     available = self.medsam2_tracking.check_if_medsam2_is_available() # option to download MedSAM2 model

        #     if available:
        #         self.gui.wait_for_tracking_gui(on_complete=lambda: self.medsam2_tracking.medsam2_tracking_method())
        #     else:
        #         return
            
        # if tracking_type == "MedSAM2 MRI Liver Lesion":
        #     available = self.medsam2_tracking.check_if_medsam2_is_available() # option to download MedSAM2 model

        #     if available:
        #         self.gui.wait_for_tracking_gui(on_complete=lambda: self.medsam2_tracking.medsam2_tracking_method())
        #     else:
        #         return



# ===== SIMPLE METHOD =====
    def simple_tracking_method(self):
        """Copies the annotation data to all subsequent frames in the central list."""
        
        selected_annotation = self.gui.video_annotation_listbox.curselection()
        selected_annotation_index = selected_annotation[0]
        current_frames = self.gui.current_frames
        current_frame_index = self.gui.current_frame_index
        current_image_id = current_frames[current_frame_index].split(".")[0]
        selected_class = self.gui.video_selected_class.get()

        # find matching row in DataFrame
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == current_image_id
        if not match.any():
            print(f"[ERROR] No entry in database for {current_image_id}")
            return
        df_index = self.gui.all_annotations[match].index[0]
        row = self.gui.all_annotations.loc[df_index]

        # Check if Polygon or Bounding Box
        selected_text = self.gui.video_annotation_listbox.get(selected_annotation_index)
        is_polygon = "Polygon" in selected_text

        # ==== Polygon Tracking ====
        if is_polygon:
            # get polygon data from data frame
            polygon_list = self._safe_parse_list(row.get('polygon'))
            polygon_class_list = self._safe_parse_list(row.get('class_polygon'))

            if selected_annotation_index >= len(polygon_list):
                print(f"[ERROR] Polygon index {selected_annotation_index} out of range.")
                return

            polygon = polygon_list[selected_annotation_index]
            polygon_class = polygon_class_list[selected_annotation_index]

            if not isinstance(polygon, list) or len(polygon) < 3:
                print("[ERROR] Invalid polygon data.")
                return

            try:
                for i in range(current_frame_index + 1, len(current_frames)):
                    next_img_id = current_frames[i].split(".")[0]

                    match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == next_img_id
                    if match_next.any():
                        next_df_index = self.gui.all_annotations[match_next].index[0]

                        existing_polygons = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, 'polygon'])
                        existing_class_polygons = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, 'class_polygon'])
                        existing_polygon_annotypes = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, 'polygon_annotype'])

                        # Stelle sicher, dass die Listen lang genug sind
                        while len(existing_polygons) <= selected_annotation_index:
                            existing_polygons.append([])
                            existing_class_polygons.append("")
                            existing_polygon_annotypes.append("")

                        # Setze an der passenden Stelle
                        existing_polygons[selected_annotation_index] = polygon
                        existing_class_polygons[selected_annotation_index] = polygon_class
                        existing_polygon_annotypes[selected_annotation_index] = "tracking"

                        # Speichere die aktualisierten Listen zurück
                        self.gui.all_annotations.at[next_df_index, 'polygon'] = existing_polygons
                        self.gui.all_annotations.at[next_df_index, 'class_polygon'] = existing_class_polygons
                        self.gui.all_annotations.at[next_df_index, 'polygon_annotype'] = existing_polygon_annotypes


                print("[INFO] Simple Tracking Method Completed")

            except:
                print("[ERROR] Simple Tracking Method could not be finished.")


        # ==== Bounding Box Tracking ====        
        else:

            # get original bounding box data from data frame
            x_list = self._safe_parse_list(row.get('x'))
            y_list = self._safe_parse_list(row.get('y'))
            w_list = self._safe_parse_list(row.get('w'))
            h_list = self._safe_parse_list(row.get('h'))
            class_list = self._safe_parse_list(row.get('class'))
            #annotype_list = self._safe_parse_list(row.get('bb_annotype'))

            try:
                x = x_list[selected_annotation_index]
                y = y_list[selected_annotation_index]
                w = w_list[selected_annotation_index]
                h = h_list[selected_annotation_index]
                selected_class = class_list[selected_annotation_index]
                #annotype = annotype_list[selected_annotation_index]

            except IndexError:
                print(f"[ERROR] rect_id Index {selected_annotation_index} out of range.")
                return


            # copy annotation and insert tracking as annotationtype
            try:
                for i in range(current_frame_index + 1, len(current_frames)):
                    next_img_id = current_frames[i].split(".")[0] # get the next frame

                    # searching for match for the next frame
                    match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == next_img_id
                    if match_next.any():
                        next_df_index = self.gui.all_annotations[match_next].index[0]

                        for col, val in zip(['x', 'y', 'w', 'h', 'class', 'bb_annotype'], [x, y, w, h, selected_class, 'tracking']):
                            existing_list = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, col])

                            # Stelle überschreiben, falls vorhanden, sonst auffüllen
                            if len(existing_list) > selected_annotation_index:
                                existing_list[selected_annotation_index] = val
                            else:
                                # Falls Liste zu kurz: mit None auffüllen und anhängen
                                while len(existing_list) < selected_annotation_index:
                                    existing_list.append(None)
                                existing_list.append(val)

                            self.gui.all_annotations.at[next_df_index, col] = existing_list

                
                print("[INFO] Simple Tracking Method Completed")
            except:
                print("[ERROR] Simple Tracking Method could not be finished.")




    def show_debug_visuals(self, image_rgb, input_box, mask, mask_cropped):
        """
        Zeigt vier nebeneinander angeordnete Bilder:
        1. Originalbild
        2. Originalbild mit Input-Box
        3. Maske (Rohdaten)
        4. Binarisierte Maske
        """
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # Originalbild
        axs[0].imshow(image_rgb)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Bild mit Box
        image_with_box = image_rgb.copy()
        x0, y0, x1, y1 = map(int, input_box)
        cv2.rectangle(image_with_box, (x0, y0), (x1, y1), (255, 0, 0), 2)  # Rotes Rechteck
        axs[1].imshow(image_with_box)
        axs[1].set_title("Image with Input Box")
        axs[1].axis('off')

        # Maske (grau)
        axs[2].imshow(mask.squeeze(), cmap='gray')
        axs[2].set_title("Raw Mask")
        axs[2].axis('off')

        # Binärmaske
        binary_mask = (mask_cropped.squeeze() > 0.5).astype(np.uint8)
        axs[3].imshow(binary_mask, cmap='gray')
        axs[3].set_title("Cropped Mask")
        axs[3].axis('off')

        plt.tight_layout()
        plt.show()





# ====== Helper Functions ======

    def delete_all_annotations_for_one_video(self):
        """
        Deletes all annotations for the selected video (i.e., for all frames of that video).
        Also removes associated rectangles from the canvas.
        """
        # selected_annotation = self.gui.video_annotation_listbox.curselection()
        # selected_annotation_index = selected_annotation[0]
        current_frames = self.gui.current_frames
        # current_frame_index = self.gui.current_frame_index
        # current_image_id = current_frames[current_frame_index].split(".")[0]

        try:
            for i in range(len(current_frames)):
                img_id = current_frames[i].split(".")[0] # get the next frame

                self.mask_handler.delete_all_masks_for_one_video(img_id) # delete all masks for the current frame

                # searching for match for the next frame
                match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == img_id
                if match_next.any():
                    next_df_index = self.gui.all_annotations[match_next].index[0]

                    for col in ['x', 'y', 'w', 'h', 'class', 'bb_annotype', 'polygon', 'class_polygon', 'polygon_annoytype']:
                        self.gui.all_annotations.at[next_df_index, col] = "NN"
            
            self.mask_handler.clear_all_masks()
            self.video_annotation_handler.clear_all_annotations()
            self.video_annotation_handler.delete_all_polygons()
            self.video_annotation_handler.delete_all_bounding_boxes()
            self.video_annotation_handler.update_video_listbox_with_annotation_colors()

        except Exception as e:
            print(f"[ERROR] Could not delete all annotations for video: {e}")
            import traceback
            traceback.print_exc()
            return
            

    def _safe_parse_list(self, value):
        """Hilfsfunktion zum sicheren Parsen von Listen aus Strings oder Listen."""
        if isinstance(value, list):
            return value
        if pd.isna(value) or value in ("NN", "", None):
            return []
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception as e:
                print(f"[ERROR] Parsing list failed: {e} | value = {repr(value)}")
        return []
    



 
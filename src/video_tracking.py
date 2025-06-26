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

from config_handler import ConfigHandler
from video_annotation_handler import VideoAnnotationHandler

class  VideoTracking:
    def __init__(self, gui):
        self.gui = gui
        self.video_annotation_handler = VideoAnnotationHandler(gui)
        
        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.image_size = config.get("image_size")


        # Define the URLs for SAM 2.1 checkpoints
        SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
        self.sam2p1_hiera_t_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
        self.sam2p1_hiera_s_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
        self.sam2p1_hiera_b_plus_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
        self.sam2p1_hiera_l_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

        model_dir="../models"  # Verzeichnis, in dem das Modell gespeichert wird
        self.abs_model_dir = os.path.abspath(model_dir)

        self.sam2_model_l_path = os.path.join(self.abs_model_dir, "sam2.1_hiera_large.pt")
        self.sam2_predictor = None






    def tracking_starter(self):

        selected_annotation = self.gui.video_annotation_listbox.curselection()
        if not selected_annotation:
            self.gui.select_annotation_before_tracking_gui()
            return


        tracking_type = self.gui.tracking_type.get()
        
        if tracking_type == "Simple":
            self.simple_tracking_method()

        if tracking_type == "SAM 2":
            available = self.check_if_sam_2_is_available() # option to download different models

            if available:
                self.sam2_tracking_method()
            else:
                return


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

        print(selected_text)


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
                    next_img_id = current_frames[i].split(".")[0] # get the next frame

                    # searching for match for the next frame
                    match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == next_img_id
                    if match_next.any():
                        next_df_index = self.gui.all_annotations[match_next].index[0]

                        #polygon_copy = [point.copy() for point in self.polygon_points]
                        self.gui.all_annotations.at[next_df_index, 'polygon'] = self._append_or_init_list(
                            self.gui.all_annotations.at[next_df_index, 'polygon'], polygon)
                        self.gui.all_annotations.at[next_df_index, 'class_polygon'] = self._append_or_init_list(
                            self.gui.all_annotations.at[next_df_index, 'class_polygon'], polygon_class)
                        self.gui.all_annotations.at[next_df_index, 'polygon_annotype'] = self._append_or_init_list(
                            self.gui.all_annotations.at[next_df_index, 'polygon_annotype'], "tracking")
                print("[INFO] Simple Tracking Method Completed")

            except:
                print("[ERROR] Simple Tracking Method could not be finished.")


        # ==== Bounding Box Tracking ====        
        else:

            # get bounding box data from data frame
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
                slected_class = class_list[selected_annotation_index]
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

                        for col, val in zip(['x', 'y', 'w', 'h', 'class', 'bb_annotype'], [x, y, w, h, slected_class, 'tracking']):
                            self.gui.all_annotations.at[next_df_index, col] = self._append_or_init_list(self.gui.all_annotations.at[next_df_index, col], val)
                print("[INFO] Simple Tracking Method Completed")
            except:
                print("[ERROR] Simple Tracking Method could not be finished.")




# ====== SAM 2 ======

    def check_if_sam_2_is_available(self):
        if os.path.exists(self.sam2_model_l_path):
            return True
        else:
            load = self.gui.ask_for_sam2_download()
            if load:
                self.download_sam_2_model()
                return True
            else:
                return False

    def download_sam_2_model(self):
        # Sicherstellen, dass das Zielverzeichnis existiert
        os.makedirs(self.abs_model_dir, exist_ok=True)

        # Überprüfen, ob die Datei bereits existiert
        if os.path.exists(self.sam2_model_l_path):
            print(f"SAM2-Modell existiert bereits unter: {self.sam2_model_l_path}")
        else:
            print(f"Versuche, SAM2-Modell herunterzuladen nach: {self.sam2_model_l_path}")
            try:
                response = requests.get(self.sam2p1_hiera_l_url, stream=True)
                response.raise_for_status()  # Fehler auslösen bei Problemen

                total_size = int(response.headers.get('content-length', 0))
                with open(self.sam2_model_l_path, 'wb') as file, tqdm(
                    desc="SAM2 Large Download",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        bar.update(len(data))


                print(f"SAM2 model successfully downloaded:: {self.sam2_model_l_path}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading the SAM2 model: {e}")
                if os.path.exists(self.sam2_model_l_path): # Lösche unvollständige Datei bei Fehler
                    os.remove(self.sam2_model_l_path)
                return None # Signalisiert einen Fehler
            except Exception as e:
                print(f"An unexpected error has occurred during the download:{e}")
                if os.path.exists(self.sam2_model_l_path):
                    os.remove(self.sam2_model_l_path)
                return None

    # def load_sam2_model(self):
    #     """
    #     Loads the SAM2 model and prepares the image predictor for SAM2 tracking.
    #     This should be called once during initialization.
    #     """

    #     # needs to be fixed for unabhängige pfade
    #     try:
    #         absolute_config_yaml_path = "/home/janik/Documents/scripts/TagMed/TagMed/models/sam2_hiera_l.yaml"

    #         config_directory = os.path.dirname(absolute_config_yaml_path)
    #         config_file_basename = os.path.basename(absolute_config_yaml_path)
    #         config_name_for_hydra, _ = os.path.splitext(config_file_basename) # z.B. "sam2_hiera_l"

    #         config_path = "sam2_hiera_l"  # Passe den Pfad ggf. an
    #         config_file = "/home/janik/Documents/scripts/TagMed/TagMed/models/sam2_hiera_l.yaml"
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         print(f"[INFO] Loading SAM2 model on {device}")

    #         model = build_sam2(config_file=config_name_for_hydra, mode_path=self.sam2_model_l_path, device=device)

    #         self.sam2_predictor = SAM2ImagePredictor(model)
    #         print("[INFO] SAM2 model successfully loaded.")

    #     except Exception as e:
    #         import traceback
    #         print("[ERROR] Failed to load SAM2 model:")
    #         traceback.print_exc()
    #         self.sam2_predictor = None


    def load_sam2_video_model(self):
        """Lädt das SAM2 Video-Modell für Tracking"""
        try:
            from sam2.build_sam import build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            model_cfg = "/home/janik/Documents/scripts/TagMed/TagMed/models/sam2_hiera_l.yaml"
            sam2_checkpoint = "/home/janik/Documents/scripts/TagMed/TagMed/models/sam2.1_hiera_large.pt"
            
            # Image Predictor für initiale Maske
            self.sam2_image_predictor = SAM2ImagePredictor(build_sam2_video_predictor(model_cfg, sam2_checkpoint))
            
            # Video Predictor für Tracking
            self.sam2_video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
            
            print("[INFO] SAM2 Video model successfully loaded.")
            
        except Exception as e:
            print(f"[ERROR] Failed to load SAM2 Video model: {e}")


            
    def sam_2_tracking_method(self):
        """
        Uses SAM2 to propagate a bounding box across all following video frames.
        FIXED VERSION - removes critical bugs from previous implementation.
        """
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

        if not hasattr(self, "sam2_predictor") or self.sam2_predictor is None:
            self.load_sam2_model()
        predictor = self.sam2_predictor

        if is_polygon:
            print("[INFO] Polygon tracking not supported with SAM2")
            return

        # get bounding box data from data frame
        try:
            x_list = self._safe_parse_list(row.get('x'))
            y_list = self._safe_parse_list(row.get('y'))
            w_list = self._safe_parse_list(row.get('w'))
            h_list = self._safe_parse_list(row.get('h'))
            class_list = self._safe_parse_list(row.get('class'))

            x = x_list[selected_annotation_index]
            y = y_list[selected_annotation_index]
            w = w_list[selected_annotation_index]
            h = h_list[selected_annotation_index]
            slected_class = class_list[selected_annotation_index]

            # Initial input box - convert x,y,w,h values into sam2 format x0, y0, x1, y1
            resize_h, resize_w = self.image_size

            scale_x = resize_w / 1280
            scale_y = resize_h / 960


            input_box = self.center_to_corners(x, y, w, h)

            print(f"[DEBUG] Starting tracking with bbox: x={x}, y={y}, w={w}, h={h}")

        except IndexError:
            print(f"[ERROR] rect_id Index {selected_annotation_index} out of range.")
            return

        try:
            for i in range(current_frame_index + 1, len(current_frames)):
                next_img_id = current_frames[i].split(".")[0]
                frame_path = os.path.join(
                    self.selected_image_folder, 
                    self.gui.patient_id, 
                    self.gui.selected_exam, 
                    current_frames[i]
                )
                
                print(f"[DEBUG] Processing frame {i}: {next_img_id}")

                image_bgr = cv2.imread(frame_path)
                if image_bgr is None:
                    print(f"[WARN] Could not read image: {frame_path}")
                    continue
                originial_height, original_width = image_bgr.shape[:2]

                image_bgr = cv2.resize(image_bgr, (resize_w, resize_h))
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # height, width = image_bgr.shape[:2]
                # print(f"[DEBUG] Frame {next_img_id} loaded with shape: {image_bgr.shape}")
                predictor.set_image(image_rgb)

                try:
                    masks = predictor.predict(box=input_box, multimask_output=False)
                    if masks[0].sum() == 0:
                        print(f"[INFO] No segmentation for frame {next_img_id}")
                        continue
                except Exception as e:
                    print(f"[ERROR] SAM2 failed on frame {next_img_id}: {e}")
                    continue

                print(f"[DEBUG] predictor.predict result: {masks}")


                mask = masks[0]
                
                # 🎯 ERST prüfen ob die ursprüngliche Maske nicht leer ist
                ys_orig, xs_orig = np.where(mask.squeeze())
                
                if len(xs_orig) == 0 or len(ys_orig) == 0:
                    print(f"[WARN] Empty original mask for frame {frame_file}")
                    continue
                
                # Dann filtern auf relevanten Bereich
                margin = 50
                x1 = max(0, int(x - margin))
                y1 = max(0, int(y - margin)) 
                x2 = min(mask.shape[1], int(x + w + margin))
                y2 = min(mask.shape[0], int(y + h + margin))
                
                # Erstelle gefilterte Maske
                filtered_mask = np.zeros_like(mask)
                filtered_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                
                # Verwende die gefilterte Maske
                ys, xs = np.where(filtered_mask.squeeze())
                
                # Falls gefilterte Maske leer ist, verwende die ursprüngliche
                if len(xs) == 0 or len(ys) == 0:
                    print(f"[DEBUG] Filtered mask empty, using original mask")
                    ys, xs = ys_orig, xs_orig

                # Calculate new bounding box
                x0, y0 = xs.min(), ys.min()
                x1, y1 = xs.max(), ys.max()
                new_x, new_y, new_w, new_h = self.corners_to_center(x0, y0, x1, y1)
                
                print(f"[DEBUG] Frame {next_img_id}: New bbox: x={new_x}, y={new_y}, w={new_w}, h={new_h}")

                # CRITICAL FIX: Update input_box for next frame!
                input_box = self.center_to_corners(new_x, new_y, new_w, new_h)

                # Save annotation to DataFrame
                match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == next_img_id
                if match_next.any():
                    next_df_index = self.gui.all_annotations[match_next].index[0]

                    for col, val in zip(['x', 'y', 'w', 'h', 'class', 'bb_annotype'], 
                                      [new_x, new_y, new_w, new_h, slected_class, 'tracking']):
                        self.gui.all_annotations.at[next_df_index, col] = self._append_or_init_list(
                            self.gui.all_annotations.at[next_df_index, col], val)

                # Optional: Save mask
                # if hasattr(self.gui, 'all_masks'):
                #     mask_uint8 = (mask_binary.astype(np.uint8)) * 255
                #     self.gui.all_masks.append({
                #         'img_ID': next_img_id,
                #         'mask': mask_uint8
                #     })
                break
            print(f"[INFO] SAM2 Tracking completed for {len(current_frames) - current_frame_index - 1} frames.")

        except Exception as e:
            print(f"[ERROR] SAM2 Tracking Method could not be finished: {e}")
            import traceback
            traceback.print_exc()

    def extract_relevant_mask_region(self, mask, current_box, margin=20):
        """
        Extrahiert nur den relevanten Bereich der Maske um die aktuelle Bounding Box.
        
        Args:
            mask: Die vollständige SAM2-Maske
            current_box: Aktuelle Bounding Box [x, y, w, h]
            margin: Zusätzlicher Rand um die Box
        
        Returns:
            Gefilterte Maske mit nur dem relevanten Bereich
        """
        x, y, w, h = current_box
        
        # Erweitere die Box um einen Margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(mask.shape[1], x + w + margin)
        y2 = min(mask.shape[0], y + h + margin)
        
        # Erstelle eine leere Maske
        filtered_mask = np.zeros_like(mask)
        
        # Kopiere nur den relevanten Bereich
        filtered_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        
        return filtered_mask





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

                # searching for match for the next frame
                match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == img_id
                if match_next.any():
                    next_df_index = self.gui.all_annotations[match_next].index[0]

                    for col in ['x', 'y', 'w', 'h', 'class', 'bb_annotype', 'polygon', 'class_polygon', 'polygon_annoytype']:
                        self.gui.all_annotations.at[next_df_index, col] = "NN"
            
            self.video_annotation_handler.clear_all_masks()
            self.video_annotation_handler.clear_all_annotations()
            self.video_annotation_handler.delete_all_polygons()
            self.video_annotation_handler.delete_all_bounding_boxes()
            self.video_annotation_handler.update_video_listbox_with_annotation_colors()

        except:
            pass

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
    
    # initialize cells 
    def _append_or_init_list(self, cell, value):
        """ HelperFunction: Appends a value to a list in a DataFrame cell or initializes it if empty."""
        if isinstance(cell, float) and pd.isna(cell):
            return [value]
        if isinstance(cell, str) and cell == "NN":
            return [value]
        if isinstance(cell, list):
            return cell + [value]
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list):
                return parsed + [value]
        except:
            pass
        return [value]
    
    def center_to_corners(self, x, y, w, h):
        """
        Converts (x_center, y_center, width, height) to (x0, y0, x1, y1)
        """
        x0 = int(x - w / 2)
        y0 = int(y - h / 2)
        x1 = int(x + w / 2)
        y1 = int(y + h / 2)
        return np.array([x0, y0, x1, y1], dtype=np.float32)
    
    def corners_to_center(self, x0, y0, x1, y1):
        """
        Converts (x0, y0, x1, y1) to (x_center, y_center, width, height)
        """
        w = int(x1 - x0)
        h = int(y1 - y0)
        x_center = int(x0 + w / 2)
        y_center = int(y0 + h / 2)
        return x_center, y_center, w, h

 
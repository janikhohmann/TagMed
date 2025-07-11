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
from video_tracking import VideoTracking
from sam2_tracking import SAM2Tracking
from mask_handler import MaskHandler


class ImageMaskPredictor:
    def __init__(self, gui):
        self.gui = gui

        self.mask_handler = MaskHandler(gui)
        
        self.video_tracking = VideoTracking(gui)
        self.sam2_tracking = SAM2Tracking(gui)
        sam2_predictor = None


        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        image_size = config.get("image_size")
        self.resize_h, self.resize_w = image_size



    def predict_mask_for_image_bb(self, df_index, x,y,w,h):
        self.sam2_tracking.check_if_sam_2_is_available()


        input_box = self.center_to_corners(x, y, w, h)
        image_id = self.gui.selected_image_index.split(".")[0]
        frame_path = os.path.join(
            self.selected_image_folder, 
            self.gui.patient_id, 
            self.gui.selected_exam, 
            self.gui.selected_image_index
            )
        
        if not self.gui.img_annotation_listbox.curselection():
            annotation_index = self.gui.img_annotation_listbox.size() - 1
        else:
            selected_annotation = self.gui.img_annotation_listbox.curselection()
            annotation_index = selected_annotation[0]
        

        if not hasattr(self, "sam2_predictor") or self.sam2_predictor is None:
            self.sam2_tracking.load_sam2_model()
        predictor = self.sam2_tracking.sam2_predictor

        image_bgr = cv2.imread(frame_path)
        if image_bgr is None:
            print(f"[WARN] Could not read image: {frame_path}")
            return

        image_bgr = cv2.resize(image_bgr, (self.resize_w, self.resize_h))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


        predictor.set_image(image_rgb)

        try:
            masks, iou_preds, low_res_masks = predictor.predict(box=input_box, multimask_output=False)
    
            if masks[0].sum() == 0:
                print(f"[INFO] No segmentation for frame {image_id}")
                return
        except Exception as e:
            print(f"[ERROR] SAM2 failed on frame {image_id}: {e}")
            return

        mask = masks[0]

        path_mask = self.mask_handler.save_mask(mask, image_id, self.gui.img_selected_class.get(), annotation_index)

        if not self.gui.img_annotation_listbox.curselection() and df_index is not None: # mask path only has to be updated if no annotation is selected in the listbox
            self.gui.all_annotations.at[df_index, "masks"] = self._append_or_init_list(self.gui.all_annotations.at[df_index, "masks"], path_mask)
            return None 
        else:
            return path_mask
        
    def predict_mask_for_image_polygon(self, df_index, polygon, width, height):
        """
        Converts a polygon (list of x,y coordinates) to a binary mask.
        """
        image_id = self.gui.selected_image_index.split(".")[0]

        if not self.gui.img_annotation_listbox.curselection():
            annotation_index = self.gui.img_annotation_listbox.size() - 1
        else:
            selected_annotation = self.gui.img_annotation_listbox.curselection()
            annotation_index = selected_annotation[0]

        # Convert polygon to numpy array and reshape
        points = np.array(polygon).reshape(-1, 2).astype(np.int32)
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill polygon
        cv2.fillPoly(mask, [points], 1)

        path_mask = self.mask_handler.save_mask(mask, image_id, self.gui.img_selected_class.get(), annotation_index)

        if not self.gui.img_annotation_listbox.curselection() and df_index is not None: # mask path only has to be updated if no annotation is selected in the listbox
            self.gui.all_annotations.at[df_index, "masks"] = self._append_or_init_list(self.gui.all_annotations.at[df_index, "masks"], path_mask)
            return None 
        else:
            return path_mask
        





# ====== Helper Functions ======


    def center_to_corners(self, x, y, w, h):
        """
        Converts (x_center, y_center, width, height) to (x0, y0, x1, y1)
        """
        x0 = int(x - w / 2)
        y0 = int(y - h / 2)
        x1 = int(x + w / 2)
        y1 = int(y + h / 2)
        return np.array([x0, y0, x1, y1], dtype=np.float32)
    

    def _mask_to_polygon(self, binary_mask):
        """
        Converts a binary mask to a polygon (simplified contour).
        Returns a list of tuples with (x, y) coordinates.
        """
        import cv2
        
        # Convert to uint8 if needed
        if binary_mask.dtype != np.uint8:
            mask_uint8 = (binary_mask * 255).astype(np.uint8)
        else:
            mask_uint8 = binary_mask
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to reduce number of points --> is not used in the current implementation
        # Uncomment the following lines if you want to simplify the contour
        # epsilon = 0.01 * cv2.arcLength(contour, True)
        # contour = cv2.approxPolyDP(contour, epsilon, True)
        

        polygon_points = []
        for point in contour:
            x, y = point[0]  # OpenCV contour format: [[x, y]]
            polygon_points.append((int(x), int(y)))

        if len(polygon_points) > 0 and polygon_points[0] != polygon_points[-1]:
            polygon_points.append(polygon_points[0])  # Ensure the polygon is closed by adding the first point at the end

        print(len(polygon_points), "points in polygon")
        
        return polygon_points
    
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

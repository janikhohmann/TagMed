"""
Image Mask Predictor - SAM2 Image Segmentation Integration

This module provides advanced mask prediction capabilities using Meta's SAM2 
(Segment Anything 2) models for single image segmentation in medical annotation workflows.
It supports both bounding box and polygon-based prompting for precise mask generation.

Features:
- SAM2 image predictor integration
- Bounding box and polygon mask prediction
- Automatic model loading and configuration
- Integration with TagMed annotation workflow
- Mask persistence and management
- Multi-device support (CUDA, MPS, CPU)

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import os
import ast
import pandas as pd
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir


from config_handler import ConfigHandler
from mask_handler import MaskHandler
from sam2_tracking import SAM2Tracking


class ImageMaskPredictor:
    def __init__(self, gui):
        """
        Mask prediction for single images using SAM2 model.
        
        This class provides comprehensive mask generation capabilities for medical images
        using Meta's SAM2 architecture. It supports various prompting methods including
        bounding boxes and polygons, with automatic model management and integration
        into the TagMed annotation pipeline.
        
        Attributes:
            gui: Reference to the main GUI interface
            mask_handler (MaskHandler): Mask visualization and persistence management
            sam2_image_predictor: Loaded SAM2 image predictor instance
            sam2_model: Loaded SAM2 model for image prediction
            selected_image_folder (str): Base directory for patient images
            resize_h, resize_w (int): Display dimensions for image resizing
        """

        self.gui = gui
        self.mask_handler = MaskHandler(gui)
        
        model_dir = "../models"  # Directory where the model is saved
        self.abs_model_dir = os.path.abspath(model_dir)
        # set a fallback model
        self.sam2_model_path = os.path.join(self.abs_model_dir, "sam2.1_hiera_large.pt")
        self.config_path = "/home/janik/Documents/scripts/TagMed/TagMed/src/configs"
        self.config_name = "sam2.1_hiera_l"

        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        image_size = config.get("image_size")
        self.resize_h, self.resize_w = image_size

        self.sam2_tracking = SAM2Tracking(gui)



    def predict_mask_for_image_bb(self, df_index, x,y,w,h):
        """
        Predicts a mask for a single image using a bounding box input with SAM2.
        Args:
            df_index (int): Index of the DataFrame row for the current annotation
            x, y (float): Center coordinates of the bounding box
            w, h (float): Width and height of the bounding box
        Returns:
            str: Path to the saved mask file or None if no mask was generated
        """
        # Check if SAM2 is available
        self.sam2_tracking.check_if_sam_2_is_available()

        input_box = self.center_to_corners(x, y, w, h) # get bounding box in the format (x0, y0, x1, y1)

        # check if an image is selected in the GUI or frame is selected
        if self.gui.selected_image_index is not None:
            image_id = self.gui.selected_image_index.split(".")[0]
            file_name = self.gui.selected_image_index
        else:
            # If no image is selected, use the current frame
            image_id = self.gui.current_frames[self.gui.current_frame_index].split(".")[0]
            file_name = self.gui.current_frames[self.gui.current_frame_index]

        # print(self.gui.selected_image_index, "selected image index")
        # print(f"Predicting mask for image {image_id} with bounding box {input_box}")

        # Use VideoFrameExtractor to get correct frame path (handles temp directories)
        if hasattr(self.gui, 'video_frame_extractor') and hasattr(self.gui, 'selected_video_index') and self.gui.selected_video_index is not None:
            # print("[DEBUG] Using VideoFrameExtractor for frame path")
            
            # selected_video_index might be the video name directly, not an index
            if isinstance(self.gui.selected_video_index, str):
                current_video = self.gui.selected_video_index
                # print(f"[DEBUG] Current video (from string): {current_video}")
            else:
                # If it's actually a numeric index, get from listbox
                current_video = self.gui.video_listbox.get(self.gui.selected_video_index)
                # print(f"[DEBUG] Current video (from index): {current_video}")
            
            # For video frames, use VideoFrameExtractor to get correct path
            image_folder = os.path.join(self.selected_image_folder, self.gui.patient_id, self.gui.selected_exam)
            frame_path = self.gui.video_frame_extractor.get_frame_path(
                self.gui.patient_id, 
                self.gui.selected_exam, 
                current_video,
                file_name,
                image_folder
            )
        else:
            # print("[DEBUG] Using fallback path for regular images")
            # Fallback for regular images (non-video frames)
            frame_path = os.path.join(
                self.selected_image_folder, 
                self.gui.patient_id, 
                self.gui.selected_exam, 
                file_name
            )
        
        if not self.gui.img_annotation_listbox.curselection():
            annotation_index = self.gui.img_annotation_listbox.size()
        else:
            selected_annotation = self.gui.img_annotation_listbox.curselection()
            annotation_index = selected_annotation[0]
        
        # print(f"[ERROR SEARCH] Using frame path: {frame_path}")

        if not hasattr(self, "sam2_image_predictor") or self.sam2_image_predictor is None:
            # select the device for computation
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            print(f"using device: {device}")

            # Clean previous Hydra initialization
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            # Initialize Hydra config context
            with initialize_config_dir(config_dir=self.config_path, version_base=None):
                # Build the predictor (Hydra will now compose internally)
                self.sam2_model = build_sam2(self.config_name, self.sam2_model_path, device=device)

            self.sam2_image_predictor = SAM2ImagePredictor(self.sam2_model)

        image_bgr = cv2.imread(frame_path)
        if image_bgr is None:
            print(f"[WARN] Could not read image: {frame_path}")
            return

        image_bgr = cv2.resize(image_bgr, (self.resize_w, self.resize_h))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


        self.sam2_image_predictor.set_image(image_rgb)

        try:
            masks, iou_preds, low_res_masks = self.sam2_image_predictor.predict(box=input_box, multimask_output=False)
    
            if masks[0].sum() == 0:
                print(f"[INFO] No segmentation for frame {image_id}")
                return
        except Exception as e:
            print(f"[ERROR] SAM2 failed on frame {image_id}: {e}")
            return

        mask = masks[0]

        # Convert mask to polygon and save mask
        path_mask = self.mask_handler.save_mask(mask, image_id, self.gui.img_selected_class.get(), annotation_index)

        if not self.gui.img_annotation_listbox.curselection() and df_index is not None: # mask path only has to be updated if no annotation is selected in the listbox
            self.gui.all_annotations.at[df_index, "masks"] = self._append_or_init_list(self.gui.all_annotations.at[df_index, "masks"], path_mask)
            return None 
        else:
            return path_mask
        
    def predict_mask_for_image_polygon(self, df_index, polygon, width, height):
        """
        Converts a polygon (list of x,y coordinates) to a binary mask.
        Args:
            df_index (int): Index of the DataFrame row for the current annotation
            polygon (list): List of tuples with (x, y) coordinates of the polygon
            width (int): Width of the image
            height (int): Height of the image
        """

        # check if an image is selected in the GUI or frame is selected
        if self.gui.selected_image_index is not None:
            image_id = self.gui.selected_image_index.split(".")[0]
            file_name = self.gui.selected_image_index
        else:
            # If no image is selected, use the current frame
            image_id = self.gui.current_frames[self.gui.current_frame_index].split(".")[0]
            file_name = self.gui.current_frames[self.gui.current_frame_index]


        if not self.gui.img_annotation_listbox.curselection():
            annotation_index = self.gui.img_annotation_listbox.size()
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

"""
SAM2 Tracking - Segment Anything 2 Video Tracking Integration

This module provides video tracking capabilities using Meta's SAM2 (Segment Anything 2)
models for object tracking and segmentation in medical videos. It supports both
large and tiny SAM2 model variants with automatic downloading and configuration.

Features:
- SAM2 large and tiny model variants
- Automatic model downloading from Meta AI
- Video frame-by-frame tracking with temporal consistency
- Integration with TagMed annotation workflow
- Hydra configuration management for SAM2 parameters
- Advanced segmentation prompting (boxes, masks)

SAM2 Paper:
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import os
import numpy as np
import cv2
import requests
import ast
import torch
import pandas as pd
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from config_handler import ConfigHandler
from video_annotation_handler import VideoAnnotationHandler
from mask_handler import MaskHandler


class SAM2Tracking:
    """
    Video tracking and segmentation using Meta's SAM2 (Segment Anything 2) models.
    
    This class provides advanced video tracking capabilities using SAM2 large
    and tiny models. It handles model downloading, configuration, and frame-by-frame
    tracking with temporal consistency for medical video annotation workflows.
    
    Attributes:
        gui: Reference to the main GUI interface
        config_handler (ConfigHandler): Configuration management
        video_annotation_handler (VideoAnnotationHandler): Video annotation interface
        mask_handler (MaskHandler): Mask visualization and management
        sam2_predictor: Loaded SAM2 model instance
        inference_state: Video tracking state for temporal consistency
    """
    
    def __init__(self, gui):
        """
        Initialize SAM2 tracking with configuration and model setup.
        
        Sets up the tracking environment, model paths, and integrates with
        the annotation workflow. Prepares for downloading and loading
        appropriate SAM2 model variants (large or tiny).
        
        Args:
            gui: Main GUI interface reference for integration
        """
        self.gui = gui
        self.config_handler = ConfigHandler()
        self.video_annotation_handler = VideoAnnotationHandler(gui)
        self.mask_handler = MaskHandler(gui)

        # Load configuration settings
        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.selected_anno_table_file = config.get("selected_anno_table_file")
        self.selected_medical_report_file = config.get("selected_medical_report_file")
        self.class_list = config.get("class_list")
        self.image_size = config.get("image_size", (600, 600))  # Default image size if not set        

        # Model directory setup
        model_dir="../models"  # Directory where the model is saved
        self.abs_model_dir = os.path.abspath(model_dir)

        # Model instances
        self.sam2_predictor = None
        # self.inference_state = None  # For video tracking state




    def check_if_sam_2_is_available(self):
        """
        Configure SAM2 model based on selected tracking type.
        
        Sets the appropriate model URL, local path, and configuration
        based on the user's selection of SAM2 variant:
        - SAM2 large: High-accuracy model for detailed segmentation
        - SAM2 tiny: Lightweight model for faster processing
        """
        SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

        # Path to custom config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(script_dir, "configs") # Directory where the config files are saved

        # Determine model based on user selection
        # print(self.gui.tracking_type.get())
        if self.gui.tracking_type.get() == "SAM2 large":
            self.sam2p1_model_path = os.path.join(self.abs_model_dir, "sam2.1_hiera_large.pt")
            self.sam2p1_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"
            self.config_name = "sam2.1_hiera_l.yaml"
        if self.gui.tracking_type.get() == "SAM2 tiny":
            self.sam2p1_model_path = os.path.join(self.abs_model_dir, "sam2.1_hiera_tiny.pt")
            self.sam2p1_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
            self.config_name = "sam2.1_hiera_t.yaml"

        # catch for image segmentation - fallback to large
        if self.gui.tracking_type.get() not in ["SAM2 large", "SAM2 tiny"]:
            self.sam2p1_model_path = os.path.join(self.abs_model_dir, "sam2.1_hiera_large.pt")
            self.sam2p1_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"
            self.config_name = "sam2.1_hiera_l.yaml"

        # Other SAM 2.1 checkpoints can be defined here
        # self.sam2p1_hiera_s_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
        # self.sam2p1_hiera_b_plus_url= f"{SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"

        if os.path.exists(self.sam2p1_model_path):
            return True
        else:
            load = self.gui.ask_for_sam2_download()
            if load:
                self.download_sam_2_model()
                return True
            else:
                return False

    def download_sam_2_model(self):
        # Ensure that the target directory exists
        os.makedirs(self.abs_model_dir, exist_ok=True)

        # Check if the file already exists
        if os.path.exists(self.sam2p1_model_path):
            print(f"SAM2 model already exists at: {self.sam2p1_model_path}")
        else:
            print(f"Trying to download SAM2 model to: {self.sam2p1_model_path}")
            try:
                response = requests.get(self.sam2p1_url, stream=True)
                response.raise_for_status()  # Raise errors on problems
                print(f"Downloading SAM2 model from {self.sam2p1_url}...")

                total_size = int(response.headers.get('content-length', 0))
                import tqdm
                with open(self.sam2p1_model_path, 'wb') as file, tqdm.tqdm(
                    desc="SAM2 Download",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        bar.update(len(data))
                        


                print(f"SAM2 model successfully downloaded:: {self.sam2p1_model_path}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading the SAM2 model: {e}")
                if os.path.exists(self.sam2p1_model_path): # Delete incomplete file on error
                    os.remove(self.sam2p1_model_path)
                return None # Signals an error
            except Exception as e:
                print(f"An unexpected error has occurred during the download:{e}")
                if os.path.exists(self.sam2p1_model_path):
                    os.remove(self.sam2p1_model_path)
                return None


    def load_sam2_model(self):
        """
        Loads the SAM2 video predictor for video tracking.
        Using Hydra to load the custom config file.
        """
        try:
            from sam2.build_sam import build_sam2_video_predictor
            
            # Disable torch compilation and inductor optimizations that cause hanging
            torch._dynamo.config.disable = True
            torch._inductor.config.disable_progress = True
            torch._inductor.config.triton.unique_kernel_names = True
            
            # Set float32 precision for better compatibility
            torch.set_float32_matmul_precision('high')

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Loading SAM2 video predictor on {device}")

            # Check if model file exists
            if not os.path.exists(self.sam2p1_model_path):
                print(f"[ERROR] SAM2 model file not found at: {self.sam2p1_model_path}")
                print("[INFO] Please ensure the SAM2 model is downloaded.")
                self.sam2_predictor = None
                return

            # Clean previous Hydra initializations
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()


            print(f"[INFO] Loading SAM2 from checkpoint: {self.sam2p1_model_path}")
            print(f"[INFO] Using config file: {self.config_name}")

            # Initialize Hydra config context
            with initialize_config_dir(config_dir=self.config_path, version_base=None):
                # Build the predictor (Hydra will now compose internally)
                self.sam2_predictor = build_sam2_video_predictor(
                    config_file=self.config_name,
                    ckpt_path=self.sam2p1_model_path,
                    apply_postprocessing=True,
                    vos_optimized=True,
                )

            #self.inference_state = None
            print("[INFO] SAM2 video predictor successfully loaded.")

        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to load SAM2 video predictor: {e}")
            traceback.print_exc()
            self.sam2_predictor = None

            
    def sam2_tracking_method(self):
        """
        Uses SAM2 video predictor to propagate annotations across all following video frames.
        """
        selected_annotation = self.gui.video_annotation_listbox.curselection()
        selected_annotation_index = selected_annotation[0]
        current_frames = self.gui.current_frames
        current_frame_index = self.gui.current_frame_index
        current_image_id = current_frames[current_frame_index].split(".")[0]
        selected_class = self.gui.video_selected_class.get()

        # Find matching row in DataFrame
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == current_image_id
        if not match.any():
            print(f"[ERROR] No entry in database for {current_image_id}")
            return
        df_index = self.gui.all_annotations[match].index[0]
        row = self.gui.all_annotations.loc[df_index]

        # Check if we have the video predictor loaded
        if not hasattr(self, "sam2_predictor") or self.sam2_predictor is None:
            # print("[INFO] SAM2 predictor not loaded, attempting to load...")
            self.load_sam2_model()
        predictor = self.sam2_predictor

        # Verify predictor is loaded after attempt
        if self.sam2_predictor is None:
            print("[ERROR] Failed to load SAM2 predictor. Cannot proceed with tracking.")
            return
        
        # Initialize video sequence - use VideoFrameExtractor for correct frame paths
        video_folder = os.path.join(self.selected_image_folder, self.gui.patient_id, self.gui.selected_exam)
        temp_video_dir = self._create_temp_video_directory_with_extractor(video_folder, current_frames)

        try:
            resize_h, resize_w = self.image_size
            
            # Track prompts per video for multi-prompt support
            video_id = f"{self.gui.patient_id}_{self.gui.selected_exam}_{self.gui.selected_video_index}"
            
            if not hasattr(self, 'sam2_video_prompts'):
                self.sam2_video_prompts = {}  # video_id -> list of prompts
            if not hasattr(self, 'sam2_temp_dirs'):
                self.sam2_temp_dirs = {}  # video_id -> temp_dir path
            
            # Initialize temp dir tracking
            if video_id not in self.sam2_temp_dirs:
                # First time: keep the newly created temp_dir
                self.sam2_temp_dirs[video_id] = temp_video_dir
            else:
                # Correction: old temp_dir might have symlinks, we need actual JPEGs
                # Clean up the old one and use the new one with proper JPEGs
                old_temp_dir = self.sam2_temp_dirs[video_id]
                if old_temp_dir != temp_video_dir:
                    print(f"[DEBUG] Cleaning up old temp directory: {old_temp_dir}")
                    self._cleanup_temp_directory(old_temp_dir)
                # Update to new temp_dir
                self.sam2_temp_dirs[video_id] = temp_video_dir
                print(f"[DEBUG] Using new temp directory: {temp_video_dir}")
            
            # Initialize prompt list for this video
            if video_id not in self.sam2_video_prompts:
                self.sam2_video_prompts[video_id] = []
                print(f"[INFO] Initializing SAM2 for new video...")
            else:
                print(f"[INFO] Adding correction prompt to existing video (total prompts: {len(self.sam2_video_prompts[video_id]) + 1})...")
            

            # Check if Polygon or Bounding Box
            selected_text = self.gui.video_annotation_listbox.get(selected_annotation_index)
            is_polygon = "Polygon" in selected_text

            # SAM2 uses 1-based object IDs
            ann_obj_id = selected_annotation_index + 1
            
            # Store current prompt information
            current_prompt = {
                'frame_idx': current_frame_index,
                'ann_obj_id': ann_obj_id,
                'is_polygon': is_polygon,
                'coords': None  # Will be set below
            }

            

            # ===== POLYGON TRACKING =====
            if is_polygon:
                print("[INFO] Starting SAM2 video polygon tracking...")
                
                try:
                    # Get polygon data from dataframe
                    polygon_list = self._safe_parse_list(row.get('polygon'))
                    polygon_class_list = self._safe_parse_list(row.get('class_polygon'))
                        
                    polygon = polygon_list[selected_annotation_index]
                    selected_class = polygon_class_list[selected_annotation_index]

                    if not isinstance(polygon, list) or len(polygon) < 3:
                        print("[ERROR] Invalid polygon data - need at least 3 points.")
                        self._cleanup_temp_directory(temp_video_dir)
                        return

                    # Convert polygon to points for SAM2 (using polygon centroid as positive click)
                    points_array = np.array(polygon).reshape(-1, 2)
                    
                    # Scale polygon coordinates from GUI size to original frame size
                    original_frame_path = self._get_frame_path(video_folder, current_frames[current_frame_index])
                    if os.path.exists(original_frame_path):
                        import cv2
                        original_frame = cv2.imread(original_frame_path)
                        original_height, original_width = original_frame.shape[:2]
                        
                        # Scale coordinates from GUI size to original size
                        scale_x = original_width / resize_w
                        scale_y = original_height / resize_h
                        
                        # Scale all polygon points
                        scaled_points = points_array.copy().astype(np.float64)  # Convert to float for scaling
                        scaled_points[:, 0] *= scale_x  # Scale x coordinates
                        scaled_points[:, 1] *= scale_y  # Scale y coordinates
                        
                        centroid_x = int(np.mean(scaled_points[:, 0]))
                        centroid_y = int(np.mean(scaled_points[:, 1]))
                        
                        # print(f"[DEBUG] Original polygon centroid: ({np.mean(points_array[:, 0])}, {np.mean(points_array[:, 1])})")
                        # print(f"[DEBUG] Scaled polygon centroid: ({centroid_x}, {centroid_y})")
                        # print(f"[DEBUG] Scale factors: scale_x={scale_x}, scale_y={scale_y}")
                    else:
                        centroid_x = int(np.mean(points_array[:, 0]))
                        centroid_y = int(np.mean(points_array[:, 1]))
                    
                    # Add the click at the centroid
                    points = np.array([[centroid_x, centroid_y]], dtype=np.float32)
                    labels = np.array([1], np.int32)  # Positive click
                    
                    # Store prompt for later (will be added after state init)
                    current_prompt['coords'] = {
                        'points': points,
                        'labels': labels
                    }
                    
                    print(f"[INFO] Stored polygon prompt on frame {current_frame_index} (obj_id={ann_obj_id})")


                except IndexError:
                    print(f"[ERROR] Polygon index {selected_annotation_index} out of range.")
                    self._cleanup_temp_directory(temp_video_dir)
                    return

            else:
                # ===== BOUNDING BOX TRACKING =====
                print("[INFO] Starting SAM2 video bounding box tracking...")
                
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
                    selected_class = class_list[selected_annotation_index]

                    # Convert to SAM2 box format (x0, y0, x1, y1)
                    # Scale coordinates from GUI size to original frame size
                    original_frame_path = self._get_frame_path(video_folder, current_frames[current_frame_index])
                    if os.path.exists(original_frame_path):
                        import cv2
                        original_frame = cv2.imread(original_frame_path)
                        original_height, original_width = original_frame.shape[:2]
                        
                        # Scale coordinates from GUI size to original size
                        scale_x = original_width / resize_w
                        scale_y = original_height / resize_h
                        
                        scaled_x = x * scale_x
                        scaled_y = y * scale_y
                        scaled_w = w * scale_x
                        scaled_h = h * scale_y
                        
                        # print(f"[DEBUG] Original coords: x={x}, y={y}, w={w}, h={h}")
                        # print(f"[DEBUG] Scaled coords: x={scaled_x}, y={scaled_y}, w={scaled_w}, h={scaled_h}")
                        # print(f"[DEBUG] Scale factors: scale_x={scale_x}, scale_y={scale_y}")
                        # print(f"[DEBUG] Original frame size: {original_width}x{original_height}, GUI size: {resize_w}x{resize_h}")
                        
                        input_box = self.center_to_corners(scaled_x, scaled_y, scaled_w, scaled_h)
                    else:
                        input_box = self.center_to_corners(x, y, w, h)

                    # print(f"[DEBUG] Starting tracking with bbox: x={x}, y={y}, w={w}, h={h}")
                    # print(f"[DEBUG] SAM2 input_box: {input_box}")
                    
                    # Store prompt for later (will be added after state init)
                    current_prompt['coords'] = {
                        'box': input_box
                    }
                    
                    print(f"[INFO] Stored bounding box prompt on frame {current_frame_index} (obj_id={ann_obj_id})")


                except IndexError:
                    print(f"[ERROR] Bounding box index {selected_annotation_index} out of range.")
                    self._cleanup_temp_directory(temp_video_dir)
                    return

            # Add current prompt to list
            self.sam2_video_prompts[video_id].append(current_prompt)
            
            # Reinitialize inference state to include all prompts
            # Use the stored temp_dir (not the potentially deleted one)
            actual_temp_dir = self.sam2_temp_dirs[video_id]
            print(f"[INFO] Initializing SAM2 state with {len(self.sam2_video_prompts[video_id])} prompt(s)...")
            print(f"[DEBUG] Using temp directory: {actual_temp_dir}")
            try:
                self.gui.inference_state = predictor.init_state(video_path=actual_temp_dir)
            except Exception as e:
                print(f"[ERROR] Failed to initialize inference state: {e}")
                self._cleanup_temp_directory(temp_video_dir)
                return
            
            # Add all prompts to the fresh state
            for prompt_idx, prompt in enumerate(self.sam2_video_prompts[video_id]):
                print(f"[DEBUG] Adding prompt {prompt_idx + 1}/{len(self.sam2_video_prompts[video_id])} on frame {prompt['frame_idx']}")
                
                if prompt['is_polygon']:
                    # Add polygon points
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=self.gui.inference_state,
                        frame_idx=prompt['frame_idx'],
                        obj_id=prompt['ann_obj_id'],
                        points=prompt['coords']['points'],
                        labels=prompt['coords']['labels'],
                    )
                else:
                    # Add bounding box
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=self.gui.inference_state,
                        frame_idx=prompt['frame_idx'],
                        obj_id=prompt['ann_obj_id'],
                        box=prompt['coords']['box'],
                    )

            # ===== PROPAGATE THROUGH VIDEO =====
            print("[INFO] Propagating annotations through video with SAM2...")
            
            # Collect results in a dict
            video_segments = {}
            try:
                print("[DEBUG] Starting propagate_in_video()...")
                frame_count = 0
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.gui.inference_state):
                    frame_count += 1
                    #if frame_count % 10 == 0:  # Progress indicator every 10 frames
                        #print(f"[DEBUG] Processed {frame_count} frames...")
                    
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                print(f"[DEBUG] propagate_in_video() completed successfully. Processed {frame_count} frames.")
            except Exception as e:
                print(f"[ERROR] Failed during video propagation: {e}")
                import traceback
                traceback.print_exc()
                self._cleanup_temp_directory(temp_video_dir)
                return
            # Process results for all frames (corrections should update entire video)
            frames_processed = 0
            for frame_idx in range(0, len(current_frames)):
                if frame_idx in video_segments and ann_obj_id in video_segments[frame_idx]:
                    mask = video_segments[frame_idx][ann_obj_id]
                    next_img_id = current_frames[frame_idx].split(".")[0]
                    
                    # Ensure mask is 2D (remove any extra dimensions)
                    if mask.ndim > 2:
                        mask = mask.squeeze()  # Remove dimensions of size 1
                    
                    # Skip empty masks and delete existing mask file if present
                    if mask.sum() == 0:
                        print(f"[INFO] Empty mask for frame {next_img_id} - removing existing mask if present")
                        
                        # Find and delete existing mask file
                        match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == next_img_id
                        if match_next.any():
                            next_df_index = self.gui.all_annotations[match_next].index[0]
                            existing_masks = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, 'masks'])
                            
                            if selected_annotation_index < len(existing_masks):
                                mask_path = existing_masks[selected_annotation_index]
                                if mask_path and os.path.exists(mask_path):
                                    try:
                                        os.remove(mask_path)
                                        print(f"[INFO] Deleted mask file: {mask_path}")
                                    except Exception as e:
                                        print(f"[WARN] Could not delete mask file {mask_path}: {e}")
                                
                                # Remove mask from list
                                existing_masks[selected_annotation_index] = None
                                self.gui.all_annotations.at[next_df_index, 'masks'] = existing_masks
                        
                        continue

                    # Scale mask to GUI size for consistency with annotations
                    if mask.shape != (resize_h, resize_w):
                        # Scale mask to GUI size
                        mask_for_gui = cv2.resize(mask.astype(np.uint8), (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
                        path_mask = self.mask_handler.save_mask(mask_for_gui, next_img_id, selected_class, selected_annotation_index)
                    else:
                        path_mask = self.mask_handler.save_mask(mask, next_img_id, selected_class, selected_annotation_index)
                    
                    # Update database for frames after the initial frame
                    if frame_idx != current_frame_index:
                        # Find or create entry for this frame
                        match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == next_img_id
                        if match_next.any():
                            next_df_index = self.gui.all_annotations[match_next].index[0]
                            
                            if is_polygon:
                                # Convert mask back to polygon - scale to GUI coordinates
                                generated_polygon = self._mask_to_polygon(mask.squeeze())
                                
                                # Scale polygon coordinates back to GUI size
                                if generated_polygon:
                                    original_frame_path = self._get_frame_path(video_folder, current_frames[frame_idx])
                                    if os.path.exists(original_frame_path):
                                        original_frame = cv2.imread(original_frame_path)
                                        original_height, original_width = original_frame.shape[:2]
                                        
                                        # Scale coordinates back from original size to GUI size
                                        scale_x = resize_w / original_width
                                        scale_y = resize_h / original_height
                                        
                                        scaled_polygon = []
                                        for point in generated_polygon:
                                            scaled_x = int(point[0] * scale_x)
                                            scaled_y = int(point[1] * scale_y)
                                            scaled_polygon.append((scaled_x, scaled_y))
                                        generated_polygon = scaled_polygon
                                
                                # Update polygon data
                                existing_polygons = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, 'polygon'])
                                existing_class_polygons = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, 'class_polygon'])
                                existing_polygon_annotypes = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, 'polygon_annotype'])
                                
                                # Ensure lists are long enough
                                while len(existing_polygons) <= selected_annotation_index:
                                    existing_polygons.append([])
                                    existing_class_polygons.append("")
                                    existing_polygon_annotypes.append("")
                                
                                # Update at the specific index
                                existing_polygons[selected_annotation_index] = generated_polygon
                                existing_class_polygons[selected_annotation_index] = selected_class
                                existing_polygon_annotypes[selected_annotation_index] = "tracking"
                                
                                # Save back to dataframe
                                self.gui.all_annotations.at[next_df_index, 'polygon'] = existing_polygons
                                self.gui.all_annotations.at[next_df_index, 'class_polygon'] = existing_class_polygons
                                self.gui.all_annotations.at[next_df_index, 'polygon_annotype'] = existing_polygon_annotypes
                                
                            else:
                                # Update bounding box data - scale back to GUI coordinates
                                ys, xs = np.where(mask.squeeze())
                                if len(xs) > 0 and len(ys) > 0:
                                    x0, y0 = xs.min(), ys.min()
                                    x1, y1 = xs.max(), ys.max()
                                    
                                    # Scale coordinates back to GUI size
                                    original_frame_path = self._get_frame_path(video_folder, current_frames[frame_idx])
                                    if os.path.exists(original_frame_path):
                                        import cv2
                                        original_frame = cv2.imread(original_frame_path)
                                        original_height, original_width = original_frame.shape[:2]
                                        
                                        # Scale coordinates back from original size to GUI size
                                        scale_x = resize_w / original_width
                                        scale_y = resize_h / original_height
                                        
                                        scaled_x0 = x0 * scale_x
                                        scaled_y0 = y0 * scale_y
                                        scaled_x1 = x1 * scale_x
                                        scaled_y1 = y1 * scale_y
                                        
                                        new_x, new_y, new_w, new_h = self.corners_to_center(scaled_x0, scaled_y0, scaled_x1, scaled_y1)
                                    else:
                                        new_x, new_y, new_w, new_h = self.corners_to_center(x0, y0, x1, y1)

                                    for col, val in zip(['x', 'y', 'w', 'h', 'class', 'bb_annotype'], 
                                                    [new_x, new_y, new_w, new_h, selected_class, 'tracking']):
                                        existing_list = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, col])

                                        # Ensure list is long enough
                                        while len(existing_list) <= selected_annotation_index:
                                            existing_list.append(None)
                                        
                                        existing_list[selected_annotation_index] = val
                                        self.gui.all_annotations.at[next_df_index, col] = existing_list

                            # Update masks
                            self.gui.all_annotations.at[next_df_index, "masks"] = self._append_or_init_list(
                                self.gui.all_annotations.at[next_df_index, "masks"], path_mask)
                    else:
                        # Save mask for initial frame
                        self.gui.all_annotations.at[df_index, "masks"] = self._append_or_init_list(
                            self.gui.all_annotations.at[df_index, "masks"], path_mask)
                    
                    frames_processed += 1

            print(f"[INFO] SAM2 Video Tracking completed for {frames_processed} frames.")

        except Exception as e:
            print(f"[ERROR] SAM2 Video Tracking could not be finished: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Keep temp_dir for future corrections - it will be reused
            # Only cleanup when user changes videos or closes the application
            pass
    

    def _get_frame_path(self, video_folder, frame_filename):
        """
        Uses VideoFrameExtractor to obtain the correct path to a frame.
        
        Args:
            video_folder (str): Base folder for videos/frames
            frame_filename (str): Name of the frame file
            
        Returns:
            str: Full path to the frame
        """
        if hasattr(self.gui, 'video_frame_extractor'):
            return self.gui.video_frame_extractor.get_frame_path(
                patient_id=self.gui.patient_id,
                selected_exam=self.gui.selected_exam,
                selected_video=self.gui.selected_video_index,
                frame_filename=frame_filename,
                image_folder=video_folder
            )
        else:
            # Fallback to old method
            return os.path.join(video_folder, frame_filename)

    def _create_temp_video_directory_with_extractor(self, video_folder, current_frames):
        """
        Creates a temporary directory with JPEG copies of video frames.
        SAM2 expects frames with names like 00000.jpg, 00001.jpg, etc.
        """
        import tempfile
        import cv2
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="sam2_video_")
        print(f"[INFO] Created temporary directory: {temp_dir}")
        
        try:
            # Create JPEG copies for each frame with the expected naming format
            for i, frame_name in enumerate(current_frames):
                # Use VideoFrameExtractor for correct path
                source_path = self._get_frame_path(video_folder, frame_name)
                
                # SAM2 expects frame names like 00000.jpg, 00001.jpg, etc.
                target_name = f"{i:05d}.jpg"
                target_path = os.path.join(temp_dir, target_name)
                
                if os.path.exists(source_path):
                    # Read and save as JPEG (SAM2 requires actual JPEG files, not PNG symlinks)
                    frame = cv2.imread(source_path)
                    if frame is not None:
                        cv2.imwrite(target_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else:
                        print(f"[WARN] Could not read frame: {source_path}")
                else:
                    print(f"[WARN] Source frame not found: {source_path}")
            
            return temp_dir
            
        except Exception as e:
            print(f"[ERROR] Failed to create temp video directory: {e}")
            # Cleanup on error
            self._cleanup_temp_directory(temp_dir)
            raise
    
    def _cleanup_temp_directory(self, temp_dir):
        """
        Removes the temporary directory and all its contents.
        """
        import shutil
        
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[INFO] Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"[WARN] Failed to cleanup temporary directory {temp_dir}: {e}")

# ====== Helper Functions ======

    def _safe_parse_list(self, value):
        """Helper function for safely parsing lists from strings or lists."""
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

    def _polygon_to_mask(self, polygon, width, height):
        """
        Converts a polygon (list of x,y coordinates) to a binary mask.
        """
        
        # Convert polygon to numpy array and reshape
        points = np.array(polygon).reshape(-1, 2).astype(np.int32)
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill polygon
        cv2.fillPoly(mask, [points], 1)

        return mask.astype(np.float32)


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
        epsilon = 0.005 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        

        polygon_points = []
        for point in contour:
            x, y = point[0]  # OpenCV contour format: [[x, y]]
            polygon_points.append((int(x), int(y)))

        if len(polygon_points) > 0 and polygon_points[0] != polygon_points[-1]:
            polygon_points.append(polygon_points[0])  # Ensure the polygon is closed by adding the first point at the end

        print(len(polygon_points), "points in polygon")
        
        return polygon_points
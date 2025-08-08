import os
import numpy as np
import cv2
import requests
import ast
import tqdm
import torch
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from config_handler import ConfigHandler
from video_annotation_handler import VideoAnnotationHandler
from mask_handler import MaskHandler


class SAM2Tracking:
    def __init__(self, gui):
        self.gui = gui
        self.config_handler = ConfigHandler()
        self.video_annotation_handler = VideoAnnotationHandler(gui)
        self.mask_handler = MaskHandler(gui)

        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.selected_anno_table_file = config.get("selected_anno_table_file")
        self.selected_medical_report_file = config.get("selected_medical_report_file")
        self.class_list = config.get("class_list")
        self.image_size = config.get("image_size", (600, 600))  # Default image size if not set


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
        self.inference_state = None  # For video tracking state




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
            print(f"SAM2 model already exists at: {self.sam2_model_l_path}")
        else:
            print(f"Trying to download SAM2 model to: {self.sam2_model_l_path}")
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
                if os.path.exists(self.sam2_model_l_path): # Delete incomplete file on error
                    os.remove(self.sam2_model_l_path)
                return None # Signals an error
            except Exception as e:
                print(f"An unexpected error has occurred during the download:{e}")
                if os.path.exists(self.sam2_model_l_path):
                    os.remove(self.sam2_model_l_path)
                return None


    def load_sam2_model(self):
        """
        Loads the MedSAM2 video predictor for video tracking.
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
            print(f"[INFO] Loading MedSAM2 video predictor on {device}")

            # Check if model file exists
            if not os.path.exists(self.sam2_model_l_path):
                print(f"[ERROR] MedSAM2 model file not found at: {self.sam2_model_l_path}")
                print("[INFO] Please ensure the MedSAM2 model is downloaded.")
                self.sam2_predictor = None
                return

            # Clean previous Hydra initialization
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            # Path to custom config
            config_path = "/home/janik/Documents/scripts/TagMed/TagMed/src/configs"
            config_name = "sam2.1_hiera_l.yaml"

            print(f"[INFO] Loading SAM2 from checkpoint: {self.sam2_model_l_path}")
            print(f"[INFO] Using config file: {config_name}")

            # Initialize Hydra config context
            with initialize_config_dir(config_dir=config_path, version_base=None):
                # Build the predictor (Hydra will now compose internally)
                self.sam2_predictor = build_sam2_video_predictor(
                    config_file=config_name,
                    ckpt_path=self.sam2_model_l_path,
                    apply_postprocessing=True,
                    vos_optimized=True,
                )

            self.inference_state = None
            print("[INFO] SAM2 video predictor successfully loaded.")

        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to load SAM2 video predictor: {e}")
            traceback.print_exc()
            self.sam2_predictor = None

    # def load_sam2_model(self):
    #     """
    #     Loads the SAM2 video predictor for video tracking.
    #     This should be called once during initialization.
    #     """
    #     try:
    #         from sam2.build_sam import build_sam2_video_predictor
            
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         print(f"[INFO] Loading SAM2 video predictor on {device}")
            
    #         # Set device-specific configurations
    #         if device == "cuda":
    #             # use bfloat16 for the entire workflow
    #             torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    #             # turn on tfloat32 for Ampere GPUs
    #             if torch.cuda.get_device_properties(0).major >= 8:
    #                 torch.backends.cuda.matmul.allow_tf32 = True
    #                 torch.backends.cudnn.allow_tf32 = True
    #         elif device == "mps":
    #             print(
    #                 "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
    #                 "give numerically different outputs and sometimes degraded performance on MPS."
    #             )
            
    #         # Use the downloaded model checkpoint
    #         sam2_checkpoint = self.sam2_model_l_path
    #         model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
    #         self.sam2_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    #         self.inference_state = None  # Will be initialized per video sequence
    #         print("[INFO] SAM2 video predictor successfully loaded.")

    #     except Exception as e:
    #         import traceback
    #         print(f"[ERROR] Failed to load SAM2 video predictor: {e}")
    #         traceback.print_exc()
    #         self.sam2_predictor = None


            
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
            self.load_sam2_model()
        predictor = self.sam2_predictor

        # Initialize video sequence - create a temporary directory with frames
        video_path = os.path.join(self.selected_image_folder, self.gui.patient_id, self.gui.selected_exam)
        temp_video_dir = self._create_temp_video_directory(video_path, current_frames)

        try:
            # Reset any previous state and initialize for this video sequence
            if hasattr(self, 'inference_state') and self.inference_state is not None:
                try:
                    predictor.reset_state(self.inference_state)
                except:
                    pass  # Ignore reset errors
            
            resize_h, resize_w = self.image_size

            print(f"[INFO] Initializing MedSAM2 video predictor for temp video path: {temp_video_dir}")
            print(f"[DEBUG] GUI image size: {resize_w}x{resize_h}")
            print(f"[DEBUG] Number of frames: {len(current_frames)}")
            print(f"[DEBUG] Current frame index: {current_frame_index}")
            
            try:
                print("[DEBUG] Calling predictor.init_state()...")
                self.inference_state = predictor.init_state(video_path=temp_video_dir)
                print("[DEBUG] init_state() completed successfully")
            except Exception as e:
                print(f"[ERROR] Failed to initialize inference state: {e}")
                self._cleanup_temp_directory(temp_video_dir)
                return
            
        
        # # Reset any previous state and initialize for this video sequence
        # if hasattr(self, 'inference_state') and self.inference_state is not None:
        #     predictor.reset_state(self.inference_state)
        
        # resize_h, resize_w = self.image_size

        # print(f"[INFO] Initializing SAM2 video predictor for temp video path: {temp_video_dir}")
        # print(f"[DEBUG] GUI image size: {resize_w}x{resize_h}")
        # print(f"[DEBUG] Number of frames: {len(current_frames)}")
        # print(f"[DEBUG] Current frame index: {current_frame_index}")
        # self.inference_state = predictor.init_state(video_path=temp_video_dir)

        # Check if Polygon or Bounding Box
            selected_text = self.gui.video_annotation_listbox.get(selected_annotation_index)
            is_polygon = "Polygon" in selected_text

            

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
                    original_frame_path = os.path.join(video_path, current_frames[current_frame_index])
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
                        
                        print(f"[DEBUG] Original polygon centroid: ({np.mean(points_array[:, 0])}, {np.mean(points_array[:, 1])})")
                        print(f"[DEBUG] Scaled polygon centroid: ({centroid_x}, {centroid_y})")
                        print(f"[DEBUG] Scale factors: scale_x={scale_x}, scale_y={scale_y}")
                    else:
                        centroid_x = int(np.mean(points_array[:, 0]))
                        centroid_y = int(np.mean(points_array[:, 1]))
                    
                    # Add the click at the centroid
                    ann_obj_id = selected_annotation_index + 1  # Object IDs should be > 0
                    points = np.array([[centroid_x, centroid_y]], dtype=np.float32)
                    labels = np.array([1], np.int32)  # Positive click
                    
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=current_frame_index,
                        obj_id=ann_obj_id,
                        points=points,
                        labels=labels,
                    )

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
                    original_frame_path = os.path.join(video_path, current_frames[current_frame_index])
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
                        
                        print(f"[DEBUG] Original coords: x={x}, y={y}, w={w}, h={h}")
                        print(f"[DEBUG] Scaled coords: x={scaled_x}, y={scaled_y}, w={scaled_w}, h={scaled_h}")
                        print(f"[DEBUG] Scale factors: scale_x={scale_x}, scale_y={scale_y}")
                        print(f"[DEBUG] Original frame size: {original_width}x{original_height}, GUI size: {resize_w}x{resize_h}")
                        
                        input_box = self.center_to_corners(scaled_x, scaled_y, scaled_w, scaled_h)
                    else:
                        input_box = self.center_to_corners(x, y, w, h)
                    
                    ann_obj_id = selected_annotation_index + 1  # Object IDs should be > 0

                    print(f"[DEBUG] Starting tracking with bbox: x={x}, y={y}, w={w}, h={h}")
                    print(f"[DEBUG] SAM2 input_box: {input_box}")
                    
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=current_frame_index,
                        obj_id=ann_obj_id,
                        box=input_box,
                    )

                except IndexError:
                    print(f"[ERROR] Bounding box index {selected_annotation_index} out of range.")
                    self._cleanup_temp_directory(temp_video_dir)
                    return

            # ===== PROPAGATE THROUGH VIDEO =====
            print("[INFO] Propagating annotations through video...")
            
            # Collect results in a dict
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Process results for frames starting from current frame
            frames_processed = 0
            for frame_idx in range(current_frame_index, len(current_frames)):
                if frame_idx in video_segments and ann_obj_id in video_segments[frame_idx]:
                    mask = video_segments[frame_idx][ann_obj_id]
                    next_img_id = current_frames[frame_idx].split(".")[0]
                    
                    # Ensure mask is 2D (remove any extra dimensions)
                    if mask.ndim > 2:
                        mask = mask.squeeze()  # Remove dimensions of size 1
                    
                    # Skip empty masks
                    if mask.sum() == 0:
                        print(f"[INFO] Empty mask for frame {next_img_id}")
                        continue

                    # Save mask (ensure it's 2D before saving)
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
                                    original_frame_path = os.path.join(video_path, current_frames[frame_idx])
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
                                    original_frame_path = os.path.join(video_path, current_frames[frame_idx])
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
            # Always cleanup temporary directory
            self._cleanup_temp_directory(temp_video_dir)
    

    def _create_temp_video_directory(self, video_path, current_frames):
        """
        Creates a temporary directory with symlinks to video frames in the format expected by SAM2.
        SAM2 expects frames named like 00000.jpg, 00001.jpg, etc.
        """
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="sam2_video_")
        print(f"[INFO] Created temporary directory: {temp_dir}")
        
        try:
            # Create symlinks for each frame with the expected naming format
            for i, frame_name in enumerate(current_frames):
                source_path = os.path.join(video_path, frame_name)
                # SAM2 expects frame names like 00000.jpg, 00001.jpg, etc.
                target_name = f"{i:05d}.jpg"
                target_path = os.path.join(temp_dir, target_name)
                
                if os.path.exists(source_path):
                    # Create symlink (faster than copying)
                    os.symlink(source_path, target_path)
                    #print(f"[DEBUG] Created symlink: {source_path} -> {target_path}")
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
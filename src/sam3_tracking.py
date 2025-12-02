"""
SAM3 Tracking - Segment Anything 3 Video Tracking Integration

This module provides video tracking capabilities using Meta's SAM3 (Segment Anything 3)
model for object tracking and segmentation in medical videos. SAM3 is accessed via
HuggingFace and uses a different API compared to SAM2 (handle_request/handle_stream_request).

Features:
- SAM3 model access via HuggingFace authentication
- local model download
- Video frame-by-frame tracking with temporal consistency
- Integration with TagMed annotation workflow
- Session-based tracking with box, and polygon prompts
- Advanced segmentation propagation through video frames

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import os
import numpy as np
import cv2
import ast
import torch
import pandas as pd

from config_handler import ConfigHandler
from video_annotation_handler import VideoAnnotationHandler
from mask_handler import MaskHandler


class SAM3Tracking:
    """
    Video tracking and segmentation using Meta's SAM3 (Segment Anything 3) model.
    
    This class provides advanced video tracking capabilities using SAM3 accessed via HuggingFace.
    Unlike SAM2, SAM3 uses a handle_request/handle_stream_request API and does not require
    local model downloads or Hydra configuration.
    
    Attributes:
        gui: Reference to the main GUI interface
        config_handler (ConfigHandler): Configuration management
        video_annotation_handler (VideoAnnotationHandler): Video annotation interface
        mask_handler (MaskHandler): Mask visualization and management
        sam3_predictor: Loaded SAM3 model instance
        sam3_session_id: Session ID for SAM3 video tracking
    """
    
    def __init__(self, gui):
        """
        Initialize SAM3 tracking with configuration and model setup.
        
        Sets up the tracking environment and integrates with the annotation workflow.
        SAM3 models are accessed via HuggingFace and do not require local downloads.
        
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

        # Model instances
        self.sam3_predictor = None
        self.sam3_session_id = None  # Session ID for video tracking




    def check_if_sam_3_is_available(self):
        """
        Check if SAM3 is available via HuggingFace authentication.
        
        Checks if user is authenticated and if SAM3 model files exist locally.
        If not, downloads the model to the models directory.
        
        Returns:
            bool: True if SAM3 is available and authenticated, False otherwise
        """
        try:
            # Try importing the SAM3 module
            from sam3.model_builder import build_sam3_video_predictor
            
            # Define local model directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.abspath(os.path.join(script_dir, "..", "models", "sam3"))
            
            # Check if HuggingFace token is configured
            print("[INFO] Checking HuggingFace authentication...")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                # Try to get user info - this will fail if not authenticated
                user_info = api.whoami()
                print(f"[INFO] HuggingFace Login successful as: {user_info.get('name', 'Unknown')}")
            except Exception as e:
                print(f"[ERROR] HuggingFace authentication failed: {e}")
                print("[WARN] SAM3 requires HuggingFace authentication.")
                print("[INFO] Please run 'huggingface-cli login' in your terminal.")
                return False
            
            # Check if model files already exist locally
            if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
                print(f"[INFO] SAM3 model found at: {model_dir}")
                self.sam3_model_dir = model_dir
                return True
            
            # Model not found locally - download it
            print(f"[INFO] SAM3 model not found locally. Starting download to: {model_dir}")
            try:
                from huggingface_hub import snapshot_download
                
                # Create model directory if it doesn't exist
                os.makedirs(model_dir, exist_ok=True)
                
                # Download the entire SAM3 model repository
                print("[INFO] Loading SAM3 model from HuggingFace (this may take a few minutes)...")
                snapshot_download(
                    repo_id="facebook/sam3",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                
                print(f"[INFO] SAM3 model successfully downloaded to: {model_dir}")
                self.sam3_model_dir = model_dir
                return True
                
            except Exception as download_error:
                print(f"[ERROR] Download of SAM3 model failed: {download_error}")
                return False
                    
        except ImportError as e:
            print(f"[ERROR] SAM3 package not found: {e}")
            return False

    def load_sam3_model(self):
        """
        Loads the SAM3 video predictor from local model files.
        The model must be downloaded first via check_if_sam_3_is_available().
        """
        try:
            from sam3.model_builder import build_sam3_video_model
            
            # Check if model directory exists
            if not hasattr(self, 'sam3_model_dir') or not os.path.exists(self.sam3_model_dir):
                print("[ERROR] SAM3 dir not found. Please run check_if_sam_3_is_available() first.")
                self.sam3_predictor = None
                return
            
            # Disable torch compilation for better compatibility
            torch._dynamo.config.disable = True
            torch._inductor.config.disable_progress = True
            torch._inductor.config.triton.unique_kernel_names = True
            
            # Set float32 precision
            torch.set_float32_matmul_precision('high')

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Loading SAM3 Video Predictor on {device}")
            print(f"[INFO] Using model from: {self.sam3_model_dir}")

            # Set environment variable to use local model directory
            os.environ["SAM3_LOCAL_DIR"] = self.sam3_model_dir
            
            # Build the SAM3 model and extract tracker (like in the notebook)
            sam3_model = build_sam3_video_model()
            self.sam3_predictor = sam3_model.tracker
            self.sam3_predictor.backbone = sam3_model.detector.backbone

            print("[INFO] SAM3 Video Predictor successfully loaded.")

        except Exception as e:
            import traceback
            print(f"[ERROR] Loading SAM3 Video Predictor failed: {e}")
            traceback.print_exc()
            self.sam3_predictor = None

    
    def _close_sam3_session(self):
        """Closes the active SAM3 session."""
        if hasattr(self, 'sam3_session_id') and self.sam3_session_id is not None and self.sam3_predictor is not None:
            try:
                self.sam3_predictor.handle_request({
                    "type": "close_session",
                    "session_id": self.sam3_session_id
                })
                print(f"[INFO] SAM3 session {self.sam3_session_id} closed.")
            except Exception as e:
                print(f"[WARN] Failed to close SAM3 session: {e}")
            finally:
                self.sam3_session_id = None
            
    def sam3_tracking_method(self):
        """
        Uses SAM3 video predictor to propagate annotations across all following video frames.
        SAM3 uses session-based API: start_session, add_prompt, propagate_in_video.
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
            return
        df_index = self.gui.all_annotations[match].index[0]
        row = self.gui.all_annotations.loc[df_index]

        # Check if we have the video predictor loaded
        if not hasattr(self, "sam3_predictor") or self.sam3_predictor is None:
            print("[INFO] SAM3 Predictor not loaded, loading now...")
            self.load_sam3_model()
        predictor = self.sam3_predictor

        # Verify predictor is loaded
        if self.sam3_predictor is None:
            print("[ERROR] SAM3 Predictor could not be loaded. Tracking aborted.")
            return
        
        # Initialize video sequence
        video_folder = os.path.join(self.selected_image_folder, self.gui.patient_id, self.gui.selected_exam)
        
        # Create temp directory with properly named frames (SAM3 expects numeric names)
        temp_video_dir = self._create_temp_video_directory_with_extractor(video_folder, current_frames)
        
        try:
            resize_h, resize_w = self.image_size

            # Check if we have an existing inference state for this video
            # This allows corrections/refinements without reinitializing
            video_id = f"{self.gui.patient_id}_{self.gui.selected_exam}_{self.gui.selected_video_index}"
            
            if not hasattr(self, 'sam3_inference_states'):
                self.sam3_inference_states = {}
            
            if video_id not in self.sam3_inference_states:
                # SAM3 uses init_state like SAM2 (SAM2-compatible API)
                print(f"[INFO] Initializing SAM3 Inference State for new video...")
                
                try:
                    # Use temp directory with numeric frame names
                    video_path = temp_video_dir
                    
                    # Initialize inference state (like SAM2)
                    inference_state = predictor.init_state(video_path=video_path)
                    
                    # Store for reuse
                    self.sam3_inference_states[video_id] = {
                        'state': inference_state,
                        'temp_dir': temp_video_dir
                    }
                    
                    print("[DEBUG] SAM3 inference_state successfully initialized")
                except Exception as e:
                    print(f"[ERROR] Initializing SAM3 Inference State failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self._cleanup_temp_directory(temp_video_dir)
                    return
            else:
                # Reuse existing inference state for corrections
                print(f"[INFO] Reusing existing SAM3 Inference State for corrections...")
                inference_state = self.sam3_inference_states[video_id]['state']
                # Cleanup new temp dir since we're using the old one
                self._cleanup_temp_directory(temp_video_dir)
                temp_video_dir = self.sam3_inference_states[video_id]['temp_dir']
            
            # Check if Polygon or Bounding Box
            selected_text = self.gui.video_annotation_listbox.get(selected_annotation_index)
            is_polygon = "Polygon" in selected_text

            # SAM3 uses 1-based object IDs like SAM2
            ann_obj_id = selected_annotation_index + 1
            ann_frame_idx = current_frame_index
            
            # Initialize video_segments dictionary
            video_segments = {}

            # ===== POLYGON TRACKING =====
            if is_polygon:
                print("[INFO] Starting SAM3 Video Polygon Tracking with Mask Prompt...")
                
                try:
                    # Get polygon data from dataframe
                    polygon_list = self._safe_parse_list(row.get('polygon'))
                    polygon_class_list = self._safe_parse_list(row.get('class_polygon'))
                        
                    polygon = polygon_list[selected_annotation_index]
                    selected_class = polygon_class_list[selected_annotation_index]

                    if not isinstance(polygon, list) or len(polygon) < 3:
                        print("[ERROR] Invalid polygon data - at least 3 points required.")
                        self._cleanup_temp_directory(temp_video_dir)
                        return

                    # Convert polygon to mask
                    points_array = np.array(polygon).reshape(-1, 2)
                    
                    # Get frame dimensions
                    original_frame_path = self._get_frame_path(video_folder, current_frames[current_frame_index])
                    if os.path.exists(original_frame_path):
                        original_frame = cv2.imread(original_frame_path)
                        original_height, original_width = original_frame.shape[:2]
                        
                        # Scale polygon coordinates from GUI size to original frame size
                        scale_x = original_width / resize_w
                        scale_y = original_height / resize_h
                        
                        scaled_points = points_array.copy().astype(np.float64)
                        scaled_points[:, 0] *= scale_x
                        scaled_points[:, 1] *= scale_y
                        
                        # Convert polygon to binary mask at original frame size
                        mask = self._polygon_to_mask(
                            scaled_points.flatten().tolist(), 
                            original_width, 
                            original_height
                        )
                    else:
                        # Fallback: use GUI size
                        mask = self._polygon_to_mask(
                            points_array.flatten().tolist(), 
                            resize_w, 
                            resize_h
                        )

                    try:
                        # SAM3 add_new_mask (SAM2-compatible API)
                        import torch
                        
                        # Convert mask to torch tensor
                        # SAM3 expects 2D mask with shape (H, W), not (1, H, W)
                        # Use same dtype as model (BFloat16 on CUDA with Ampere+)
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
                            mask_tensor = torch.from_numpy(mask).to(device=device, dtype=torch.bfloat16)
                        else:
                            mask_tensor = torch.from_numpy(mask).float().to(device)
                        
                        print(f"[DEBUG] Adding mask prompt from polygon")
                        print(f"[DEBUG] Mask shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}, non-zero pixels: {(mask_tensor > 0).sum().item()}")
                        
                        # Use add_new_mask like in the notebook
                        _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            mask=mask_tensor,
                        )
                        
                        print(f"[DEBUG] Polygon prompt successfully added (as mask) - Object IDs: {out_obj_ids}")
                        
                    except Exception as e:
                        print(f"[ERROR] Error adding polygon prompt: {e}")
                        import traceback
                        traceback.print_exc()
                        self._cleanup_temp_directory(temp_video_dir)
                        return

                except IndexError:
                    print(f"[ERROR] Polygon Index {selected_annotation_index} out of valid range.")
                    self._cleanup_temp_directory(temp_video_dir)
                    return

            else:
                # ===== BOUNDING BOX TRACKING =====
                print("[INFO] Starting SAM3 Video Bounding Box Tracking...")
                
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

                    # Scale coordinates from GUI size to original frame size
                    original_frame_path = self._get_frame_path(video_folder, current_frames[current_frame_index])
                    if os.path.exists(original_frame_path):
                        original_frame = cv2.imread(original_frame_path)
                        original_height, original_width = original_frame.shape[:2]
                        
                        scale_x = original_width / resize_w
                        scale_y = original_height / resize_h
                        
                        # Scale bounding box
                        scaled_x = x * scale_x
                        scaled_y = y * scale_y
                        scaled_w = w * scale_x
                        scaled_h = h * scale_y
                        
                        # Convert center format to corners format [x_min, y_min, x_max, y_max]
                        x_min = scaled_x - scaled_w / 2
                        y_min = scaled_y - scaled_h / 2
                        x_max = scaled_x + scaled_w / 2
                        y_max = scaled_y + scaled_h / 2
                        
                        # SAM3 requires normalized coordinates [0, 1] in corners format
                        rel_box = np.array([[
                            x_min / original_width,
                            y_min / original_height,
                            x_max / original_width,
                            y_max / original_height
                        ]], dtype=np.float32)
                        
                        # Convert to tensor immediately with correct device
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        rel_box = torch.from_numpy(rel_box).to(device=device, dtype=torch.float32)
                    else:
                        # Fallback: normalize by GUI size
                        x_min = x - w / 2
                        y_min = y - h / 2
                        x_max = x + w / 2
                        y_max = y + h / 2
                        rel_box_np = np.array([[
                            x_min / resize_w,
                            y_min / resize_h,
                            x_max / resize_w,
                            y_max / resize_h
                        ]], dtype=np.float32)
                        
                        # Convert to tensor immediately with correct device
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        rel_box = torch.from_numpy(rel_box_np).to(device=device, dtype=torch.float32)

                    try:
                        # SAM3 add_new_points_or_box (SAM2-compatible API)
                        import torch
                        
                        print(f"[DEBUG] Adding box prompt (BBox): {rel_box[0].cpu().numpy()}")
                        print(f"[DEBUG] Box device: {rel_box.device}, dtype: {rel_box.dtype}")
                        
                        # Workaround for SAM3 bug: provide empty points on same device as box
                        # This prevents device mismatch when concatenating box_coords with points
                        device = rel_box.device
                        empty_points = torch.zeros(0, 2, dtype=torch.float32, device=device)
                        empty_labels = torch.zeros(0, dtype=torch.int32, device=device)
                        
                        # Important: normalize_coords=False because our box is already normalized to [0, 1]
                        _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            points=empty_points,  # Empty points on correct device
                            labels=empty_labels,   # Empty labels on correct device
                            box=rel_box,
                            normalize_coords=False,  # Box is already normalized!
                        )
                        
                        print(f"[DEBUG] Bounding Box prompt successfully added - Object IDs: {out_obj_ids}")
                        
                    except Exception as e:
                        print(f"[ERROR] Error adding bounding box prompt: {e}")
                        import traceback
                        traceback.print_exc()
                        self._cleanup_temp_directory(temp_video_dir)
                        return

                except IndexError:
                    print(f"[ERROR] Bounding Box Index {selected_annotation_index} out of valid range.")
                    self._cleanup_temp_directory(temp_video_dir)
                    return

            # ===== PROPAGATE THROUGH VIDEO =====
            print("[INFO] Propagating annotations through video with SAM3...")
            
            try:
                print(f"[DEBUG] Starting propagate_in_video() from frame {ann_frame_idx}...")
                
                # SAM3 propagate_in_video (SAM2-compatible API)
                # This is a generator that yields results for each frame
                for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
                    inference_state, 
                    start_frame_idx=0, 
                    max_frame_num_to_track=len(current_frames), 
                    reverse=False, 
                    propagate_preflight=True
                ):
                    video_segments[frame_idx] = {
                        out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(obj_ids)
                    }
                
                print(f"[DEBUG] propagate_in_video() successfully completed. {len(video_segments)} frames processed.")
            except Exception as e:
                print(f"[ERROR] Error during video propagation: {e}")
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
                    
                    # Ensure mask is 2D
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    
                    # Skip empty masks
                    if mask.sum() == 0:
                        print(f"[INFO] Empty mask for frame {next_img_id}")
                        continue

                    # Scale mask to GUI size
                    if mask.shape != (resize_h, resize_w):
                        mask_for_gui = cv2.resize(mask.astype(np.uint8), (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
                        path_mask = self.mask_handler.save_mask(mask_for_gui, next_img_id, selected_class, selected_annotation_index)
                    else:
                        path_mask = self.mask_handler.save_mask(mask, next_img_id, selected_class, selected_annotation_index)
                    
                    # Update database for frames after the initial frame
                    if frame_idx != current_frame_index:
                        match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == next_img_id
                        if match_next.any():
                            next_df_index = self.gui.all_annotations[match_next].index[0]
                            
                            if is_polygon:
                                # Convert mask back to polygon
                                generated_polygon = self._mask_to_polygon(mask.squeeze())
                                
                                # Scale polygon back to GUI coordinates
                                if generated_polygon:
                                    original_frame_path = self._get_frame_path(video_folder, current_frames[frame_idx])
                                    if os.path.exists(original_frame_path):
                                        original_frame = cv2.imread(original_frame_path)
                                        original_height, original_width = original_frame.shape[:2]
                                        
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
                                
                                while len(existing_polygons) <= selected_annotation_index:
                                    existing_polygons.append([])
                                    existing_class_polygons.append("")
                                    existing_polygon_annotypes.append("")
                                
                                existing_polygons[selected_annotation_index] = generated_polygon
                                existing_class_polygons[selected_annotation_index] = selected_class
                                existing_polygon_annotypes[selected_annotation_index] = "sam3_tracking"
                                
                                self.gui.all_annotations.at[next_df_index, 'polygon'] = existing_polygons
                                self.gui.all_annotations.at[next_df_index, 'class_polygon'] = existing_class_polygons
                                self.gui.all_annotations.at[next_df_index, 'polygon_annotype'] = existing_polygon_annotypes
                                
                            else:
                                # Update bounding box data
                                ys, xs = np.where(mask.squeeze())
                                if len(xs) > 0 and len(ys) > 0:
                                    x0, y0 = xs.min(), ys.min()
                                    x1, y1 = xs.max(), ys.max()
                                    
                                    # Scale back to GUI size
                                    original_frame_path = self._get_frame_path(video_folder, current_frames[frame_idx])
                                    if os.path.exists(original_frame_path):
                                        original_frame = cv2.imread(original_frame_path)
                                        original_height, original_width = original_frame.shape[:2]
                                        
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
                                                    [new_x, new_y, new_w, new_h, selected_class, 'sam3_tracking']):
                                        existing_list = self._safe_parse_list(self.gui.all_annotations.at[next_df_index, col])

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

            print(f"[INFO] SAM3 Video Tracking completed for {frames_processed} frames.")

        except Exception as e:
            print(f"[ERROR] SAM3 Video Tracking could not be completed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup temporary directory
            self._cleanup_temp_directory(temp_video_dir)
    

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
        Creates a temporary directory with symlinks to video frames using the VideoFrameExtractor.
        SAM3 expects frames with names like 00000.jpg, 00001.jpg, etc.
        """
        import tempfile
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
        print(f"[INFO] Created temporary directory: {temp_dir}")
        
        try:
            # Create symlinks for each frame with the expected naming format
            for i, frame_name in enumerate(current_frames):
                # Use VideoFrameExtractor for correct path
                source_path = self._get_frame_path(video_folder, frame_name)
                
                # SAM3 expects frame names like 00000.jpg, 00001.jpg, etc.
                target_name = f"{i:05d}.jpg"
                target_path = os.path.join(temp_dir, target_name)
                
                if os.path.exists(source_path):
                    # Create symlink (faster than copying)
                    os.symlink(source_path, target_path)
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
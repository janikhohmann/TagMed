import pandas as pd
import os
import ast
import tkinter as tk 
from PIL import Image, ImageTk
import statistics as stat
from tkinter import  ttk, messagebox

from annotation_loader import AnnotationLoader
from config_handler import ConfigHandler 

from mask_handler import MaskHandler


class VideoAnnotationHandler():
    """
    Handles video frame annotation functionality for the TagMed medical annotation application.
    
    This class provides comprehensive annotation capabilities for video frames including:
    - Bounding box annotations with interactive resizing and moving
    - Polygon annotations with point-based drawing and editing
    - Single-point prompt annotations for advanced ML workflows
    - Frame-by-frame navigation and annotation management
    - Integration with mask prediction and video processing systems
    - Color-coded video status indicators for annotation progress tracking
    
    The handler manages both drawing new annotations on video frames and modifying
    existing ones, maintaining synchronization between visual canvas elements and
    the underlying annotation data stored in pandas DataFrames. It coordinates
    with video frame extraction and intelligent frame management systems.
    """
    def __init__(self, gui):
        """
        Initialize the video annotation handler.
        
        Args:
            gui: Main GUI instance providing access to video canvas, annotation data,
                 and UI controls for the medical video annotation interface.
        """
        self.gui = gui

        # Load configuration settings for video display
        config = ConfigHandler()
        self.image_size = config.get("image_size", (600, 600))  # Default frame size for display

        # Initialize annotation data management
        self.annotation_loader = AnnotationLoader()

        # Bounding box drawing state management
        self.is_drawing = False  # Flag indicating if currently drawing a rectangle
        self.rect_start = None   # Starting coordinates (x, y) for rectangle drawing
        self.rect_end = None     # Ending coordinates (x, y) for rectangle drawing
        self.rect_id = None      # Canvas ID of the currently drawn rectangle
        self.temp_rect_id = None # Temporary canvas ID for rectangle during drawing
        self.resize_handles = [] # List of canvas IDs for resize handle elements
        self.resize_handle_size = 3  # Size of resize handles in pixels

        # Annotation selection and interaction state
        self.selected_annotation_index = None  # Index of currently selected annotation

        # Rectangle manipulation state for modify mode
        self.dragging_handle = None      # Which resize handle is being dragged
        self.dragging_rectangle = False  # Flag for rectangle movement mode
        self.last_mouse_pos = None       # Last recorded mouse position for dragging
        self.drawn_rect_ids = []         # List of all rectangle/polygon canvas IDs for current frame

        # Polygon-specific attributes for polygon annotation mode
        self.polygon_points = []      # List of [x, y] coordinate pairs for current polygon
        self.polygon_point_ids = []   # Canvas IDs for polygon point visual elements
        self.polygon_line_id = None   # Canvas ID for polygon outline
        self.selected_point_index = None  # Index of currently selected polygon point
        self.listbox_index = None     # Index of selected annotation in GUI listbox
        self.polygon_index = None     # Index of polygon within annotation data
        self.is_drawing_polygon = False  # Flag for polygon drawing mode
        self.dragging_polygon = False    # Flag for polygon dragging mode

        # Single-point prompt annotation system for advanced ML workflows
        self.single_point_prompts = []      # List of single-point coordinates for ML prompts
        self.single_point_prompt_ids = []   # Canvas IDs for single-point prompt visual elements
        self.is_giving_single_points = False  # Flag for single-point annotation mode

        # Initialize mask handling for video frame predictions
        self.mask_handler = MaskHandler(self)



    def add_annotation(self):
        """
        Add a new annotation to the current video frame based on the selected annotation type.
        
        Supports two main annotation types for video frames:
        1. Bounding Box: Creates rectangular annotations with optional mask prediction
        2. Polygon: Creates polygon annotations from manually drawn points
        
        The method validates compatibility with existing annotations (prevents mixing
        bounding box and polygon annotations on the same frame), updates the annotation
        DataFrame, refreshes the GUI listbox, and optionally generates masks for video frames.
        
        Video-specific considerations:
        - Works with current frame ID from video navigation
        - Uses video-specific GUI elements (video_annotation_listbox, video_selected_class)
        - Integrates with frame mask prediction for video workflows
        
        Side Effects:
            - Updates self.gui.all_annotations DataFrame with new frame annotation data
            - Adds visual representation to the video frame canvas
            - Updates video annotation listbox in GUI
            - May trigger mask prediction if frame mask creation is enabled
            - Updates video listbox color coding for annotation progress
        """
        # Import mask predictor dynamically to avoid circular imports
        from image_mask_predictor import ImageMaskPredictor
        self.image_mask_predictor = ImageMaskPredictor(gui=self.gui)

        # Extract annotation parameters from video-specific GUI controls
        img_selected_class = self.gui.video_selected_class.get()    # Medical class/category for video
        img_annotation_type = self.gui.video_annotation_type.get()  # Annotation type (BB/Polygon)
        image_id = self.current_image_id  # Current frame ID from video navigation

        # Validate that annotation DataFrame contains required frame ID column
        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return


        # Search for existing row matching current frame ID
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(image_id).strip()
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{image_id}'")
            return
        idx = self.gui.all_annotations[match].index[0]  # Get DataFrame row index


        # Helper function for managing list-based annotation data
        def append_or_init_list(cell, value):
            """
            Helper Function: Appends a value to a list in a DataFrame cell or initializes it if empty.
            
            Handles various cell states:
            - NaN values: Initialize new list with value
            - "NN" strings: Replace with new list containing value
            - Existing lists: Append value to existing list
            - String representations of lists: Parse and append
            
            Args:
                cell: Current cell value (can be NaN, str, list, or parseable string)
                value: Value to append or use for initialization
                
            Returns:
                list: Updated list containing the new value
            """
            if isinstance(cell, float) and pd.isna(cell):
                return [value]
            if isinstance(cell, str) and cell == "NN":
                return [value]
            if isinstance(cell, list):
                return cell + [value]
            # Handle single float/int values that were previously stored as lists
            if isinstance(cell, (int, float)) and not pd.isna(cell):
                return [cell, value]  # Convert single value to list and append new value
            if isinstance(cell, str):
                try:
                    parsed = ast.literal_eval(cell)
                    if isinstance(parsed, list):
                        return parsed + [value]
                except:
                    # If parsing fails, treat as single string value
                    return [cell, value]
            return [value]


        # === Bounding Box Annotation Processing ===
        if img_annotation_type == "Bounding Box":
            # Validate annotation type compatibility - prevent mixing BB and polygon annotations
            polygon_value = self.gui.all_annotations.at[idx, "polygon"]
            # Check if polygon annotations already exist (handle lists and single values)
            has_polygon = False
            if isinstance(polygon_value, list) and len(polygon_value) > 0:
                has_polygon = True
            elif not isinstance(polygon_value, list) and pd.notna(polygon_value):
                has_polygon = True
            
            if has_polygon:
                self.gui.wrong_annotation_warning_gui("Bounding Box")
                self.delete_all_bounding_boxes()
                return
                
            # Calculate bounding box parameters from drawn rectangle
            x, y, w, h = self.calculate_rectangle()
            
            # Create formatted text representation for video annotation listbox display
            annotation_text = f"{img_selected_class.ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
            self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

            # Store bounding box data in annotation DataFrame
            # Updates: x, y, w, h coordinates, class label, and annotation type
            
            # Ensure columns can store lists by converting to object dtype
            for col in ['x', 'y', 'w', 'h', 'class', 'bb_annotype']:
                if col in self.gui.all_annotations.columns:
                    self.gui.all_annotations[col] = self.gui.all_annotations[col].astype('object')
            
            for col, val in zip(['x', 'y', 'w', 'h', 'class', 'bb_annotype'], [x, y, w, h, img_selected_class, 'manually']):
                old_value = self.gui.all_annotations.at[idx, col]
                new_value = append_or_init_list(old_value, val)
                #print(f"[DEBUG] Column '{col}': {old_value} + {val} -> {new_value}")
                self.gui.all_annotations.at[idx, col] = new_value

            # Register rectangle canvas element for future reference and manipulation
            self.rect_id = self.temp_rect_id
            self.drawn_rect_ids.append(self.rect_id)
            self.temp_rect_id = None  # Clear temporary rectangle ID after assignment

            self.gui.frame_canvas.itemconfig(self.rect_id, tags=("boundingbox",)) # Set permanent tag

            # Generate frame mask prediction if mask creation is enabled
            if self.gui.create_frame_mask_var.get():
                self.image_mask_predictor.predict_mask_for_image_bb(idx, x, y, w, h)
            
            print(f"[DEBUG] Added Bounding Box with rect_id {self.rect_id}")

        # === Polygon Annotation Processing ===
        elif img_annotation_type == "Polygon":
            # Validate annotation type compatibility - prevent mixing BB and polygon annotations
            class_value = self.gui.all_annotations.at[idx, "class"]
            # Check if bounding box annotations already exist (handle lists and single values)
            has_bbox = False
            if isinstance(class_value, list) and len(class_value) > 0:
                has_bbox = True
            elif not isinstance(class_value, list) and pd.notna(class_value):
                has_bbox = True
            
            if has_bbox:
                self.gui.wrong_annotation_warning_gui("Polygon")
                self.delete_all_polygons()
                return
    
            # Validate that polygon points have been collected
            if not hasattr(self, "polygon_points") or not self.polygon_points:
                print("[ERROR] No polygon points available.")
                return

            # Create formatted text representation for video annotation listbox display
            annotation_text = f"{img_selected_class.ljust(12)} Polygon: {len(self.polygon_points)} points"
            self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

            # Store polygon data in annotation DataFrame
            # Initialize polygon column if it doesn't exist
            if 'polygon' not in self.gui.all_annotations.columns:
                self.gui.all_annotations['polygon'] = None

            # Create deep copy of polygon points to prevent reference issues
            polygon_copy = [point.copy() for point in self.polygon_points]
            
            # Ensure columns can store lists by converting to object dtype
            for col in ['polygon', 'class_polygon', 'polygon_annotype']:
                if col in self.gui.all_annotations.columns:
                    self.gui.all_annotations[col] = self.gui.all_annotations[col].astype('object')
            
            # Update DataFrame with polygon annotation data
            self.gui.all_annotations.at[idx, 'polygon'] = append_or_init_list(
                self.gui.all_annotations.at[idx, 'polygon'], polygon_copy)
            self.gui.all_annotations.at[idx, 'class_polygon'] = append_or_init_list(
                self.gui.all_annotations.at[idx, 'class_polygon'], img_selected_class)
            self.gui.all_annotations.at[idx, 'polygon_annotype'] = append_or_init_list(
                self.gui.all_annotations.at[idx, 'polygon_annotype'], "manually")

            # Generate frame mask prediction for polygon if mask creation is enabled
            if self.gui.create_frame_mask_var.get():
                width, height = self.gui.image_size
                self.image_mask_predictor.predict_mask_for_image_polygon(idx, self.polygon_points.copy(), width, height)

            print(f"[DEBUG] Added Polygon with {len(self.polygon_points)} points")

            # Clear polygon points after successful storage to prepare for next annotation
            self.polygon_points.clear()

        # === Error Handling for Unsupported Annotation Types ===
        else:
            print(f"[ERROR] Unknown annotation type: {img_annotation_type}")
            return

        # Update visual representation of annotation status in video listbox
        self.update_video_listbox_with_annotation_colors()

    def delete_annotation(self):
        """
        Delete the selected annotation from both the visual canvas and annotation data.
        
        Handles deletion of both bounding box and polygon annotations for video frames, including:
        - Removing visual elements from frame canvas (rectangles, polygons, points)
        - Updating annotation DataFrame by removing corresponding entries
        - Managing mask data associated with deleted frame annotations
        - Refreshing GUI listbox and visual representations
        - Proper cleanup of modify mode state if active
        
        Video-specific considerations:
        - Works with current frame ID and video annotation listbox
        - Uses frame canvas for visual element management
        - Integrates with video annotation progress tracking
        
        The method automatically determines annotation type and performs
        appropriate cleanup for the specific annotation format.
        """
        # Get currently selected annotation from video annotation listbox
        selected = self.gui.video_annotation_listbox.curselection()
        if not selected:
            return
        
        # Handle cleanup if currently in modify mode
        if self.gui.modify_mode.get():  # Clean up modify mode state
            self.gui.modify_mode.set(False)
            self.remove_resize_handles()
            self.update_annotation_in_listbox()
            # Clean up polygon editing elements from frame canvas
            for pid in getattr(self, 'polygon_point_ids', []):
                self.gui.frame_canvas.delete(pid)
                self.polygon_point_ids = []
                self.polygon_points = []

        # Extract deletion parameters
        index = selected[0]  # Position in the video annotation listbox
        image_id = self.current_image_id  # Current frame ID

        # Validate annotation DataFrame structure
        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return

        # Find matching row in annotation DataFrame for current frame
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(image_id).strip()
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{image_id}'")
            return

        idx = self.gui.all_annotations[match].index[0]

        try:
            row = self.gui.all_annotations.loc[idx]

            # Determine if the selected annotation is polygon-based or bounding box-based
            is_polygon = False
            if 'polygon' in row:
                polygons = self._safe_parse_list(row['polygon'])
                if index < len(polygons):
                    is_polygon = True

            # === Polygon Annotation Deletion ===
            if is_polygon:
                # Extract polygon-related data lists from DataFrame
                polygons = self._safe_parse_list(self.gui.all_annotations.at[idx, 'polygon'])
                class_list = self._safe_parse_list(self.gui.all_annotations.at[idx, 'class_polygon'])
                annotype_list = self._safe_parse_list(self.gui.all_annotations.at[idx, 'polygon_annotype'])

                # Remove data at specified index from all polygon-related lists
                polygons.pop(index)
                if index < len(class_list):
                    class_list.pop(index)
                if index < len(annotype_list):
                    annotype_list.pop(index)

                # Update DataFrame with modified lists (use None if lists become empty)
                self.gui.all_annotations.at[idx, 'polygon'] = polygons if polygons else None
                self.gui.all_annotations.at[idx, 'class_polygon'] = class_list if class_list else None
                self.gui.all_annotations.at[idx, 'polygon_annotype'] = annotype_list if annotype_list else None

            # === Bounding Box Annotation Deletion ===
            else:
                # Delete associated mask before removing annotation data
                mask_paths = self._safe_parse_list(self.gui.all_annotations.at[idx, 'masks'])
                if index < len(mask_paths):
                    path_to_remove = mask_paths[index]
                    print(f"[DEBUG] Deleting mask at path: {path_to_remove}")
                    self.mask_handler.delete_mask(path_to_remove)

                # Remove data from all bounding box-related columns
                for col in ['x', 'y', 'w', 'h', 'class', 'bb_annotype', 'masks']:
                    val = self._safe_parse_list(self.gui.all_annotations.at[idx, col])
                    if index < len(val):
                        val.pop(index)
                    # Update DataFrame column with modified list
                    self.gui.all_annotations.at[idx, col] = val if val else None

            # === Visual Element Cleanup ===
            # Remove visual element from frame canvas (rectangle or polygon)
            if index < len(self.drawn_rect_ids):
                shape_id = self.drawn_rect_ids[index]
                self.gui.frame_canvas.delete(shape_id)
                self.drawn_rect_ids.pop(index)

            # Remove resize handles if present
            for handle in self.resize_handles:
                self.gui.frame_canvas.delete(handle)
            self.resize_handles.clear()

            # Remove entry from video annotation listbox
            self.gui.video_annotation_listbox.delete(index)

            # Clean up polygon elements and reload annotations for current frame
            self.delete_all_polygons()
            self.load_annotations_for_frame()

            # Debug output for canvas element tracking
            all_items = self.gui.frame_canvas.find_all()
            print("All canvas elements:")
            for item in all_items:
                tags = self.gui.frame_canvas.gettags(item)
                coords = self.gui.frame_canvas.coords(item)
                print(f"ID: {item}, Tags: {tags}, Coordinates: {coords}")

            # Update visual annotation status indicators in video listbox
            self.update_video_listbox_with_annotation_colors()
            print(f"[DEBUG] Deleted annotation at index {index} from frame {image_id}.")

        except Exception as e:
            print(f"[ERROR] Failed to delete annotation: {e}")



    def load_annotations_for_frame(self):
        """
        Load and display all annotations (bounding boxes and polygons) for the current video frame.
        
        This method reconstructs the visual representation of all stored annotations
        for the current video frame by reading from the annotation DataFrame and creating
        corresponding canvas elements on the frame canvas. It handles both bounding box 
        and polygon annotations, validating data integrity and skipping invalid entries.
        
        Video-specific considerations:
        - Uses current_frame_index to determine frame ID
        - Updates current_image_id for frame tracking
        - Works with frame_canvas for video display
        - Integrates with video annotation listbox
        
        Side Effects:
            - Clears all existing visual annotations from frame canvas
            - Creates new rectangle and polygon canvas elements
            - Populates video annotation listbox with formatted entries
            - Updates drawn_rect_ids list for canvas element tracking
            - Sets current_image_id for frame reference
        """
        # Clear all existing annotations from frame canvas and GUI
        self.clear_all_annotations()

        # Determine current frame ID from video navigation
        current_image_id = self.gui.current_frames[self.gui.current_frame_index].split(".")[0]
        self.current_image_id = current_image_id

        # print(f"[DEBUG] Handler: Loading annotations for frame_id: '{self.current_image_id}'")

        df = self.gui.all_annotations

        # Filter annotation DataFrame for the specified frame ID
        filtered_df = df[df['img_ID'].astype(str).str.strip() == str(self.current_image_id).strip()]

        if filtered_df.empty:
            print("[INFO] No annotations found for this frame.")
            return

        # Ensure rect_id column exists for canvas element tracking
        if 'rect_id' not in self.gui.all_annotations.columns:
            self.gui.all_annotations['rect_id'] = None

        # Extract annotation data from the matching row
        idx = filtered_df.index[0]
        row = filtered_df.iloc[0]

        # === Extract Bounding Box Data ===
        x_list = self._safe_parse_list(row.get('x'))      # Center X coordinates
        y_list = self._safe_parse_list(row.get('y'))      # Center Y coordinates  
        w_list = self._safe_parse_list(row.get('w'))      # Widths
        h_list = self._safe_parse_list(row.get('h'))      # Heights
        class_list = self._safe_parse_list(row.get('class'))  # Medical class labels

        # === Extract Polygon Data ===
        polygon_list = self._safe_parse_list(row.get('polygon')) if 'polygon' in row else []
        polygon_class_list = self._safe_parse_list(row.get('class_polygon')) if 'class_polygon' in row else []

        # === Draw Bounding Boxes ===
        # Ensure all bounding box data arrays have consistent length
        count = min(len(x_list), len(y_list), len(w_list), len(h_list), len(class_list))
        
        for i in range(count):
            try:
                # Convert to integers and extract bounding box parameters
                x, y, w, h = int(x_list[i]), int(y_list[i]), int(w_list[i]), int(h_list[i])
                class_label = class_list[i]

                # Skip invalid bounding boxes with zero or negative dimensions
                if w <= 0 or h <= 0:
                    print(f"[WARN] Skipping invalid bounding box [{i}] (w or h <= 0)")
                    continue

                # Convert center coordinates and dimensions to corner coordinates
                x1, y1 = x - w // 2, y - h // 2  # Top-left corner
                x2, y2 = x + w // 2, y + h // 2  # Bottom-right corner

                # Create rectangle on frame canvas with red outline
                rect_id = self.gui.frame_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
                self.drawn_rect_ids.append(rect_id)

                # Create formatted text entry for video annotation listbox
                annotation_text = f"{str(class_label).ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
                self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

                print(f"[DEBUG] Drew bounding box rect_id {rect_id} at [{x1}, {y1}, {x2}, {y2}]")

            except Exception as e:
                print(f"[ERROR] Could not draw bounding box {i}: {e}")

        # === Draw Polygons ===
        for j, polygon in enumerate(polygon_list):
            try:
                # Validate polygon structure (must be list with at least 3 points)
                if not isinstance(polygon, list) or len(polygon) < 3:
                    print(f"[WARN] Skipping invalid polygon [{j}]")
                    continue

                # Flatten coordinate pairs into single list for tkinter polygon creation
                flat_points = [coord for point in polygon for coord in point]
                
                # Create polygon on frame canvas with blue outline and unique tag
                polygon_id = self.gui.frame_canvas.create_polygon(
                    flat_points, outline="blue", fill="", width=2, tags=f"polygon_{j}"
                )

                self.drawn_rect_ids.append(polygon_id)

                # Get class label or use default
                label = polygon_class_list[j] if j < len(polygon_class_list) else "unknown"
                
                # Create formatted text entry for video annotation listbox
                annotation_text = f"{label.ljust(12)} Polygon: {len(polygon)} points"
                self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

                print(f"[DEBUG] Drew polygon with id {polygon_id}, {len(polygon)} points")

            except Exception as e:
                print(f"[ERROR] Could not draw polygon {j}: {e}")


    def _safe_parse_list(self, value):
        """
        Safely parse list data from DataFrame cells that may contain various data types.
        
        Handles conversion of annotation data stored in different formats:
        - Existing lists: Return as-is
        - NaN or "NN" values: Return empty list
        - String representations: Attempt to parse using ast.literal_eval
        - Invalid data: Return empty list with error logging
        
        Args:
            value: Cell value from DataFrame (list, str, NaN, or other)
            
        Returns:
            list: Parsed list data or empty list if parsing fails
        """
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
    
    
    def on_press(self, event):
        """
        Handle mouse press events for video frame annotation drawing and modification.
        
        This method manages the initiation of annotation workflows based on the current
        annotation mode (Bounding Box, Polygon) and whether the system is in drawing
        mode or modify mode. It coordinates between different annotation types and
        interaction states specifically for video frame annotation.
        
        Video-specific considerations:
        - Works with frame_canvas for video display
        - Uses video annotation type and listbox controls
        - Supports only Bounding Box and Polygon modes (no Magic Wand for video)
        
        Args:
            event: Tkinter mouse event containing coordinates and button information
            
        Behavior:
            - Drawing Mode: Initiates new annotation creation on current frame
            - Modify Mode: Begins editing of existing frame annotations
            - Validates annotation type compatibility and mode states
        """
        # remove any existing temporary bounding box by tag
        self.gui.frame_canvas.delete("temp_boundingbox")

        x, y = event.x, event.y  # Extract mouse position from event
        annotation_mode = self.gui.video_annotation_type.get()  # Get current video annotation type

        # Validate supported annotation types for video
        if annotation_mode not in ["Bounding Box", "Polygon"]:
            print("[ERROR] Invalid annotation type selected. Only 'Bounding Box' and 'Polygon' are supported.")
            return

        # === Drawing Mode - Create New Frame Annotations ===   
        if not self.gui.modify_mode.get():
            if annotation_mode == "Bounding Box":
                # Initialize bounding box drawing on frame canvas
                self.is_drawing = True
                self.rect_start = (x, y)
                self.temp_rect_id = self.gui.frame_canvas.create_rectangle(
                    x, y, x, y, outline="red", width=2, tags="temp_boundingbox") # Temporary tag for new rectangle
                    
            else:  # Polygon mode
                # Determine polygon index for proper canvas organization
                polygon_index = 0
                for i in range(self.gui.video_annotation_listbox.size()):
                    entry = self.gui.video_annotation_listbox.get(i)
                    if "Polygon" in entry:
                        polygon_index += 1
                self.polygon_index = polygon_index

                # Add point to current polygon and create visual representation
                self.polygon_points.append([x, y])
                point_id = self.gui.frame_canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", tags="polygon")
                self.polygon_point_ids.append(point_id)
                self.redraw_polygon()  # Update polygon outline
        
        # === Modify Mode - Handle Frame Annotation Editing ===
        # Determine annotation type for appropriate modification behavior
        try:
            selected_text = self.gui.video_annotation_listbox.get(self.listbox_index)
            is_polygon = "Polygon" in selected_text
        except: 
            return
        
        if self.gui.modify_mode.get(): 
            if not is_polygon:
                # === Bounding Box Modification ===
                # Check if user clicked on resize handle or inside rectangle
                self.dragging_handle = self.get_handle_at_position(x, y)
                self.dragging_rectangle = self.is_inside_rectangle(x, y) and not self.dragging_handle
                self.last_mouse_pos = (x, y)
                
            else:
                # === Polygon Point Modification ===
                # Check if user clicked near a polygon point for editing (6 pixel tolerance)
                for i, (px, py) in enumerate(self.polygon_points):
                    if abs(event.x - px) < 6 and abs(event.y - py) < 6:
                        self.selected_point_index = i
                        return
                    

                    
    def on_drag(self, event):
        """
        Handle mouse drag events for continuous video frame annotation drawing and modification.
        
        This method manages real-time updates during mouse movement with button pressed
        for video frame annotations. It handles different behaviors based on annotation
        mode and interaction state:
        - Drawing mode: Updates visual representation during creation
        - Modify mode: Handles resizing, moving, and point manipulation
        
        Video-specific considerations:
        - Works with frame_canvas for video display
        - Uses video annotation listbox for state determination
        - Supports frame-by-frame annotation workflows
        
        Args:
            event: Tkinter mouse motion event with current coordinates
        """
        x, y = event.x, event.y  # Extract current mouse position
        annotation_mode = self.gui.video_annotation_type.get()  # Get video annotation type

        # === Drawing Mode - Real-time Frame Annotation Creation ===
        if not self.gui.modify_mode.get():
            if annotation_mode == "Bounding Box":
                # Update rectangle coordinates during drawing on frame canvas
                # use the temporary rectangle id while drawing (temp_rect_id)
                if self.is_drawing and self.temp_rect_id:
                    self.gui.frame_canvas.coords(
                        self.temp_rect_id,
                        self.rect_start[0], self.rect_start[1],
                        x, y
                    )
                    
            else:  # Polygon mode
                # Add point during continuous polygon drawing
                self.polygon_points.append([x, y])
                point_id = self.gui.frame_canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", tags="polygon")
                self.polygon_point_ids.append(point_id)
                self.redraw_polygon()  # Update polygon outline

        # === Modify Mode - Handle Frame Annotation Editing ===
        # Determine annotation type for appropriate modification behavior
        try:
            selected_text = self.gui.video_annotation_listbox.get(self.listbox_index)
            is_polygon = "Polygon" in selected_text
        except: 
            return

        if self.gui.modify_mode.get():
            if not is_polygon:
                # === Bounding Box Modification ===
                if self.dragging_handle:
                    # Resize rectangle by dragging corner handles
                    self.resize_rectangle(x, y, self.dragging_handle)
                elif self.dragging_rectangle:
                    # Move entire rectangle to new position
                    self.move_rectangle(x, y)
                self.last_mouse_pos = (x, y)
                
            else:
                # === Polygon Point Modification ===
                if self.selected_point_index is not None:
                    # Update coordinates of selected polygon point
                    self.polygon_points[self.selected_point_index] = [x, y]

                    # Update visual representation of the point on frame canvas
                    point_id = self.polygon_point_ids[self.selected_point_index]
                    self.gui.frame_canvas.coords(point_id, x-4, y-4, x+4, y+4)

                    # Redraw polygon with updated point position
                    self.redraw_polygon()

    def on_release(self, event):
        """
        Handle mouse button release events to finalize video frame annotation operations.
        
        This method completes annotation creation or modification operations for video frames:
        - Drawing mode: Finalizes new annotation coordinates
        - Modify mode: Clears interaction state and handles
        
        Args:
            event: Tkinter mouse button release event
        """
        x, y = event.x, event.y # get mouse position
        # get annotation type from GUI
        annotation_mode = self.gui.video_annotation_type.get()

        if annotation_mode == "Bounding Box":
            if not self.gui.modify_mode.get():
                if self.is_drawing:
                    self.rect_end = (x, y)
                    self.is_drawing = False
            else:
                self.dragging_handle = None
                self.dragging_rectangle = False
                self.last_mouse_pos = None

        if annotation_mode == "Polygon":
            self.selected_point_index = None



    def calculate_rectangle(self):
        """
        Calculates center x/y and width/height from current start and end points.
        """
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        canvas_x = round(stat.mean([x1, x2]))
        canvas_y = round(stat.mean([y1, y2]))
        canvas_width = abs(x2 - x1)
        canvas_height = abs(y2 - y1)

        return canvas_x, canvas_y, canvas_width, canvas_height


    def on_annotation_selected(self, event):
        """
        Triggered when a listbox annotation is selected. Clears handles.
        """

        self.remove_resize_handles()
        self.gui.modify_mode.set(False)

        selection = self.gui.video_annotation_listbox.curselection()
        if not selection:
            return
        listbox_index = selection[0]

        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == self.current_image_id 
        if not match.any():
            print(f"[ERROR] No row found for img_ID = {self.current_image_id }")
            return
        df_index = self.gui.all_annotations[match].index[0]

        # Setze den DataFrame Index und row als Attribute, damit update es nutzen kann
        self.selected_annotation_original_index = df_index
        self.row = self.gui.all_annotations.loc[df_index]
        self.listbox_index = listbox_index

        # get correct polygon index from listbox
        polygon_index = -1
        for i in range(listbox_index + 1):
            entry = self.gui.video_annotation_listbox.get(i)
            if "Polygon" in entry:
                polygon_index += 1
        self.polygon_index = polygon_index



    def clear_all_annotations(self):
        """Clear all rectangles, polygons from canvas and the listbox, including rect_id entries in the DataFrame."""

        # ==== Delte all rectangles from canvas ====
        for rect_id in self.drawn_rect_ids:
            self.gui.frame_canvas.delete(rect_id)
        self.drawn_rect_ids.clear()

        # remove resize handles
        self.remove_resize_handles()

        # === Delete all polygons from canvas ===
        # delete lines
        if hasattr(self, 'polygon_line_id') and self.polygon_line_id is not None:
            self.gui.frame_canvas.delete(self.polygon_line_id)
            self.polygon_line_id = None

        # delete points
        for pid in getattr(self, 'polygon_point_ids', []):
            self.gui.frame_canvas.delete(pid)
        self.polygon_point_ids = []
        self.polygon_points = []

        # to make sure everything is deleted
        self.delete_all_polygons()

        # empty listbox 
        self.gui.video_annotation_listbox.delete(0, tk.END)
        
        # set internal state back to initial
        self.selected_point_index = None
        self.rect_id = None
        self.listbox_index = None



    def delete_all_polygons(self):
        self.gui.frame_canvas.delete("polygon")  # deletes everything with the tag "polygon"
        self.polygon_point_ids = []
        self.polygon_points = []
        self.polygon_line_id = None
        #print("[DEBUG] All polygon elements deleted.")

    def delete_all_bounding_boxes(self):
        self.gui.frame_canvas.delete("boundingbox")  # deletes all with the tag "boundingbox"
        self.drawn_rect_ids = []
        #print("[DEBUG] All bounding box elements deleted.")



    def modify_annotation(self):
        """Modifies selected annotation, handling both bounding boxes and polygons."""
        if not self.gui.modify_mode.get():
            self.remove_resize_handles()
            self.update_annotation_in_listbox()
            for pid in getattr(self, 'polygon_point_ids', []):
                self.gui.frame_canvas.delete(pid)
                self.polygon_point_ids = []
                self.polygon_points = []
            return
    
        selection = self.gui.video_annotation_listbox.curselection()
        if not selection:
            print("Modify Error: No annotation selected in the listbox.")
            messagebox.showwarning("Modify Error", "No annotation selected to modify.", parent=self.gui.patient_window)

            # Automatically turn off modify mode and update toggle button
            self.gui.modify_mode.set(False)
            if hasattr(self.gui, 'gallery_navigator') and hasattr(self.gui.gallery_navigator, 'modify_toggle_button'):
                self.gui.gallery_navigator.modify_toggle_button.config(text="Modify: Off")
            
            return

        listbox_index = selection[0]
        self.listbox_index = listbox_index

        # Finde die passende Zeile im DataFrame
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == self.current_image_id 
        if not match.any():
            print(f"[ERROR] No row found for img_ID = {self.current_image_id }")
            return

        df_index = self.gui.all_annotations[match].index[0]
        row = self.gui.all_annotations.loc[df_index]
        self.row = row  # safe now for later
        self.selected_annotation_original_index = df_index

        # Bounding Box or Polygon ?
        selected_text = self.gui.video_annotation_listbox.get(listbox_index)
        is_polygon = "Polygon" in selected_text

        if is_polygon:
            # ----- POLYGON-ANNOTATION MODIFICATION -----

            # removes resize handles if they exist
            self.remove_resize_handles()  

            for pid in getattr(self, 'polygon_point_ids', []):
                self.gui.frame_canvas.delete(pid)
                self.polygon_point_ids = []
                self.polygon_points = []


            if hasattr(self, 'polygon_index') and self.polygon_index is not None:
                tag_to_delete = f"polygon_{self.polygon_index}"
                self.gui.frame_canvas.delete(tag_to_delete)


            # get polygon data from DataFrame
            polygon_list = self._safe_parse_list(row.get('polygon'))
            if self.polygon_index >= len(polygon_list):
                print(f"[ERROR] Polygon index {self.polygon_index} out of range.")
                return

            polygon = polygon_list[self.polygon_index]
            if not isinstance(polygon, list) or len(polygon) < 3:
                print("[ERROR] Invalid polygon data.")
                return

            # draw polygon points on canvas
            for x, y in polygon:
                self.polygon_points.append([x, y])
                point_id = self.gui.frame_canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4, fill="red"
                )
                self.polygon_point_ids.append(point_id)

            polygon_annotype_list = self._safe_parse_list(row.get('polygon_annotype'))
            polygon_annotype_list[self.polygon_index] = "manually"
            self.gui.all_annotations.at[df_index, 'polygon_annotype'] = polygon_annotype_list


            # draw polygon outline
            self.redraw_polygon()

            print(f"[DEBUG] Polygon mit {len(polygon)} Punkten geladen.")
        else:
            try:
                self.rect_id = self.drawn_rect_ids[self.listbox_index]
                coords = self.gui.frame_canvas.coords(self.rect_id)

                # PARSE             bb_annotype as list
                bbox_annotype_list = self._safe_parse_list(row.get('bb_annotype'))

                bbox_annotype_list[self.listbox_index] = "manually"
                self.gui.all_annotations.at[df_index, 'bb_annotype'] = bbox_annotype_list

                # Delete Mask if it exists
                mask_paths = self._safe_parse_list(self.gui.all_annotations.at[df_index, 'masks'])
                try:
                    if self.listbox_index < len(mask_paths):
                        path_to_remove = mask_paths[self.listbox_index]
                    self.mask_handler.delete_mask(path_to_remove)
                except:
                    pass

                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    self.rect_start = (x1, y1)
                    self.rect_end = (x2, y2)
                    self.create_resize_handles() 
                else:
                    print(f"[ERROR] Unexpected coords for rect_id: {coords}")
                    return
            except IndexError:
                print(f"[ERROR] rect_id index {self.listbox_index} out of range.")
                return


    def clear_polygon_from_canvas(self):
        """Helper function to delete all polygon points and lines from the canvas."""
        for pid in getattr(self, 'polygon_point_ids', []):
            self.gui.frame_canvas.delete(pid)
        self.polygon_point_ids = []

        if hasattr(self, 'polygon_line_id') and self.polygon_line_id is not None:
            self.gui.frame_canvas.delete(self.polygon_line_id)
            self.polygon_line_id = None

    def create_resize_handles(self):
        """
        Draws four square handles on the rectangle corners for resizing.
        """
        if not self.rect_start or not self.rect_end:
            return

        self.remove_resize_handles()

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        size = self.resize_handle_size
        canvas = self.gui.frame_canvas

        self.resize_handles = [
            canvas.create_rectangle(x1 - size, y1 - size, x1 + size, y1 + size, fill="blue", tags="handle_tl"),
            canvas.create_rectangle(x2 - size, y1 - size, x2 + size, y1 + size, fill="blue", tags="handle_tr"),
            canvas.create_rectangle(x1 - size, y2 - size, x1 + size, y2 + size, fill="blue", tags="handle_bl"),
            canvas.create_rectangle(x2 - size, y2 - size, x2 + size, y2 + size, fill="blue", tags="handle_br"),
        ]


    def remove_resize_handles(self):
        """
        Removes all resize handles from the canvas.
        """
        for handle in self.resize_handles:
            self.gui.frame_canvas.delete(handle)
        self.resize_handles = []

    def move_rectangle(self, x, y):
        """
        Moves the rectangle and updates its position and annotation.
        Only active in modify mode.
        """
        if not self.gui.modify_mode.get():
            return

        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        self.rect_start = (x1 + dx, y1 + dy)
        self.rect_end = (x2 + dx, y2 + dy)

        self.gui.frame_canvas.coords(self.rect_id, self.rect_start[0], self.rect_start[1], self.rect_end[0], self.rect_end[1])
        self.update_resize_handles()

    def resize_rectangle(self, x, y, handle):
        """
        Resizes the rectangle based on which corner handle is dragged.
        Only active in modify mode.
        """
        if not self.gui.modify_mode.get():
            return

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end

        if handle == "handle_tl":
            self.rect_start = (x, y)
        elif handle == "handle_tr":
            self.rect_start = (x1, y)
            self.rect_end = (x, y2)
        elif handle == "handle_bl":
            self.rect_start = (x, y1)
            self.rect_end = (x2, y)
        elif handle == "handle_br":
            self.rect_end = (x, y)

        self.gui.frame_canvas.coords(self.rect_id, self.rect_start[0], self.rect_start[1], self.rect_end[0], self.rect_end[1])
        self.update_resize_handles()


    def update_resize_handles(self):
        """
        Refreshes the resize handles to match the new rectangle position.
        """
        self.remove_resize_handles()
        self.create_resize_handles()

    def is_inside_rectangle(self, x, y):
        """
        Checks if (x, y) is inside the current rectangle.
        """

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end

        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        return x1 <= x <= x2 and y1 <= y <= y2
        

    def get_handle_at_position(self, x, y):
        """
        Returns the handle tag name if the position (x, y) is near any handle.
        """
        if not self.rect_start or not self.rect_end:
            return None

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        size = self.resize_handle_size

        if abs(x - x1) <= size and abs(y - y1) <= size:
            return "handle_tl"
        if abs(x - x2) <= size and abs(y - y1) <= size:
            return "handle_tr"
        if abs(x - x1) <= size and abs(y - y2) <= size:
            return "handle_bl"
        if abs(x - x2) <= size and abs(y - y2) <= size:
            return "handle_br"

        return None



    def update_annotation_in_listbox(self):
        """
        Updates the annotation DataFrame and listbox entry after rectangle or polygon modification.
        Uses self.selected_annotation_original_index and self.listbox_index.
        """
        if self.selected_annotation_original_index is None:
            print("[Update Error] No annotation was selected for modification (original index is None).")
            return

        if self.selected_annotation_original_index not in self.gui.all_annotations.index:
            print(f"[Update Error] Index {self.selected_annotation_original_index} not found in DataFrame.")
            return
    

        annotation_text = self.gui.video_annotation_listbox.get(self.listbox_index)
        class_label = annotation_text.split(" ")[0]

        # ======= Polygon =======
        if "Polygon" in annotation_text:
            
            polygon_list = self._safe_parse_list(self.row.get("polygon"))
            if self.polygon_index >= len(polygon_list):
                print(f"[Update Error] Polygon index {self.listbox_index} out of range.")
                return

            # save new polygon points
            polygon_list[self.polygon_index] = self.polygon_points.copy()
            self.gui.all_annotations.at[self.selected_annotation_original_index, "polygon"] = polygon_list

            updated_text = f"{str(class_label).ljust(12)} Polygon: {len(self.polygon_points)} points"
            self.gui.video_annotation_listbox.delete(self.listbox_index)
            self.gui.video_annotation_listbox.insert(self.listbox_index, updated_text)
            self.gui.video_annotation_listbox.selection_set(self.listbox_index)
            self.gui.video_annotation_listbox.activate(self.listbox_index)

            print(f"[INFO] Updated Polygon annotation at index {self.listbox_index}")
            return

        # ======= Bounding Box =======
        try:
            new_x, new_y, new_w, new_h = self.calculate_rectangle()
        except TypeError:
            print("[Update Error] Could not calculate rectangle – rect_start or rect_end is None.")
            return

        x_list = self._safe_parse_list(self.row.get("x"))
        y_list = self._safe_parse_list(self.row.get("y"))
        w_list = self._safe_parse_list(self.row.get("w"))
        h_list = self._safe_parse_list(self.row.get("h"))

        if self.listbox_index >= len(x_list) or self.listbox_index >= len(y_list) or \
        self.listbox_index >= len(w_list) or self.listbox_index >= len(h_list):
            print(f"[Update Error] annotation index {self.listbox_index} out of range in DataFrame lists.")
            return

        x_list[self.listbox_index] = new_x
        y_list[self.listbox_index] = new_y
        w_list[self.listbox_index] = new_w
        h_list[self.listbox_index] = new_h

        self.gui.all_annotations.at[self.selected_annotation_original_index, 'x'] = x_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'y'] = y_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'w'] = w_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'h'] = h_list


        updated_text = f"{str(class_label).ljust(12)} x:{str(new_x).ljust(5)} y:{str(new_y).ljust(5)} w:{str(new_w).ljust(5)} h:{str(new_h).ljust(5)}"

        self.gui.video_annotation_listbox.delete(self.listbox_index)
        self.gui.video_annotation_listbox.insert(self.listbox_index, updated_text)
        self.gui.video_annotation_listbox.selection_set(self.listbox_index)
        self.gui.video_annotation_listbox.activate(self.listbox_index)

        print(f"[INFO] Updated Bounding Box annotation at index {self.listbox_index}")



    def update_video_listbox_with_annotation_colors(self):
        """
        Update the video listbox with color-coded background indicating annotation progress.
        
        This method provides comprehensive visual feedback in the video listbox by coloring
        each video entry based on its annotation completion status:
        - Red background: No frames annotated (0 annotated frames)
        - Orange background: Partially annotated (some but not all frames annotated)
        - Green background: Fully annotated (all frames have annotations)
        
        The method analyzes annotation data across all video frames to determine progress,
        counts total available frames per video, and applies appropriate background colors
        to help users quickly identify annotation progress and prioritize work.
        
        Video-specific considerations:
        - Counts frames per video using frame file naming conventions
        - Tracks annotation progress across frame sequences
        - Integrates with video frame extraction system
        
        Side Effects:
            - Clears and repopulates the GUI video listbox
            - Applies background colors based on annotation progress
            - Updates internal frame count tracking
            - Validates DataFrame structure and handles missing columns
        """
        # Clear existing video listbox entries
        self.gui.video_listbox.delete(0, tk.END)

        # Construct path to video folder for current patient and exam
        image_folder = os.path.join(
            self.gui.selected_image_folder,
            self.gui.patient_id,
            self.gui.selected_exam
        )

        self.image_folder = image_folder    

        # Get list of valid video files
        videos = [
            i for i in os.listdir(image_folder)
            if i.lower().endswith(('.mp4'))
        ]

        # Load annotation data from DataFrame
        annotated_df = self.gui.all_annotations

        # Validate that DataFrame contains required annotation columns
        needed_columns = ['img_ID', 'class', 'x', 'y', 'w', 'h', 'polygon', 'class_polygon']
        missing_columns = [col for col in needed_columns if col not in annotated_df.columns]
        if missing_columns:
            print(f"[ERROR] Annotation DataFrame is missing necessary columns: {missing_columns}")
            return

        # Normalize annotation data - ensure empty/NaN values are consistently marked as None
        for col in ['class', 'class_polygon']:
            annotated_df[col] = annotated_df[col].apply(
                lambda x: None if (not hasattr(x, '__len__') and pd.isna(x)) or (hasattr(x, '__len__') and len(x) == 0) else x
            )
        
        # Count annotated frames per video by analyzing frame IDs
        annotated_frame_counts = annotated_df.loc[
                (pd.notna(annotated_df['class'])) | (pd.notna(annotated_df['class_polygon'])),
                'img_ID'
            ].astype(str).str.strip()

        # Get total frame counts for all videos
        self.get_number_of_frames_for_videos()

        # Process each video and apply color coding based on annotation progress
        for video in videos:
            video_id = video.split(".")[0]
            total_frames = self.number_of_frames.get(video_id, 0)
            annotated_frames = annotated_frame_counts.str.startswith(video_id + "_").sum()

            self.gui.video_listbox.insert(tk.END, video)
            index = self.gui.video_listbox.size() - 1

            # Apply progressive color coding based on annotation completion
            if annotated_frames == 0:
                color = '#fcd4d4'  # Red - no annotations
            elif annotated_frames < total_frames:
                color = '#fcf3d4'  # Orange - partially annotated
            else:
                color = '#d4fcd4'  # Green - fully annotated

            self.gui.video_listbox.itemconfig(index, {'bg': color})

    def get_number_of_frames_for_videos(self):
        """
        Count available frames in the current video folder for each video.
        
        This method analyzes the video folder to determine how many frame files
        exist for each video, which is essential for calculating annotation
        progress percentages. It uses file naming conventions to associate
        frame files with their parent videos.
        
        The method stores results in self.number_of_frames as a dictionary
        mapping video IDs to their frame counts, enabling efficient progress
        tracking without repeated file system operations.
        
        Side Effects:
            - Updates self.number_of_frames dictionary with video frame counts
            - Performs file system analysis of video folder structure
        """
        number_of_frames = {}

        # Get list of video files in current folder
        videos = [
            i for i in os.listdir(self.image_folder)
            if i.lower().endswith(('.mp4'))
        ]

        # Count frames for each video by analyzing file naming patterns
        for video in videos:
            video_id = video.split(".")[0]
            
            # Count files that contain video ID and "frame" in their name
            amount_of_frames = len([
                file for file in os.listdir(self.image_folder)
                if video_id in file and "frame" in file
            ])
            number_of_frames[video_id] = amount_of_frames

        # Store frame counts for progress calculation
        self.number_of_frames = number_of_frames

    def get_all_video_frames(self, patient_id, selected_exam):
        """
        Get all frame files for the current selected video.
        
        DEPRECATED: This method is replaced by VideoFrameExtractor.get_video_frames_intelligent().
        Maintained only for compatibility with legacy code.
        
        This method manually searches for frame files based on video ID and file
        naming conventions. The newer VideoFrameExtractor provides more robust
        and intelligent frame management with better error handling.
        
        Args:
            patient_id (str): Patient identifier (unused in current implementation)
            selected_exam (str): Exam identifier (unused in current implementation)
            
        Returns:
            list: Sorted list of frame filenames for the selected video
            
        Side Effects:
            - Updates self.current_frames with frame list
        """
        # Extract video ID from selected video filename
        video_id = self.gui.selected_video_index.split(".")[0]

        # Find all frame files associated with this video ID
        current_frames = sorted([
            file for file in os.listdir(self.image_folder) 
            if video_id in file and "frame" in file 
        ])
        
        self.current_frames = current_frames
        return current_frames

    def display_current_frame(self):
        """
        Display the current frame of the selected video with annotations.
        
        This method handles the complete workflow for displaying a video frame:
        - Loads the current frame image using VideoFrameExtractor
        - Resizes frame to fit display requirements
        - Updates the frame canvas with the new image
        - Loads and displays existing annotations for the frame
        - Updates frame navigation controls and labels
        
        Uses VideoFrameExtractor for intelligent frame path resolution,
        ensuring compatibility with different storage structures and
        temporary frame management systems.
        
        Side Effects:
            - Updates frame canvas with new image
            - Loads frame annotations via load_annotations_for_frame()
            - Updates mask visibility based on current settings
            - Updates frame navigation UI elements
            - Clears existing visual elements before loading new frame
        """
        
        # Clear existing visual elements from frame canvas
        for rect_id in self.drawn_rect_ids:
            self.gui.frame_canvas.delete(rect_id)
        self.drawn_rect_ids.clear()

        # Clear all mask overlays from previous frame
        self.mask_handler.clear_all_masks()

        # Validate that frame data is available
        if not hasattr(self.gui, 'current_frames') or not self.gui.current_frames:
            print("[WARNING] No frames available for display")
            return

        # Get current frame filename from navigation state
        current_frame_name = self.gui.current_frames[self.gui.current_frame_index]
        
        # Use VideoFrameExtractor for intelligent frame path resolution
        frame_path = self.gui.video_frame_extractor.get_frame_path(
            patient_id=self.gui.patient_id,
            selected_exam=self.gui.selected_exam,
            selected_video=self.gui.selected_video_index,
            frame_filename=current_frame_name,
            image_folder=self.image_folder
        )
        
        self.frame_path = frame_path
        # print(f"[DEBUG] Loading frame: {frame_path}")
        
        try:
            # Load and process frame image
            pil_frame = Image.open(frame_path)

            # Resize frame to fit display canvas while maintaining aspect ratio
            width, height = self.image_size
            pil_frame = pil_frame.resize((width, height), Image.Resampling.LANCZOS)

            # Convert to tkinter-compatible format
            self.tk_frame = ImageTk.PhotoImage(pil_frame)
            x = 0 
            y = 0

            # Update frame canvas with new image
            if not hasattr(self, 'frame_on_canvas') or self.frame_on_canvas is None:
                self.frame_on_canvas = self.gui.frame_canvas.create_image(x, y, anchor="nw", image=self.tk_frame)
            else:
                self.gui.frame_canvas.itemconfig(self.frame_on_canvas, image=self.tk_frame)

            # Load and display annotations for the current frame
            self.load_annotations_for_frame()
            
            # Apply current mask visibility settings
            self.gui.toggle_mask_visibility()
            
            # Update frame navigation UI elements
            self.gui.frame_index_label.config(text=f"Frame {self.gui.current_frame_index + 1} / {len(self.gui.current_frames)}")
            self.gui.video_slider.config(to=len(self.gui.current_frames)-1)
            
        except Exception as e:
            print(f"[ERROR] Error loading frame '{frame_path}': {e}")

    def redraw_polygon(self):
        """
        Redraw polygon outline connecting all current polygon points.
        
        This method updates the visual representation of a polygon being drawn or
        edited by creating a new polygon line connecting all points. It removes
        any existing polygon lines and creates a closed polygon shape on the
        frame canvas.
        
        Video-specific considerations:
        - Uses frame_canvas for video display
        - Integrates with video annotation workflows
        
        The polygon is drawn with a unique canvas tag for proper identification
        and management during editing operations.
        """
        # Remove any existing polygon lines with the current polygon's tag
        canvas_tag = f"polygon_{self.polygon_index}"
        self.gui.frame_canvas.delete(canvas_tag)

        # Only draw polygon if there are at least 2 points
        if len(self.polygon_points) >= 2:
            # Flatten coordinate pairs into single list for tkinter
            flat_points = [coord for point in self.polygon_points for coord in point]
            
            # Close polygon by adding first point at the end
            flat_points += self.polygon_points[0]

            # Create polygon outline on frame canvas
            self.polygon_line_id = self.gui.frame_canvas.create_polygon(
                flat_points, outline="blue", fill="", width=2, tags=(canvas_tag, "polygon")
            )

    def delete_last_polygon_point(self):
        """
        Delete the last point of a polygon annotation from frame canvas and internal state.
        
        This method provides undo functionality for polygon drawing by removing
        the most recently added point. It updates both the visual representation
        on the frame canvas and internal coordinate tracking, then redraws the
        polygon with the remaining points.
        
        Video-specific considerations:
        - Works with frame_canvas for video display
        - Integrates with video polygon annotation workflows
        
        Commonly triggered by keyboard shortcuts like Ctrl+Z during polygon creation.
        """
        # Check if there are points to delete
        if not self.polygon_points or not self.polygon_point_ids:
            print("[INFO] No polygon point to delete.")
            return

        # Remove the last point from frame canvas display
        last_id = self.polygon_point_ids.pop()
        try:
            self.gui.frame_canvas.delete(last_id)
        except Exception as e:
            print(f"[WARN] Could not delete canvas item: {e}")

        # Remove the point from coordinate list
        self.polygon_points.pop()

        # Redraw polygon with remaining points
        self.redraw_polygon()



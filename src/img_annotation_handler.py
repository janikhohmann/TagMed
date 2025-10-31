"""
Image Annotation Handler - Handling of image annotations for TagMed application


Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""


import tkinter as tk
import pandas as pd
import statistics as stat
import os
import ast
from tkinter import  ttk, messagebox

from annotation_loader import AnnotationLoader
from image_mask_predictor import ImageMaskPredictor
from mask_handler import MaskHandler


class ImgAnnotationHandler:
    """
    Handles image annotation functionality for the TagMed medical annotation application.
    
    This class provides comprehensive annotation capabilities including:
    - Bounding box annotations with interactive resizing and moving
    - Polygon annotations with point-based drawing and editing
    - Integration with mask prediction and handling systems
    - GUI interaction management for annotation workflows
    
    The handler manages both drawing new annotations and modifying existing ones,
    maintaining synchronization between the visual canvas elements and the
    underlying annotation data stored in pandas DataFrames.
    """
    def __init__(self, gui):
        """
        Initialize the image annotation handler.
        
        Args:
            gui: Main GUI instance providing access to canvas, annotation data,
                 and UI controls for the medical annotation interface.
        """
        self.gui = gui

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
        self.drawn_rect_ids = []         # List of all rectangle canvas IDs for current image

        # Polygon-specific attributes for polygon annotation mode
        self.polygon_points = []      # List of [x, y] coordinate pairs for current polygon
        self.polygon_point_ids = []   # Canvas IDs for polygon point visual elements
        self.polygon_line_id = None   # Canvas ID for polygon outline
        self.selected_point_index = None  # Index of currently selected polygon point
        self.listbox_index = None     # Index of selected annotation in GUI listbox
        self.polygon_index = None     # Index of polygon within annotation data
        self.is_drawing_polygon = False  # Flag for polygon drawing mode

        # Initialize dependent handler components
        self.annotation_loader = AnnotationLoader()           # Handles annotation data persistence
        self.image_mask_predictor = ImageMaskPredictor(gui)   # Manages mask prediction functionality
        self.mask_handler = MaskHandler(gui)                  # Handles mask visualization and management


    def add_annotation(self):
        """
        Add a new annotation to the current image based on the selected annotation type.
        
        Supports two annotation types:
        1. Bounding Box: Creates rectangular annotations with optional mask prediction
        2. Polygon: Creates polygon annotations from manually drawn points
        
        The method validates compatibility with existing annotations (prevents mixing
        bounding box and polygon annotations on the same image), updates the annotation
        DataFrame, refreshes the GUI listbox, and optionally generates masks.
        
        Side Effects:
            - Updates self.gui.all_annotations DataFrame with new annotation data
            - Adds visual representation to the canvas
            - Updates annotation listbox in GUI
            - May trigger mask prediction if enabled
        """
        # Extract annotation parameters from GUI controls
        img_selected_class = self.gui.img_selected_class.get()  # Medical class/category
        img_annotation_type = self.gui.img_annotation_type.get()  # Annotation type (BB/Polygon)
        image_id = self.gui.selected_image_index.split(".")[0]   # Image identifier without extension
        self.image_id = image_id

        # Validate that annotation DataFrame contains required image ID column
        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return

        # Search for existing row matching current image ID
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
            
            # Create formatted text representation for GUI listbox display
            annotation_text = f"{img_selected_class.ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
            self.gui.img_annotation_listbox.insert(tk.END, annotation_text)

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

            self.gui.image_canvas.itemconfig(self.rect_id, tags=("boundingbox",)) # Set permanent tag
            
            # Generate mask prediction if mask creation is enabled
            if self.gui.create_mask_var.get():
                self.image_mask_predictor.predict_mask_for_image_bb(idx, x, y, w, h)

            print(f"[DEBUG] Added Bounding Box with rect_id {self.rect_id}")

        # === Polygon Annotation Processing ===
        elif img_annotation_type == "Polygon":
            # Validate annotation type compatibility - prevent mixing BB and polygon annotations
            class_value = self.gui.all_annotations.at[idx, "class"]
            if pd.notna(class_value):  # Check if bounding box annotations already exist
                self.gui.wrong_annotation_warning_gui("Polygon")
                self.delete_all_polygons()
                return
            
            # Validate that polygon points have been collected
            if not hasattr(self, "polygon_points") or not self.polygon_points:
                print("[ERROR] No polygon points available.")
                return

            # Create formatted text representation for GUI listbox display
            annotation_text = f"{img_selected_class.ljust(12)} Polygon: {len(self.polygon_points)} points"
            self.gui.img_annotation_listbox.insert(tk.END, annotation_text)

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
            
            # Generate mask prediction for polygon if mask creation is enabled
            if self.gui.create_mask_var.get():
                width, height = self.gui.image_size
                self.image_mask_predictor.predict_mask_for_image_polygon(idx, self.polygon_points.copy(), width, height)

            print(f"[DEBUG] Added Polygon with {len(self.polygon_points)} points")

            # Clear polygon points after successful storage to prepare for next annotation
            self.polygon_points.clear()

        # === Error Handling for Unsupported Annotation Types ===
        else:
            print(f"[ERROR] Unknown annotation type: {img_annotation_type}")
            return

        # Update visual representation of annotation status in image listbox
        self.update_image_listbox_with_annotation_colors()

    def delete_annotation(self):
        """
        Delete the selected annotation from both the visual canvas and annotation data.
        
        Handles deletion of both bounding box and polygon annotations, including:
        - Removing visual elements from canvas (rectangles, polygons, points)
        - Updating annotation DataFrame by removing corresponding entries
        - Managing mask data associated with deleted annotations
        - Refreshing GUI listbox and visual representations
        - Proper cleanup of modify mode state if active
        
        The method automatically determines annotation type and performs
        appropriate cleanup for the specific annotation format.
        """
        # Get currently selected annotation from GUI listbox
        selected = self.gui.img_annotation_listbox.curselection()
        if not selected:
            return

        # Handle cleanup if currently in modify mode
        if self.gui.modify_mode.get():  # Clean up modify mode state
            self.gui.modify_mode.set(False)
            self.remove_resize_handles()
            self.update_annotation_in_listbox()
            # Clean up polygon editing elements
            for pid in getattr(self, 'polygon_point_ids', []):
                self.gui.image_canvas.delete(pid)
                self.polygon_point_ids = []
                self.polygon_points = []

        # Extract deletion parameters
        index = selected[0]  # Position in the listbox
        image_id = self.gui.selected_image_index.split(".")[0]

        # Validate annotation DataFrame structure
        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return

        # Find matching row in annotation DataFrame for current image
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
            
            # Retrieve associated mask data for potential cleanup
            masks_list = self._safe_parse_list(self.gui.all_annotations.at[idx, 'masks'])

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
                    
                # Delete associated mask if it exists
                if index < len(masks_list):
                    mask_to_delete = masks_list[index]
                    self.mask_handler.delete_mask(mask_to_delete)
                    masks_list.pop(index)

                # Update DataFrame with modified lists (use None if lists become empty)
                self.gui.all_annotations.at[idx, 'polygon'] = polygons if polygons else None
                self.gui.all_annotations.at[idx, 'class_polygon'] = class_list if class_list else None
                self.gui.all_annotations.at[idx, 'polygon_annotype'] = annotype_list if annotype_list else None
                self.gui.all_annotations.at[idx, 'masks'] = masks_list if masks_list else None

            # === Bounding Box Annotation Deletion ===
            else:
                # Remove data from all bounding box-related columns
                for col in ['x', 'y', 'w', 'h', 'class', 'bb_annotype', 'masks']:
                    val = self._safe_parse_list(self.gui.all_annotations.at[idx, col])

                    # Delete associated mask before removing mask reference
                    if col == 'masks' and index < len(val):
                        mask_to_delete = masks_list[index]
                        self.mask_handler.delete_mask(mask_to_delete)

                    # Remove entry at specified index
                    if index < len(val):
                        val.pop(index)
                    
                    # Update DataFrame column with modified list
                    self.gui.all_annotations.at[idx, col] = val if val else None

            # === Visual Element Cleanup ===
            # Remove visual element from canvas (rectangle or polygon)
            if index < len(self.drawn_rect_ids):
                shape_id = self.drawn_rect_ids[index]
                self.gui.image_canvas.delete(shape_id)
                self.drawn_rect_ids.pop(index)
            
            # Clear all masks from canvas display
            self.mask_handler.clear_all_masks()

            # Remove resize handles if present
            for handle in self.resize_handles:
                self.gui.image_canvas.delete(handle)
            self.resize_handles.clear()

            # Remove entry from GUI listbox
            self.gui.img_annotation_listbox.delete(index)

            # Reload and redraw all annotations for current image
            self.load_annotations_for_image(image_id)
            
            # Update visual annotation status indicators in image listbox
            self.update_image_listbox_with_annotation_colors()
            print(f"[DEBUG] Deleted annotation at index {index} from image {image_id}.")

        except Exception as e:
            print(f"[ERROR] Failed to delete annotation: {e}")


    def load_annotations_for_image(self, image_id):
        """
        Load and display all annotations (bounding boxes and polygons) for the specified image.
        
        This method reconstructs the visual representation of all stored annotations
        for a given image by reading from the annotation DataFrame and creating
        corresponding canvas elements. It handles both bounding box and polygon
        annotations, validating data integrity and skipping invalid entries.
        
        Args:
            image_id (str): Image identifier without file extension
            
        Side Effects:
            - Clears all existing visual annotations from canvas
            - Creates new rectangle and polygon canvas elements
            - Populates GUI annotation listbox with formatted entries
            - Updates drawn_rect_ids list for canvas element tracking
        """
        # Clear all existing annotations from canvas and GUI
        self.clear_all_annotations()
        print(f"[DEBUG] Handler: Loading annotations for image_id: '{image_id}'")

        df = self.gui.all_annotations

        # Filter annotation DataFrame for the specified image ID
        filtered_df = df[df['img_ID'].astype(str).str.strip() == str(image_id).strip()]

        if filtered_df.empty:
            print("[INFO] No annotations found for this image.")
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

                # Create rectangle on canvas with red outline
                rect_id = self.gui.image_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
                self.drawn_rect_ids.append(rect_id)

                # Create formatted text entry for GUI listbox
                annotation_text = f"{str(class_label).ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
                self.gui.img_annotation_listbox.insert(tk.END, annotation_text)

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
                
                # Create polygon on canvas with blue outline and unique tag
                polygon_id = self.gui.image_canvas.create_polygon(
                    flat_points, outline="blue", fill="", width=2, tags=f"polygon_{j}"
                )

                self.drawn_rect_ids.append(polygon_id)

                # Get class label or use default
                label = polygon_class_list[j] if j < len(polygon_class_list) else "unknown"
                
                # Create formatted text entry for GUI listbox
                annotation_text = f"{label.ljust(12)} Polygon: {len(polygon)} points"
                self.gui.img_annotation_listbox.insert(tk.END, annotation_text)

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
        Handle mouse press events for annotation drawing and modification.
        
        This method manages the initiation of annotation workflows based on the current
        annotation mode (Bounding Box, Polygon, Magic Wand) and whether the system is
        in drawing mode or modify mode. It coordinates between different annotation
        types and interaction states.
        
        Args:
            event: Tkinter mouse event containing coordinates and button information
            
        Behavior:
            - Drawing Mode: Initiates new annotation creation
            - Modify Mode: Begins editing of existing annotations
            - Validates annotation type compatibility and mode states
        """

        # remove any existing temporary bounding box by tag
        self.gui.image_canvas.delete("temp_boundingbox")

        # remove crosshair lines when bounding box or polygon drawing starts
        self.gui.image_canvas.bind(self.gui.image_canvas.delete("crosshair_line"))

        x, y = event.x, event.y  # Extract mouse position from event
        annotation_mode = self.gui.img_annotation_type.get()  # Get current annotation type

        # Validate supported annotation types
        if annotation_mode not in ["Bounding Box", "Polygon"]:
            print("[ERROR] Invalid annotation type selected. Only 'Bounding Box' and 'Polygon' are supported.")
            return

        # === Drawing Mode - Create New Annotations ===   
        if not self.gui.modify_mode.get():
            if annotation_mode == "Bounding Box":
                # Initialize bounding box drawing
                self.is_drawing = True
                self.rect_start = (x, y)
                self.temp_rect_id = self.gui.image_canvas.create_rectangle(
                    x, y, x, y, outline="red", width=2, tags="temp_boundingbox")
                    
            else:  # Polygon
                # Calculate polygon index for proper canvas element organization
                polygon_index = 0
                for i in range(self.gui.img_annotation_listbox.size()):
                    entry = self.gui.img_annotation_listbox.get(i)
                    if "Polygon" in entry:
                        polygon_index += 1
                self.polygon_index = polygon_index


                # Add clicked point to polygon and create visual marker
                self.polygon_points.append([x, y])
                point_id = self.gui.image_canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", tags="polygon")
                self.polygon_point_ids.append(point_id)
                self.redraw_polygon()  # Update polygon outline

        # === Modify Mode - Edit Existing Annotations ===
        # Determine annotation type of selected item for proper modification handling
        try:
            selected_text = self.gui.img_annotation_listbox.get(self.listbox_index)
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
                # === Polygon Modification ===
                # Check if user clicked near a polygon point for editing (6 pixel tolerance)
                for i, (px, py) in enumerate(self.polygon_points):
                    if abs(event.x - px) < 6 and abs(event.y - py) < 6:
                        self.selected_point_index = i
                        return

    def on_drag(self, event):
        """
        Handle mouse drag events for continuous annotation drawing and modification.
        
        This method manages real-time updates during mouse movement with button pressed.
        It handles different behaviors based on annotation mode and interaction state:
        - Drawing mode: Updates visual representation during creation
        - Modify mode: Handles resizing, moving, and point manipulation
        
        Args:
            event: Tkinter mouse motion event with current coordinates
        """
        x, y = event.x, event.y  # Extract current mouse position
        annotation_mode = self.gui.img_annotation_type.get()  # Get annotation type

        # === Drawing Mode - Real-time Annotation Creation ===
        if not self.gui.modify_mode.get():
            if annotation_mode == "Bounding Box":
                # Update rectangle coordinates during drawing
                # use the temporary rectangle id while drawing (temp_rect_id)
                if self.is_drawing and self.temp_rect_id:
                    self.gui.image_canvas.coords(
                        self.temp_rect_id,
                        self.rect_start[0], self.rect_start[1],
                        x, y
                    )
                    
            else:  # Polygon
                # Add point during continuous polygon drawing
                self.polygon_points.append([x, y])
                point_id = self.gui.image_canvas.create_oval(x-4, y-4, x+4, y+4, fill="red")
                self.polygon_point_ids.append(point_id)
                self.redraw_polygon()  # Update polygon outline

        # === Modify Mode - Handle Annotation Editing ===
        # Determine annotation type for appropriate modification behavior
        try:
            selected_text = self.gui.img_annotation_listbox.get(self.listbox_index)
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

                    # Update visual representation of the point on canvas
                    point_id = self.polygon_point_ids[self.selected_point_index]
                    self.gui.image_canvas.coords(point_id, x-4, y-4, x+4, y+4)

                    # Redraw polygon with updated point position
                    self.redraw_polygon()

    def on_release(self, event):
        """
        Handle mouse button release events to finalize annotation operations.
        
        This method completes annotation creation or modification operations:
        - Drawing mode: Finalizes new annotation coordinates
        - Modify mode: Clears interaction state and handles
        
        Args:
            event: Tkinter mouse button release event
        """
        x, y = event.x, event.y  # Extract mouse position
        annotation_mode = self.gui.img_annotation_type.get()  # Get annotation type

        # === Finalize Bounding Box Operations ===
        if annotation_mode == "Bounding Box":
            if not self.gui.modify_mode.get():
                # Complete new bounding box drawing
                if self.is_drawing:
                    self.rect_end = (x, y)
                    self.is_drawing = False
            else:
                # Clear modify mode interaction state
                self.dragging_handle = None
                self.dragging_rectangle = False
                self.last_mouse_pos = None

        # === Finalize Polygon Operations ===
        if annotation_mode == "Polygon":
            # Clear selected point index for polygon editing
            self.selected_point_index = None

    def calculate_rectangle(self):
        """
        Calculate bounding box parameters from drawn rectangle coordinates.
        
        Converts corner coordinates from drawing operation into center coordinates
        and dimensions suitable for annotation storage. Handles coordinate ordering
        to ensure consistent results regardless of drawing direction.
        
        Returns:
            tuple: (center_x, center_y, width, height) as integers
                   representing the bounding box in annotation format
        """
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end

        # Ensure coordinates are ordered correctly (top-left to bottom-right)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Calculate center coordinates and dimensions
        canvas_x = round(stat.mean([x1, x2]))
        canvas_y = round(stat.mean([y1, y2]))
        canvas_width = abs(x2 - x1)
        canvas_height = abs(y2 - y1)

        return canvas_x, canvas_y, canvas_width, canvas_height

    def on_annotation_selected(self, event):
        """
        Handle selection of annotations from the GUI listbox.
        
        This method is triggered when a user clicks on an annotation entry in the
        listbox. It prepares the annotation for potential modification by:
        - Clearing any existing resize handles
        - Resetting modify mode state
        - Extracting annotation metadata and indices for later use
        - Calculating polygon indices for proper reference
        
        Args:
            event: Tkinter listbox selection event
            
        Side Effects:
            - Updates internal state variables for selected annotation
            - Clears visual modification elements from canvas
            - Prepares system for potential modification operations
        """
        # Clear any existing resize handles and reset modify mode
        self.remove_resize_handles()
        self.gui.modify_mode.set(False)

        # Get selected annotation from listbox
        selection = self.gui.img_annotation_listbox.curselection()
        if not selection:
            return
        listbox_index = selection[0]

        # Find corresponding DataFrame entry for current image
        current_image_id = self.gui.selected_image_index.split(".")[0].strip()
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == current_image_id
        if not match.any():
            print(f"[ERROR] No row found for img_ID = {current_image_id}")
            return
        df_index = self.gui.all_annotations[match].index[0]

        # Store annotation metadata for modification operations
        self.selected_annotation_original_index = df_index
        self.row = self.gui.all_annotations.loc[df_index]
        self.listbox_index = listbox_index

        # Calculate polygon index for proper polygon reference
        polygon_index = -1
        for i in range(listbox_index + 1):
            entry = self.gui.img_annotation_listbox.get(i)
            if "Polygon" in entry:
                polygon_index += 1
        self.polygon_index = polygon_index

    def clear_all_annotations(self):
        """
        Clear all annotation visual elements from canvas and reset internal state.
        
        This comprehensive cleanup method removes:
        - All rectangle and polygon visual elements from canvas
        - Resize handles and modification tools
        - Polygon points and lines
        - GUI listbox entries
        - Internal state variables and tracking lists
        
        Used when switching between images or refreshing annotation display.
        """
        """Clear all rectangles, polygons from canvas and the listbox, including rect_id entries in the DataFrame."""

        # ==== Delte all rectangles from canvas ====
        for rect_id in self.drawn_rect_ids:
            self.gui.image_canvas.delete(rect_id)
        self.drawn_rect_ids.clear()

        # remove resize handles
        self.remove_resize_handles()

        # === Delete all polygons from canvas ===
        # delete lines
        if hasattr(self, 'polygon_line_id') and self.polygon_line_id is not None:
            self.gui.image_canvas.delete(self.polygon_line_id)
            self.polygon_line_id = None

        # delete points
        for pid in getattr(self, 'polygon_point_ids', []):
            self.gui.image_canvas.delete(pid)
        self.polygon_point_ids = []
        self.polygon_points = []

        # to make sure everything is deleted
        self.delete_all_polygons()

        # empty listbox 
        self.gui.img_annotation_listbox.delete(0, tk.END)
        
        # set internal state back to initial
        self.selected_point_index = None
        self.rect_id = None
        self.listbox_index = None


    def delete_all_polygons(self):
        self.gui.image_canvas.delete("polygon")  # deletes everything with the tag "polygon"
        self.polygon_point_ids = []
        self.polygon_points = []
        self.polygon_line_id = None
        # print("[DEBUG] All polygon elements have been deleted.")

    def delete_all_bounding_boxes(self):
        self.gui.image_canvas.delete("boundingbox")  # deletes all with the tag "boundingbox"
        self.drawn_rect_ids = []
        # print("[DEBUG] All bounding box elements have been deleted.")


        
    def modify_annotation(self):
        """Modifies selected annotation, handling both bounding boxes and polygons."""
        if not self.gui.modify_mode.get():
            self.remove_resize_handles()
            self.update_annotation_in_listbox()
            for pid in getattr(self, 'polygon_point_ids', []):
                self.gui.image_canvas.delete(pid)
                self.polygon_point_ids = []
                self.polygon_points = []
            return
    
        selection = self.gui.img_annotation_listbox.curselection()
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

        current_image_id = self.gui.selected_image_index.split(".")[0].strip()

        # Find the matching row in the DataFrame
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == current_image_id
        if not match.any():
            print(f"[ERROR] No row found for img_ID = {current_image_id}")
            return

        df_index = self.gui.all_annotations[match].index[0]
        row = self.gui.all_annotations.loc[df_index]
        self.row = row  # Save for later use
        self.selected_annotation_original_index = df_index

        # Polygon or Bounding Box?
        selected_text = self.gui.img_annotation_listbox.get(listbox_index)
        is_polygon = "Polygon" in selected_text

        if is_polygon:
            # ----- POLYGON-ANNOTATION MODIFICATION -----

            self.remove_resize_handles()  

            for pid in getattr(self, 'polygon_point_ids', []):
                self.gui.image_canvas.delete(pid)
            self.polygon_point_ids = []
            self.polygon_points = []


            if hasattr(self, 'polygon_index') and self.polygon_index is not None:
                tag_to_delete = f"polygon_{self.polygon_index}"
                self.gui.image_canvas.delete(tag_to_delete)


            # get polygon from DataFrame
            polygon_list = self._safe_parse_list(row.get('polygon'))
            if self.polygon_index >= len(polygon_list):
                print(f"[ERROR] Polygon index {self.polygon_index} out of range.")
                return

            polygon = polygon_list[self.polygon_index]
            if not isinstance(polygon, list) or len(polygon) < 3:
                print("[ERROR] Invalid polygon data.")
                return

            # New points drawing
            for x, y in polygon:
                self.polygon_points.append([x, y])
                point_id = self.gui.image_canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4, fill="red"
                )
                self.polygon_point_ids.append(point_id)


            # Polygon line drawing
            self.redraw_polygon()

            print(f"[DEBUG] Polygon mit {len(polygon)} Punkten geladen.")
        else:
            try:
                self.rect_id = self.drawn_rect_ids[self.listbox_index]
                coords = self.gui.image_canvas.coords(self.rect_id)

                # parse bb_annotype as list
                bbox_annotype_list = self._safe_parse_list(row.get('bb_annotype'))


                bbox_annotype_list[self.listbox_index] = "manually"
                self.gui.all_annotations.at[df_index, 'bb_annotype'] = bbox_annotype_list

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
        # Deletes all polygon points and lines from the canvas.
        for pid in getattr(self, 'polygon_point_ids', []):
            self.gui.image_canvas.delete(pid)
        self.polygon_point_ids = []

        if hasattr(self, 'polygon_line_id') and self.polygon_line_id is not None:
            self.gui.image_canvas.delete(self.polygon_line_id)
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
        canvas = self.gui.image_canvas

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
            self.gui.image_canvas.delete(handle)
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

        self.gui.image_canvas.coords(self.rect_id, self.rect_start[0], self.rect_start[1], self.rect_end[0], self.rect_end[1])
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

        self.gui.image_canvas.coords(self.rect_id, self.rect_start[0], self.rect_start[1], self.rect_end[0], self.rect_end[1])
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

        annotation_text = self.gui.img_annotation_listbox.get(self.listbox_index)
        class_label = annotation_text.split(" ")[0]

        # ======= Polygon =======
        if "Polygon" in annotation_text:
            polygon_list = self._safe_parse_list(self.row.get("polygon"))
            if self.polygon_index >= len(polygon_list):
                print(f"[Update Error] Polygon index {self.listbox_index} out of range.")
                return

            # Neue Polygon-Koordinaten speichern
            polygon_list[self.polygon_index] = self.polygon_points.copy()
            self.gui.all_annotations.at[self.selected_annotation_original_index, "polygon"] = polygon_list

            updated_text = f"{str(class_label).ljust(12)} Polygon: {len(self.polygon_points)} points"
            self.gui.img_annotation_listbox.delete(self.listbox_index)
            self.gui.img_annotation_listbox.insert(self.listbox_index, updated_text)
            self.gui.img_annotation_listbox.selection_set(self.listbox_index)
            self.gui.img_annotation_listbox.activate(self.listbox_index)

            # safe new mask for polygon
            width, height = self.gui.image_size
            self.image_mask_predictor.predict_mask_for_image_polygon(None, self.polygon_points.copy(), width, height)

            # print(f"[INFO] Updated Polygon annotation at index {self.polygon_index}")
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
        
        self.image_mask_predictor.predict_mask_for_image_bb(None, new_x,new_y,new_w,new_h)

        x_list[self.listbox_index] = new_x
        y_list[self.listbox_index] = new_y
        w_list[self.listbox_index] = new_w
        h_list[self.listbox_index] = new_h
        #path_list[self.listbox_index] = mask_path

        self.gui.all_annotations.at[self.selected_annotation_original_index, 'x'] = x_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'y'] = y_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'w'] = w_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'h'] = h_list

        # print(class_label)

        updated_text = f"{str(class_label).ljust(12)} x:{str(new_x).ljust(5)} y:{str(new_y).ljust(5)} w:{str(new_w).ljust(5)} h:{str(new_h).ljust(5)}"

        self.gui.img_annotation_listbox.delete(self.listbox_index)
        self.gui.img_annotation_listbox.insert(self.listbox_index, updated_text)
        self.gui.img_annotation_listbox.selection_set(self.listbox_index)
        self.gui.img_annotation_listbox.activate(self.listbox_index)


        # print(f"[INFO] Updated Bounding Box annotation at index {self.listbox_index}")

    

    def update_image_listbox_with_annotation_colors(self):
        """
        Update the image listbox with color-coded background indicating annotation status.
        
        This method provides visual feedback in the main image listbox by coloring
        each image entry based on its annotation status:
        - Green background: Image has annotations (class ≠ NN or empty)
        - Red background: Image has no annotations
        
        The method reads annotation data from the DataFrame and determines whether
        each image has bounding box or polygon annotations, then applies appropriate
        background colors to help users quickly identify annotation progress.
        
        Side Effects:
            - Clears and repopulates the GUI image listbox
            - Applies background colors based on annotation status
            - Validates DataFrame structure and handles missing columns
        """
        # Clear existing image listbox entries
        self.gui.image_listbox.delete(0, tk.END)

        # Construct path to image folder for current patient and exam
        image_folder = os.path.join(
            self.gui.selected_image_folder,
            self.gui.patient_id,
            self.gui.selected_exam
        )

        # Get list of valid image files (excluding video frames)
        images = [
            i for i in os.listdir(image_folder)
            if i.lower().endswith(('.png', '.jpg', '.jpeg')) and "frame" not in i.lower()
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

        # Identify all images that have annotations (either bounding box or polygon)
        annotated_image_ids = set(
            annotated_df.loc[
                (pd.notna(annotated_df['class'])| 
                (pd.notna(annotated_df['class_polygon']))),
                'img_ID'
            ].astype(str).str.strip()
        )

        # Populate listbox with images and apply color coding based on annotation status
        for image in images:
            image_id = image.split(".")[0].strip()
            self.gui.image_listbox.insert(tk.END, image)
            index = self.gui.image_listbox.size() - 1

            # Apply color coding based on annotation status
            if image_id in annotated_image_ids:
                self.gui.image_listbox.itemconfig(index, {'bg': '#d4fcd4'})  # Green - annotated
            else:
                self.gui.image_listbox.itemconfig(index, {'bg': '#fcd4d4'})  # Red - not annotated

    def redraw_polygon(self):
        """
        Redraw polygon outline connecting all current polygon points.
        
        This method updates the visual representation of a polygon being drawn or
        edited by creating a new polygon line connecting all points. It removes
        any existing polygon lines and creates a closed polygon shape.
        
        The polygon is drawn with a unique canvas tag for proper identification
        and management during editing operations.
        """
        # Remove any existing polygon lines with the current polygon's tag
        canvas_tag = f"polygon_{self.polygon_index}"
        self.gui.image_canvas.delete(canvas_tag)

        # Only draw polygon if there are at least 2 points
        if len(self.polygon_points) >= 2:
            # Flatten coordinate pairs into single list for tkinter
            flat_points = [coord for point in self.polygon_points for coord in point]
            
            # Close polygon by adding first point at the end
            flat_points += self.polygon_points[0]

            # Create polygon outline on canvas
            self.polygon_line_id = self.gui.image_canvas.create_polygon(
                flat_points, outline="blue", fill="", width=2, tags=(canvas_tag, "polygon")
            )

    def delete_last_polygon_point(self):
        """
        Delete the last point of a polygon annotation from canvas and internal state.
        
        This method provides undo functionality for polygon drawing by removing
        the most recently added point. It updates both the visual representation
        and internal coordinate tracking, then redraws the polygon with the
        remaining points.
        
        Commonly triggered by keyboard shortcuts like Ctrl+Z during polygon creation.
        """
        # Check if there are points to delete
        if not self.polygon_points or not self.polygon_point_ids:
            print("[INFO] No polygon point to delete.")
            return

        # Remove the last point from canvas display
        last_id = self.polygon_point_ids.pop()
        try:
            self.gui.image_canvas.delete(last_id)
        except Exception as e:
            print(f"[WARN] Could not delete canvas item: {e}")

        # Remove the point from coordinate list
        self.polygon_points.pop()

        # Redraw polygon with remaining points
        self.redraw_polygon()



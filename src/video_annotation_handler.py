import pandas as pd
import os
import ast
import tkinter as tk 
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import statistics as stat
import numpy as np

from annotation_loader import AnnotationLoader


class VideoAnnotationHandler():
    def __init__(self, gui):
        self.gui = gui

        self.is_drawing = False
        self.rect_start = None
        self.rect_end = None
        self.rect_id = None
        self.resize_handles = []
        self.resize_handle_size = 3

        self.selected_annotation_index = None

        self.dragging_handle = None
        self.dragging_rectangle = False
        self.last_mouse_pos = None
        self.drawn_rect_ids = []

        # Polygon-specific attributes
        self.polygon_points = []
        self.polygon_point_ids = []
        self.polygon_line_id = None
        self.selected_point_index = None
        self.listbox_index = None
        self.polygon_index = None
        self.is_drawing_polygon = False


        self.drawn_mask_ids = []

        self.annotation_loader = AnnotationLoader()

    def add_annotation(self):
        img_selected_class = self.gui.video_selected_class.get()
        img_annotation_type = self.gui.video_annotation_type.get()
        image_id = self.current_image_id 

        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return

        # Search for row with Image-ID
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(image_id).strip()
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{image_id}'")
            return
        idx = self.gui.all_annotations[match].index[0] # return index of row in dataframe

        # initialize cells 
        def append_or_init_list(cell, value):
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


        # === Bounding Box Annotation ===
        if img_annotation_type == "Bounding Box":
            if self.gui.all_annotations.at[idx, "polygon"] != "NN": # safe exit when already an polygon annotation exists
                self.gui.wrong_annotation_warning_gui("Bounding Box")
                self.delete_all_bounding_boxes()
                return
            x, y, w, h = self.calculate_rectangle()
            annotation_text = f"{img_selected_class.ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
            self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

            for col, val in zip(['x', 'y', 'w', 'h', 'class', 'bb_annotype'], [x, y, w, h, img_selected_class, 'manually']):
                self.gui.all_annotations.at[idx, col] = append_or_init_list(self.gui.all_annotations.at[idx, col], val)

            self.drawn_rect_ids.append(self.rect_id)
            print(f"[DEBUG] Added Bounding Box with rect_id {self.rect_id}")

        # === Polygon Annotation ===
        elif img_annotation_type == "Polygon":
            if self.gui.all_annotations.at[idx, "class"] != "NN": # safe exit when already an BB annotation exists
                self.gui.wrong_annotation_warning_gui("Polygon")
                self.delete_all_polygons()
                return
    
            if not hasattr(self, "polygon_points") or not self.polygon_points:
                print("[ERROR] No polygon points available.")
                return

            annotation_text = f"{img_selected_class.ljust(12)} Polygon: {len(self.polygon_points)} points"
            self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

            # save polygon points in DataFrame
            if 'polygon' not in self.gui.all_annotations.columns:
                self.gui.all_annotations['polygon'] = None  # initalize column if not exists

            polygon_copy = [point.copy() for point in self.polygon_points]
            self.gui.all_annotations.at[idx, 'polygon'] = append_or_init_list(
                self.gui.all_annotations.at[idx, 'polygon'], polygon_copy)
            self.gui.all_annotations.at[idx, 'class_polygon'] = append_or_init_list(
                self.gui.all_annotations.at[idx, 'class_polygon'], img_selected_class)
            self.gui.all_annotations.at[idx, 'polygon_annotype'] = append_or_init_list(
                self.gui.all_annotations.at[idx, 'polygon_annotype'], "manually")

            print(f"[DEBUG] Added Polygon with {len(self.polygon_points)} points")

            # clean list after adding to dataframe
            self.polygon_points.clear()

        else:
            print(f"[ERROR] Unknown annotation type:: {img_annotation_type}")
            return

        self.update_video_listbox_with_annotation_colors()

    def delete_annotation(self):
        selected = self.gui.video_annotation_listbox.curselection()
        if not selected:
            return
        
        if self.gui.modify_mode.get(): # abfangen falls im modify modus
            self.gui.modify_mode.set(False)
            self.remove_resize_handles()
            self.update_annotation_in_listbox()
            for pid in getattr(self, 'polygon_point_ids', []):
                self.gui.frame_canvas.delete(pid)
                self.polygon_point_ids = []
                self.polygon_points = []

        index = selected[0]  # Position in der Listbox
        image_id = self.current_image_id 

        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return

        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(image_id).strip()
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{image_id}'")
            return

        idx = self.gui.all_annotations[match].index[0]

        try:
            row = self.gui.all_annotations.loc[idx]

            # Prüfe, ob Polygon-Spalte existiert und der Eintrag polygonbasiert ist
            is_polygon = False
            if 'polygon' in row:
                polygons = self._safe_parse_list(row['polygon'])
                if index < len(polygons):
                    is_polygon = True

            if is_polygon:
                # Entferne Polygon-Daten
                polygons = self._safe_parse_list(self.gui.all_annotations.at[idx, 'polygon'])
                class_list = self._safe_parse_list(self.gui.all_annotations.at[idx, 'class_polygon'])
                annotype_list = self._safe_parse_list(self.gui.all_annotations.at[idx, 'polygon_annotype'])

                polygons.pop(index)
                if index < len(class_list):
                    class_list.pop(index)
                if index < len(annotype_list):
                    annotype_list.pop(index)

                self.gui.all_annotations.at[idx, 'polygon'] = polygons if polygons else "NN"
                self.gui.all_annotations.at[idx, 'class_polygon'] = class_list if class_list else "NN"
                self.gui.all_annotations.at[idx, 'polygon_annotype'] = annotype_list if annotype_list else "NN"
            else:
                # Entferne Bounding Box-Daten
                for col in ['x', 'y', 'w', 'h', 'class', 'bb_annotype']:
                    val = self._safe_parse_list(self.gui.all_annotations.at[idx, col])
                    if index < len(val):
                        val.pop(index)
                    self.gui.all_annotations.at[idx, col] = val if val else "NN"


            # Entferne visuelles Element vom Canvas
            if index < len(self.drawn_rect_ids):
                shape_id = self.drawn_rect_ids[index]
                self.gui.frame_canvas.delete(shape_id)
                self.drawn_rect_ids.pop(index)

            # Resize-Handles löschen
            for handle in self.resize_handles:
                self.gui.frame_canvas.delete(handle)
            self.resize_handles.clear()

            # Entferne Listbox-Eintrag
            self.gui.video_annotation_listbox.delete(index)

            self.delete_all_polygons()
            self.load_annotations_for_frame()

            all_items = self.gui.frame_canvas.find_all()
            print("Alle Elemente auf dem Canvas:")

            for item in all_items:
                tags = self.gui.frame_canvas.gettags(item)
                coords = self.gui.frame_canvas.coords(item)
                print(f"ID: {item}, Tags: {tags}, Koordinaten: {coords}")

            self.update_video_listbox_with_annotation_colors()
            print(f"[DEBUG] Deleted annotation at index {index} from image {image_id}.")

        except Exception as e:
            print(f"[ERROR] Failed to delete annotation: {e}")



    def load_annotations_for_frame(self):
        """Load all annotations (list-style) for the selected frame."""
        self.clear_all_annotations()

        current_image_id = self.gui.current_frames[self.gui.current_frame_index].split(".")[0]
        self.current_image_id = current_image_id

        print(f"[DEBUG] Handler: Loading annotations for image_id: '{self.current_image_id }'")

        df = self.gui.all_annotations

        # Filter nach Bild-ID
        filtered_df = df[df['img_ID'].astype(str).str.strip() == str(self.current_image_id ).strip()]

        if filtered_df.empty:
            print("[INFO] No annotations found for this image.")
            return

        if 'rect_id' not in self.gui.all_annotations.columns:
            self.gui.all_annotations['rect_id'] = None

        idx = filtered_df.index[0]
        row = filtered_df.iloc[0]

        # --- Bounding Box Daten extrahieren ---
        x_list = self._safe_parse_list(row.get('x'))
        y_list = self._safe_parse_list(row.get('y'))
        w_list = self._safe_parse_list(row.get('w'))
        h_list = self._safe_parse_list(row.get('h'))
        class_list = self._safe_parse_list(row.get('class'))

        # --- Polygon Daten extrahieren ---
        polygon_list = self._safe_parse_list(row.get('polygon')) if 'polygon' in row else []
        polygon_class_list = self._safe_parse_list(row.get('class_polygon')) if 'class_polygon' in row else []


        # --- Bounding Boxes zeichnen ---
        count = min(len(x_list), len(y_list), len(w_list), len(h_list), len(class_list))
        for i in range(count):
            try:
                x, y, w, h = int(x_list[i]), int(y_list[i]), int(w_list[i]), int(h_list[i])
                class_label = class_list[i]

                if w <= 0 or h <= 0:
                    print(f"[WARN] Skipping invalid bounding box [{i}] (w or h <= 0)")
                    continue

                x1, y1 = x - w // 2, y - h // 2
                x2, y2 = x + w // 2, y + h // 2

                rect_id = self.gui.frame_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
                self.drawn_rect_ids.append(rect_id)

                annotation_text = f"{str(class_label).ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
                self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

                print(f"[DEBUG] Drew bounding box rect_id {rect_id} at [{x1}, {y1}, {x2}, {y2}]")

            except Exception as e:
                print(f"[ERROR] Could not draw bounding box {i}: {e}")

        # --- Polygone zeichnen ---
        for j, polygon in enumerate(polygon_list):
            try:
                if not isinstance(polygon, list) or len(polygon) < 3:
                    print(f"[WARN] Skipping invalid polygon [{j}]")
                    continue

                flat_points = [coord for point in polygon for coord in point]
                polygon_id = self.gui.frame_canvas.create_polygon(
                    flat_points, outline="blue", fill="", width=2, tags=f"polygon_{j}"
                )

                self.drawn_rect_ids.append(polygon_id)

                label = polygon_class_list[j] if j < len(polygon_class_list) else "unknown"
                annotation_text = f"{label.ljust(12)} Polygon: {len(polygon)} points"
                self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

                print(f"[DEBUG] Drew polygon with id {polygon_id}, {len(polygon)} points")

            except Exception as e:
                print(f"[ERROR] Could not draw polygon {j}: {e}")



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
    
    
    def on_press(self, event):
        """
        Starts drawing or modifying a rectangle or polygon depending on mode.
        """
        x, y = event.x, event.y # get mouse position
        # get annotation type from GUI
        annotation_mode = self.gui.video_annotation_type.get()

        if annotation_mode not in ["Bounding Box", "Polygon"]:
            print("[ERROR] Invalid annotation type selected. Only 'Bounding Box' and 'Polygon' are supported.")
            return
        


        # ==== draw mode ====   
        if not self.gui.modify_mode.get():
            if annotation_mode == "Bounding Box":
                # draw new rectangle
                self.is_drawing = True
                self.rect_start = (x, y)
                self.rect_id = self.gui.frame_canvas.create_rectangle(
                    x, y, x, y, outline="red", width=2, tags="boundingbox")
            else:
                #if self.polygon_index is None:
                    # get correct polygon index from listbox
                polygon_index = 0
                for i in range(self.gui.video_annotation_listbox.size()):
                    entry = self.gui.video_annotation_listbox.get(i)
                    if "Polygon" in entry:
                        polygon_index += 1
                self.polygon_index = polygon_index

                self.polygon_points.append([x, y])
                point_id = self.gui.frame_canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", tags="polygon")
                self.polygon_point_ids.append(point_id)
                self.redraw_polygon()
        
        # Entscheide, ob es sich um Bounding Box oder Polygon handelt
        try:
            selected_text = self.gui.video_annotation_listbox.get(self.listbox_index)
            is_polygon = "Polygon" in selected_text
        except: 
            return
        
        # ==== annotation mode ====
        if self.gui.modify_mode.get(): 
            if not is_polygon:
                # start modify mode for rectangle
                self.dragging_handle = self.get_handle_at_position(x, y)
                self.dragging_rectangle = self.is_inside_rectangle(x, y) and not self.dragging_handle
                self.last_mouse_pos = (x, y)
            else:
                # Check if a point is selected for modification
                for i, (px, py) in enumerate(self.polygon_points):
                    if abs(event.x - px) < 6 and abs(event.y - py) < 6:
                        self.selected_point_index = i
                        return
                    

                    
    def on_drag(self, event):
        """
        Continues drawing or modifying the rectangle while the mouse is dragged.
        """
        x, y = event.x, event.y # get mouse position
        # get annotation type from GUI
        annotation_mode = self.gui.video_annotation_type.get()


        # ==== draw mode ====
        if not self.gui.modify_mode.get():
            if annotation_mode == "Bounding Box":
                if self.is_drawing and self.rect_id:
                    self.gui.frame_canvas.coords(
                        self.rect_id,
                        self.rect_start[0], self.rect_start[1],
                        x, y
                    )
            else: 
                self.polygon_points.append([x, y])
                point_id = self.gui.frame_canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", tags="polygon")
                self.polygon_point_ids.append(point_id)
                self.redraw_polygon()



        # Entscheide, ob es sich um Bounding Box oder Polygon handelt
        try:
            selected_text = self.gui.video_annotation_listbox.get(self.listbox_index)
            is_polygon = "Polygon" in selected_text
        except: 
            return

        # ==== modify mode ====
        if self.gui.modify_mode.get():
            if not is_polygon:
                if self.dragging_handle:
                    self.resize_rectangle(x, y, self.dragging_handle)
                elif self.dragging_rectangle:
                    self.move_rectangle(x, y)
                self.last_mouse_pos = (x, y)
            else:
                if self.selected_point_index is not None:
                    self.polygon_points[self.selected_point_index] = [x, y]

                    # Update Punktkreis auf Canvas
                    point_id = self.polygon_point_ids[self.selected_point_index]
                    self.gui.frame_canvas.coords(point_id, x-4, y-4, x+4, y+4)

                    self.redraw_polygon()
                

    def on_release(self, event):
        """
        Finishes drawing or modifying. Automatically updates the annotation.
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

        # Entferne alle rect_id Einträge aus dem DataFrame für das aktuelle Bild - can this be deleted?  bezieht sich auf unterschiedliches in img and video
        # current_image = self.gui.current_frame_index
        # mask = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(current_image).strip()
        # if 'rect_id' in self.gui.all_annotations.columns:
        #     self.gui.all_annotations.loc[mask, 'rect_id'] = None

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
        self.gui.frame_canvas.delete("polygon")  # entfernt alle mit dem Tag "polygon"
        self.polygon_point_ids = []
        self.polygon_points = []
        self.polygon_line_id = None
        #print("[DEBUG] Alle Polygon-Elemente wurden gelöscht.")

    def delete_all_bounding_boxes(self):
        self.gui.frame_canvas.delete("boundingbox")  # entfernt alle mit dem Tag "boundingbox"
        self.drawn_rect_ids = []
        #print("[DEBUG] Alle BoundingBox-Elemente wurden gelöscht.")



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
        self.row = row  # Zur späteren Nutzung speichern
        self.selected_annotation_original_index = df_index

        # Entscheide, ob es sich um Bounding Box oder Polygon handelt
        selected_text = self.gui.video_annotation_listbox.get(listbox_index)
        is_polygon = "Polygon" in selected_text

        if is_polygon:
            # ----- POLYGON-ANNOTATION BEARBEITEN -----

            # Alte Handles, Punkte und Polygon-Linie löschen
            self.remove_resize_handles()  # Entfernt ggf. alte Bounding Box Handles

            for pid in getattr(self, 'polygon_point_ids', []):
                self.gui.frame_canvas.delete(pid)
                self.polygon_point_ids = []
                self.polygon_points = []


            if hasattr(self, 'polygon_index') and self.polygon_index is not None:
                tag_to_delete = f"polygon_{self.polygon_index}"
                self.gui.frame_canvas.delete(tag_to_delete)


            # Polygon aus dem DataFrame holen
            polygon_list = self._safe_parse_list(row.get('polygon'))
            if self.polygon_index >= len(polygon_list):
                print(f"[ERROR] Polygon index {self.polygon_index} out of range.")
                return

            polygon = polygon_list[self.polygon_index]
            if not isinstance(polygon, list) or len(polygon) < 3:
                print("[ERROR] Invalid polygon data.")
                return

            # Neue Punkte zeichnen
            for x, y in polygon:
                self.polygon_points.append([x, y])
                point_id = self.gui.frame_canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4, fill="red"
                )
                self.polygon_point_ids.append(point_id)

            polygon_annotype_list = self._safe_parse_list(row.get('polygon_annotype'))
            polygon_annotype_list[self.polygon_index] = "manually"
            self.gui.all_annotations.at[df_index, 'polygon_annotype'] = polygon_annotype_list


            # Polygon-Linie zeichnen
            self.redraw_polygon()

            print(f"[DEBUG] Polygon mit {len(polygon)} Punkten geladen.")
        else:
            try:
                self.rect_id = self.drawn_rect_ids[self.listbox_index]
                coords = self.gui.frame_canvas.coords(self.rect_id)

                # bb_annotype als Liste parsen
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
        # Hilfsfunktion, um alle Polygonpunkte und Linien vom Canvas zu löschen
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

            # Neue Polygon-Koordinaten speichern
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



    def load_masks_for_frame(self):
        """Load masks for the selected frame into canvas."""
        self.clear_all_masks()

        found_count = 0 

        if self.gui.masks_visible.get():
            for index, mask_entry in enumerate(self.gui.all_masks):
                mask_img_id = mask_entry.get('img_ID', 'MISSING_ID')

                if str(mask_img_id).strip() == str(self.current_image_id ).strip():
                    found_count += 1

                    try:
                        mask_array = mask_entry['mask']  # uint8 NumPy-Array, value 0 or 255

                        # Create RGBA-Bild from mask
                        height, width = mask_array.shape
                        red_color = (255, 0, 0, 60)  # bright, transparent red (alpha=60/255)
                        rgba_array = np.zeros((height, width, 4), dtype=np.uint8)

                        rgba_array[mask_array > 0] = red_color  # only fill mask

                        # Convert in  in PIL-Image then in PhotoImage
                        pil_image = Image.fromarray(rgba_array, mode="RGBA")
                        tk_image = ImageTk.PhotoImage(pil_image)

                        # Show Image on Canvas
                        mask_id = self.gui.frame_canvas.create_image(0, 0, anchor="nw", image=tk_image)

                        # save reference
                        if not hasattr(self.gui, 'mask_image_refs'):
                            self.gui.mask_image_refs = []
                        self.gui.mask_image_refs.append(tk_image)

                        self.gui.all_masks[index]['mask_id'] = mask_id
                        self.drawn_mask_ids.append(mask_id)

                    except Exception as e:
                        print(f"[ERROR] Failed to draw mask at index {index} (Error: {e}): {mask_entry}")
                        self.gui.all_masks[index]['mask_id'] = None
                        continue



    def clear_all_masks(self):
        """Clear all masks."""

        for mask_id in self.drawn_mask_ids:
            self.gui.frame_canvas.delete(mask_id)

        self.drawn_mask_ids.clear()

        if hasattr(self.gui, 'mask_image_refs'):
            self.gui.mask_image_refs.clear()

    def toggle_mask_visibility(self):

        if self.gui.masks_visible.get():
            print("[DEBUG] Masks will be shown.")
            self.load_masks_for_frame()
        else:
            print("[DEBUG] Mask will not be shown.")
            self.clear_all_masks()

    def update_video_listbox_with_annotation_colors(self):
        """
        Aktualisiert die Video-Listbox mit Hintergrundfarben abhängig vom Annotationsstatus.
        Grün = annotiert (class ≠ NN oder leer), Rot = nicht annotiert.
        """
        self.gui.video_listbox.delete(0, tk.END)

        image_folder = os.path.join(
            self.gui.selected_image_folder,
            self.gui.patient_id,
            self.gui.selected_exam
        )

        self.image_folder = image_folder    

        videos = [
            i for i in os.listdir(image_folder)
            if i.lower().endswith(('.mov'))
        ]

        # Lade Annotationen aus CSV oder gespeicherter Quelle
        annotated_df = self.gui.all_annotations

        # make sure the DataFrame has the necessary columns
        needed_columns = ['img_ID' , 'class', 'x', 'y', 'w', 'h', 'polygon', 'class_polygon']
        missing_columns = [col for col in needed_columns if col not in annotated_df.columns]
        if missing_columns:
            print(f"[ERROR] Annotation DataFrame is missing necessary columns: {missing_columns}")
            return

        # set default values for 'class' and 'class_polygon'
        for col in ['class', 'class_polygon']:
            annotated_df[col] = annotated_df[col].apply(
                lambda x: "NN" if (not hasattr(x, '__len__') and pd.isna(x)) or (hasattr(x, '__len__') and len(x) == 0) else x
            )
        
        # Zähle annotierte Frames pro Video
        annotated_frame_counts = annotated_df.loc[
                (annotated_df['class'] != "NN") | (annotated_df['class_polygon'] != "NN"),
                'img_ID'
            ].astype(str).str.strip()

        self.get_number_of_frames_for_videos()

        for video in videos:
            video_id = video.split(".")[0]
            total_frames = self.number_of_frames.get(video_id, 0)
            annotated_frames = annotated_frame_counts.str.startswith(video_id + "_").sum()

            #print(f"[DEBUG] Video: {video}, Total Frames: {total_frames}, Annotated Frames: {annotated_frames}")

            self.gui.video_listbox.insert(tk.END, video)
            index = self.gui.video_listbox.size() - 1

            # Farblogik
            if annotated_frames == 0:
                color = '#fcd4d4'  # rot
            elif annotated_frames < total_frames:
                color = '#fcf3d4'  # orange
            else:
                color = '#d4fcd4'  # grün

            self.gui.video_listbox.itemconfig(index, {'bg': color})

    def get_number_of_frames_for_videos(self):
        """
        Counts available frames in the current video folder for each video.
        Stores result in self.number_of_frames as dict: {video_id: frame_count}.
        """
        number_of_frames = {}


        videos = [
            i for i in os.listdir(self.image_folder)
            if i.lower().endswith(('.mov'))
        ]

        for video in videos:
            video_id = video.split(".")[0]
            amount_of_frames = len([
                file for file in os.listdir(self.image_folder)
                if video_id in file and "frame" in file
            ])
            number_of_frames[video_id] = amount_of_frames

        self.number_of_frames = number_of_frames



    def get_all_video_frames(self, patient_id, selected_exam):
        """
        Gets all frames for the current selected video.
        Returns list with frames.
        """

        video_id = self.gui.selected_video_index.split(".")[0]

        current_frames = sorted([
            file for file in os.listdir(self.image_folder) if video_id in file and "frame" in file 
        ])
        self.current_frames = current_frames
        return current_frames

    def display_current_frame(self):
        """
        Shows the current frame of the selected video.
        """
        
        #self.clear_all_annotations()
        for rect_id in self.drawn_rect_ids:
            self.gui.frame_canvas.delete(rect_id)
        self.drawn_rect_ids.clear()
        self.clear_all_masks()

        
        if not hasattr(self, 'current_frames') or not self.current_frames:
            return

        frame_path = os.path.join(self.image_folder, self.current_frames[self.gui.current_frame_index])
        self.frame_path = frame_path
        
        try:
            pil_frame = Image.open(frame_path)

            width = 600
            height = 600
            pil_frame = pil_frame.resize((width, height), Image.Resampling.LANCZOS)

            self.tk_frame = ImageTk.PhotoImage(pil_frame)
            x = 0 
            y = 0

            if not hasattr(self, 'frame_on_canvas') or self.frame_on_canvas is None:
                self.frame_on_canvas = self.gui.frame_canvas.create_image(x, y, anchor="nw", image=self.tk_frame)
            else:
                self.gui.frame_canvas.itemconfig(self.frame_on_canvas, image=self.tk_frame)

            # Load annotations for the current frame
            self.load_annotations_for_frame()
            self.load_masks_for_frame()
            
            # Update frame label
            self.gui.frame_index_label.config(text=f"Frame {self.gui.current_frame_index + 1} / {len(self.current_frames)}")
            self.gui.video_slider.config(to=len(self.current_frames)-1)
        except Exception as e:
            print(f"Error loading frame: {e}")

    def redraw_polygon(self):
        # Entferne alle alten Polygon-Linien
        canvas_tag = f"polygon_{self.polygon_index}"

        self.gui.frame_canvas.delete(canvas_tag)

        if len(self.polygon_points) >= 2:
            flat_points = [coord for point in self.polygon_points for coord in point]
            
            # Polygon schließen, indem man den ersten Punkt erneut anhängt
            flat_points += self.polygon_points[0]

            self.polygon_line_id = self.gui.frame_canvas.create_polygon(
                flat_points, outline="blue", fill="", width=2, tags=(canvas_tag, "polygon")
                )

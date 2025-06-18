import pandas as pd
import os
import ast
import tkinter as tk 
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import statistics as stat
import numpy as np

from config_handler import ConfigHandler
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
        self.rect_ids = []
        self.drawn_rect_ids = []
        self.drawn_mask_ids = []

        self.annotation_loader = AnnotationLoader()


    def add_annotation(self):
        img_selected_class = self.gui.video_selected_class.get()
        img_annotation_type = self.gui.video_annotation_type.get()

        x, y, w, h = self.calculate_rectangle()
        annotation_text = f"{img_selected_class.ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"

        self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return
        
        # find row with image_id
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(self.gui.current_frame_id)
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{self.gui.current_frame_id}'")
            print(self.gui.all_annotations['img_ID'])
            return
        idx = self.gui.all_annotations[match].index[0]

        # Initialize the list in the DataFrame cell
        def append_or_init_list(cell, value):

            if pd.isna(cell) or cell == "NN":
                return [value]
    
            # Falls String wie "[123, 456]", dann versuche in Liste zu parsen
            if isinstance(cell, str):
                try:
                    parsed = ast.literal_eval(cell)
                    if isinstance(parsed, list):
                        return parsed + [value]
                except (ValueError, SyntaxError):
                    pass
                return [cell, value]

            # Falls echte Liste
            if isinstance(cell, list):
                return cell + [value]

            # Fallback für Einzelwerte (int, float, etc.)
            return [cell, value]
    
        for col, val in zip(['x', 'y', 'w', 'h', 'class'], [x, y, w, h, img_selected_class]):
            self.gui.all_annotations.at[idx, col] = append_or_init_list(self.gui.all_annotations.at[idx, col], val)

        
        print(f"[DEBUG] Add Annotation: new rect with rect_id {self.rect_id} was added to internal list.")

        self.drawn_rect_ids.append(self.rect_id)
        print(f"[DEBUG] Added rect_id {self.rect_id} to drawn_rect_ids. List size now: {len(self.drawn_rect_ids)}")
        
        self.update_video_listbox_with_annotation_colors() # does not work yet


    def delete_annotation(self):

        selected = self.gui.video_annotation_listbox.curselection()
        if not selected:
            return

        index = selected[0]  # Position in der Listbox

        # Sicherstellen, dass 'img_ID' Spalte vorhanden ist
        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return

        # Passende Zeile zur Bild-ID finden
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(self.gui.current_frame_id).strip()
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{self.gui.current_frame_id}'")
            return

        idx = self.gui.all_annotations[match].index[0]

        try:
            # Spaltennamen
            cols = ['x', 'y', 'w', 'h', 'class']

            for col in cols:
                val = self.gui.all_annotations.at[idx, col]
                # Versuche, val in Liste umzuwandeln, wenn nötig
                if isinstance(val, str) and val.startswith('['):
                    val = ast.literal_eval(val)
                elif not isinstance(val, list):
                    val = []

                # Eintrag an 'index' entfernen, falls möglich
                if len(val) > index:
                    val.pop(index)

                # Zurückschreiben
                self.gui.all_annotations.at[idx, col] = val if val else "NN"

            # Rechteck entfernen
            if index < len(self.drawn_rect_ids):
                rect_id = self.drawn_rect_ids[index]
                self.gui.frame_canvas.delete(rect_id)
                self.drawn_rect_ids.pop(index)

            # Resize-Handles entfernen
            for handle in self.resize_handles:
                self.gui.frame_canvas.delete(handle)
            self.resize_handles.clear()

            # Listbox-Eintrag entfernen
            self.gui.video_annotation_listbox.delete(index)

            self.update_video_listbox_with_annotation_colors()
            print(f"[DEBUG] Deleted annotation at index {index} from image {self.gui.current_frame_id}.")

        except Exception as e:
            print(f"[ERROR] Failed to delete annotation: {e}")


    def load_annotations_for_frame(self):
        """Load all annotations (list-style) for the selected frame."""
        self.clear_all_annotations()
        #print(f"[DEBUG] Handler: Loading annotations for image_id: '{self.gui.current_frame_id}'")

        df = self.gui.all_annotations

        # Filter nach Bild-ID
        filtered_df = df[df['img_ID'].astype(str).str.strip() == str(self.gui.current_frame_id).strip()]

        if filtered_df.empty:
            print("[INFO] No annotations found for this image.")
            return

        # rect_id-Spalte sicherstellen
        if 'rect_id' not in self.gui.all_annotations.columns:
            self.gui.all_annotations['rect_id'] = None

        idx = filtered_df.index[0]
        row = filtered_df.iloc[0]

        # Hole Annotationen-Listen
        x_list = ast.literal_eval(row['x']) if isinstance(row['x'], str) and row['x'].startswith('[') else row['x'] if isinstance(row['x'], list) else []
        y_list = ast.literal_eval(row['y']) if isinstance(row['y'], str) and row['y'].startswith('[') else row['y'] if isinstance(row['y'], list) else []
        w_list = ast.literal_eval(row['w']) if isinstance(row['w'], str) and row['w'].startswith('[') else row['w'] if isinstance(row['w'], list) else []
        h_list = ast.literal_eval(row['h']) if isinstance(row['h'], str) and row['h'].startswith('[') else row['h'] if isinstance(row['h'], list) else []
        class_list = ast.literal_eval(row['class']) if isinstance(row['class'], str) and row['x'].startswith('[') else row['class'] if isinstance(row['class'], list) else []



        count = min(len(x_list), len(y_list), len(w_list), len(h_list), len(class_list))

        for i in range(count):
            try:
                x, y, w, h = int(x_list[i]), int(y_list[i]), int(w_list[i]), int(h_list[i])
                class_label = class_list[i]

                if w <= 0 or h <= 0:
                    print(f"[WARN] Skipping invalid annotation [{i}] (w or h <= 0)")
                    continue

                x1, y1 = x - w // 2, y - h // 2
                x2, y2 = x + w // 2, y + h // 2

                # Rechteck zeichnen
                rect_id = self.gui.frame_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
                self.rect_ids.append(rect_id)
                self.drawn_rect_ids.append(rect_id)

                # Listbox-Eintrag
                annotation_text = f"{str(class_label).ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
                self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

                #print(f"[DEBUG] Drew rect_id {rect_id} for annotation {i}")

            except Exception as e:
                print(f"[ERROR] Could not draw annotation {i}: {e}")
                continue

    def on_press(self, event):
        """
        Starts drawing or modifying a rectangle depending on mode.
        """

        if not self.gui.modify_mode.get():
            # Neuer Draw-Start
            self.is_drawing = True
            self.rect_start = (event.x, event.y)
            self.rect_id = self.gui.frame_canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="red", width=2)
        else:
            # Modify-Modus starten
            self.dragging_handle = self.get_handle_at_position(event.x, event.y)
            self.dragging_rectangle = self.is_inside_rectangle(event.x, event.y) and not self.dragging_handle
            self.last_mouse_pos = (event.x, event.y)


    def on_drag(self, event):
        """
        Continues drawing or modifying the rectangle while the mouse is dragged.
        """
        if not self.gui.modify_mode.get():
            if self.is_drawing and self.rect_id:
                self.gui.frame_canvas.coords(
                    self.rect_id,
                    self.rect_start[0], self.rect_start[1],
                    event.x, event.y
                )
        else:
            if self.dragging_handle:
                self.resize_rectangle(event.x, event.y, self.dragging_handle)
            elif self.dragging_rectangle:
                self.move_rectangle(event.x, event.y)
            self.last_mouse_pos = (event.x, event.y)


    def on_release(self, event):
        """
        Finishes drawing or modifying. Automatically updates the annotation.
        """
        if not self.gui.modify_mode.get():
            if self.is_drawing:
                self.rect_end = (event.x, event.y)
                self.is_drawing = False
        else:
            self.dragging_handle = None
            self.dragging_rectangle = False
            self.last_mouse_pos = None


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

        #self.remove_resize_handles()
        selection = self.gui.video_annotation_listbox.curselection()
        if not selection:
            return
        listbox_index = selection[0]
        
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == self.gui.current_frame_id
        if not match.any():
            print(f"[ERROR] No row found for img_ID = {self.gui.current_frame_id}")
            return
        df_index = self.gui.all_annotations[match].index[0]

        # Setze den DataFrame Index und row als Attribute, damit update es nutzen kann
        self.selected_annotation_original_index = df_index
        self.row = self.gui.all_annotations.loc[df_index]
        self.listbox_index = listbox_index

    def clear_all_annotations(self):
        """Clear all rectangles and the listbox, including rect_id entries in the DataFrame."""
        # Entferne alle Rechtecke vom Canvas

        # print("[DEBUG] Clearing all annotations from canvas and listbox.")
        # print(self.drawn_rect_ids)

        for rect_id in self.drawn_rect_ids:
            self.gui.frame_canvas.delete(rect_id)
        self.drawn_rect_ids.clear()

        self.gui.video_annotation_listbox.delete(0, tk.END)

        self.remove_resize_handles()


    def modify_annotation(self):

        if not self.gui.modify_mode.get():
            self.remove_resize_handles()
            self.update_annotation_in_listbox()
            return

        selection = self.gui.video_annotation_listbox.curselection()
        if not selection:
            print("Modify Error: No annotation selected in the listbox.")
            return

        # Filter die richtige Zeile (ein Bild = eine Zeile in df)
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == self.gui.current_frame_id
        if not match.any():
            print(f"[ERROR] No row found for img_ID = {self.gui.current_frame_id}")
            return

        df_index = self.gui.all_annotations[match].index[0]

        row = self.gui.all_annotations.loc[df_index]
        self.row = row
        
        x_list = ast.literal_eval(row['x']) if isinstance(row['x'], str) and row['x'].startswith('[') else row['x'] if isinstance(row['x'], list) else []
        y_list = ast.literal_eval(row['y']) if isinstance(row['y'], str) and row['y'].startswith('[') else row['y'] if isinstance(row['y'], list) else []
        w_list = ast.literal_eval(row['w']) if isinstance(row['w'], str) and row['w'].startswith('[') else row['w'] if isinstance(row['w'], list) else []
        h_list = ast.literal_eval(row['h']) if isinstance(row['h'], str) and row['h'].startswith('[') else row['h'] if isinstance(row['h'], list) else []
        class_list = ast.literal_eval(row['class']) if isinstance(row['class'], str) and row['x'].startswith('[') else row['class'] if isinstance(row['class'], list) else []

        try:
            # Zugriff auf die Annotation an der Stelle des Listbox-Eintrags
            x = x_list[self.listbox_index]
            y = y_list[self.listbox_index]
            w = w_list[self.listbox_index]
            h = h_list[self.listbox_index]
        except IndexError:
            print(f"[ERROR] Annotation index {self.listbox_index} out of range in DataFrame lists.")
            return

        # print(f"[DEBUG] rect_id: {self.rect_id}, x: {x}, y: {y}, w: {w}, h: {h}")
        # print(self.drawn_rect_ids)
        # Canvas-Element finden: du brauchst dafür einen rect_id-Cache pro Annotation (z.B. in extra Liste speichern)
        try:
            self.rect_id = self.drawn_rect_ids[self.listbox_index]
        except IndexError:
            print(f"[ERROR] rect_id index {self.listbox_index} out of range.")
            return


        try:
            
            coords = self.gui.frame_canvas.coords(self.rect_id)
            if not coords or len(coords) < 4:
                print(f"Invalid rect_id: {self.rect_id}, coords: {coords}")
                return
        except tk.TclError:
            print(f"rect_id {self.rect_id} is invalid (possibly deleted).")
            return

        # Setze Zustand für Änderung
        self.selected_annotation_df_index = df_index
        self.rect_start = (coords[0], coords[1])
        self.rect_end = (coords[2], coords[3])
        self.create_resize_handles()


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
        Updates the annotation DataFrame and listbox entry after rectangle modification.
        Uses self.selected_annotation_original_index, which must refer to a valid DataFrame index.
        """
        if self.selected_annotation_original_index is None:
            print("[Update Error] No annotation was selected for modification (original index is None).")
            return

        if self.selected_annotation_original_index not in self.gui.all_annotations.index:
            print(f"[Update Error] Index {self.selected_annotation_original_index} not found in DataFrame.")
            return

        # Neue Geometrie berechnen
        new_x, new_y, new_w, new_h = self.calculate_rectangle()
        
        # Bestehende Listen aus dem DataFrame lesen und parsen
        x_list = ast.literal_eval(self.row['x']) if isinstance(self.row['x'], str) and self.row['x'].startswith('[') else self.row['x'] if isinstance(self.row['x'], list) else []
        y_list = ast.literal_eval(self.row['y']) if isinstance(self.row['y'], str) and self.row['y'].startswith('[') else self.row['y'] if isinstance(self.row['y'], list) else []
        w_list = ast.literal_eval(self.row['w']) if isinstance(self.row['w'], str) and self.row['w'].startswith('[') else self.row['w'] if isinstance(self.row['w'], list) else []
        h_list = ast.literal_eval(self.row['h']) if isinstance(self.row['h'], str) and self.row['h'].startswith('[') else self.row['h'] if isinstance(self.row['h'], list) else []
        class_list = ast.literal_eval(self.row['class']) if isinstance(self.row['class'], str) and self.row['x'].startswith('[') else self.row['class'] if isinstance(self.row['class'], list) else []


        # Prüfen, ob annotation_idx gültig ist
        if self.listbox_index >= len(x_list) or self.listbox_index >= len(y_list) or self.listbox_index >= len(w_list) or self.listbox_index >= len(h_list):
            print(f"[Update Error] annotation index {self.listbox_index} out of range in DataFrame lists.")
            return

        # Listen an der Stelle annotation_idx aktualisieren
        x_list[self.listbox_index] = new_x
        y_list[self.listbox_index] = new_y
        w_list[self.listbox_index] = new_w
        h_list[self.listbox_index] = new_h

        print("DEBUG x list ", x_list)

        # Aktualisierte Listen zurück in DataFrame schreiben
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'x'] = x_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'y'] = y_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'w'] = w_list
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'h'] = h_list


        # Listbox-Eintrag aktualisieren (Klassen-Label hier einzeln aus der Liste holen)
        if self.listbox_index < len(class_list):
            class_label = class_list[self.listbox_index]
        else:
            class_label = "Unknown"

        selected_in_listbox = self.gui.video_annotation_listbox.curselection()
        if selected_in_listbox:
            updated_text = f"{str(class_label).ljust(12)} x:{str(new_x).ljust(5)} y:{str(new_y).ljust(5)} w:{str(new_w).ljust(5)} h:{str(new_h).ljust(5)}"

            self.gui.video_annotation_listbox.delete(self.listbox_index)
            self.gui.video_annotation_listbox.insert(self.listbox_index, updated_text)
            self.gui.video_annotation_listbox.selection_set(self.listbox_index)
            self.gui.video_annotation_listbox.activate(self.listbox_index)
        else:
            print("[Warning] Could not find selected item in listbox to update its text.")


    def load_masks_for_frame(self, image_id):
        """Load masks for the selected frame into canvas."""
        self.clear_all_masks()

        found_count = 0 

        if self.gui.masks_visible.get():
            for index, mask_entry in enumerate(self.gui.all_masks):
                mask_img_id = mask_entry.get('img_ID', 'MISSING_ID')

                if str(mask_img_id).strip() == str(image_id).strip():
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

        #print(f"[DEBUG] Loaded and drew {found_count} masks for frame {image_id}")



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
            self.load_masks_for_frame(self.gui.current_frame_id)
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
        annotated_df = self.annotation_loader.load_annotations_from_annotable()

        # Video-ID extrahieren aus img_ID → alles vor '_frame...'
        annotated_df['video_ID'] = annotated_df['img_ID'].apply(
            lambda x: x.split("_frame")[0] if "_frame" in x else x)
        
        # Zähle annotierte Frames pro Video
        annotated_frame_counts = (
            annotated_df[annotated_df['class'] != 'NN']['video_ID']
            .value_counts()
            .to_dict()
            )

        self.get_number_of_frames_for_videos()

        for video in videos:
            video_id = video.split(".")[0]
            total_frames = self.number_of_frames.get(video_id, 0)
            annotated_frames = annotated_frame_counts.get(video_id, 0)
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
            self.load_masks_for_frame(self.gui.current_frame_id)
            
            # Update frame label
            self.gui.frame_index_label.config(text=f"Frame {self.gui.current_frame_index + 1} / {len(self.current_frames)}")
            self.gui.video_slider.config(to=len(self.current_frames)-1)
        except Exception as e:
            print(f"Error loading frame: {e}")
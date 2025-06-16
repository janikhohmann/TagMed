import tkinter as tk
import pandas as pd
import statistics as stat
import os
import ast

from annotation_loader import AnnotationLoader


class ImgAnnotationHandler:
    def __init__(self, gui):
        self.gui = gui

        self.is_drawing = False
        self.rect_start = None
        self.rect_end = None
        self.rect_id = None
        self.resize_handles = []
        self.resize_handle_size = 3

        self.selected_annotation_index = None
        #self.modify_mode = False

        self.dragging_handle = None
        self.dragging_rectangle = False
        self.last_mouse_pos = None
        self.drawn_rect_ids = []

        self.annotation_loader = AnnotationLoader()


    def add_annotation(self):
        img_selected_class = self.gui.img_selected_class.get()
        img_annotation_type = self.gui.img_annotation_type.get()

        x, y, w, h = self.calculate_rectangle()
        annotation_text = f"{img_selected_class.ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"

        self.gui.img_annotation_listbox.insert(tk.END, annotation_text)
        image_id = self.gui.selected_image_index.split(".")[0]

        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return
        
        # find row with image_id
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(image_id).strip()
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{image_id}'")
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
        print(self.gui.all_annotations["x"])

        
        print(f"[DEBUG] Add Annotation: new rect with rect_id {self.rect_id} was added to internal list.")

        self.drawn_rect_ids.append(self.rect_id)
        print(f"[DEBUG] Added rect_id {self.rect_id} to drawn_rect_ids. List size now: {len(self.drawn_rect_ids)}")
        
        self.update_image_listbox_with_annotation_colors() # does not work yet


    def delete_annotation(self):
        selected = self.gui.img_annotation_listbox.curselection()
        if not selected:
            return

        index = selected[0]  # Position in der Listbox
        image_id = self.gui.selected_image_index.split(".")[0]

        # Sicherstellen, dass 'img_ID' Spalte vorhanden ist
        if 'img_ID' not in self.gui.all_annotations.columns:
            print("[ERROR] 'img_ID' column not found in DataFrame.")
            return

        # Passende Zeile zur Bild-ID finden
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(image_id).strip()
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{image_id}'")
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
                self.gui.image_canvas.delete(rect_id)
                self.drawn_rect_ids.pop(index)

            # Resize-Handles entfernen
            for handle in self.resize_handles:
                self.gui.image_canvas.delete(handle)
            self.resize_handles.clear()

            # Listbox-Eintrag entfernen
            self.gui.img_annotation_listbox.delete(index)

            self.update_image_listbox_with_annotation_colors()
            print(f"[DEBUG] Deleted annotation at index {index} from image {image_id}.")

        except Exception as e:
            print(f"[ERROR] Failed to delete annotation: {e}")




    def load_annotations_for_image(self, image_id):
        """Load all annotations (list-style) for the selected image."""
        self.clear_all_annotations()
        print(f"[DEBUG] Handler: Loading annotations for image_id: '{image_id}'")

        df = self.gui.all_annotations

        # Filter nach Bild-ID
        filtered_df = df[df['img_ID'].astype(str).str.strip() == str(image_id).strip()]

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

        print(f"[DEBUG] x_list: {x_list}")


        count = min(len(x_list), len(y_list), len(w_list), len(h_list), len(class_list))
        rect_ids = []

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
                rect_id = self.gui.image_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
                rect_ids.append(rect_id)
                self.drawn_rect_ids.append(rect_id)

                # Listbox-Eintrag
                annotation_text = f"{str(class_label).ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
                self.gui.img_annotation_listbox.insert(tk.END, annotation_text)

                print(f"[DEBUG] Drew rect_id {rect_id} for annotation {i}")

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
            self.rect_id = self.gui.image_canvas.create_rectangle(
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
                self.gui.image_canvas.coords(
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
        #self.gui.modify_mode = tk.BooleanVar(value=False)  # Turns off modify mode when selecting an annotation
        # self.remove_resize_handles()
        # selection = self.gui.img_annotation_listbox.curselection()
        # if selection:
        #     self.selected_annotation_original_index = selection[0]
        #     #self.modify_mode = True
        self.remove_resize_handles()
        selection = self.gui.img_annotation_listbox.curselection()
        if not selection:
            return
        listbox_index = selection[0]

        current_image_id = self.gui.selected_image_index.split(".")[0].strip()
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == current_image_id
        if not match.any():
            print(f"[ERROR] No row found for img_ID = {current_image_id}")
            return
        df_index = self.gui.all_annotations[match].index[0]

        # Setze den DataFrame Index und row als Attribute, damit update es nutzen kann
        self.selected_annotation_original_index = df_index
        self.row = self.gui.all_annotations.loc[df_index]
        self.listbox_index = listbox_index


    def clear_all_annotations(self):
        """Clear all rectangles and the listbox, including rect_id entries in the DataFrame."""
        # Entferne alle Rechtecke vom Canvas
        for rect_id in self.drawn_rect_ids:
            self.gui.image_canvas.delete(rect_id)
        self.drawn_rect_ids.clear()

        # Entferne alle rect_id Einträge aus dem DataFrame für das aktuell ausgewählte Bild
        current_image = self.gui.selected_image_index.split(".")[0]
        mask = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(current_image).strip()
        if 'rect_id' in self.gui.all_annotations.columns:
            self.gui.all_annotations.loc[mask, 'rect_id'] = None

        # Listbox leeren
        self.gui.img_annotation_listbox.delete(0, tk.END)

        # Resize-Handles entfernen
        self.remove_resize_handles()


    def modify_annotation(self):

        if not self.gui.modify_mode.get():
            self.remove_resize_handles()
            self.update_annotation_in_listbox()
            return

        selection = self.gui.img_annotation_listbox.curselection()
        if not selection:
            print("Modify Error: No annotation selected in the listbox.")
            return

        current_image_id = self.gui.selected_image_index.split(".")[0].strip()

        # Filter die richtige Zeile (ein Bild = eine Zeile in df)
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == current_image_id
        if not match.any():
            print(f"[ERROR] No row found for img_ID = {current_image_id}")
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

        print(f"[DEBUG] rect_id: {self.rect_id}, x: {x}, y: {y}, w: {w}, h: {h}")
        print(self.drawn_rect_ids)
        # Canvas-Element finden: du brauchst dafür einen rect_id-Cache pro Annotation (z.B. in extra Liste speichern)
        try:
            self.rect_id = self.drawn_rect_ids[self.listbox_index]
            print(self.rect_id)
        except IndexError:
            print(f"[ERROR] rect_id index {self.listbox_index} out of range.")
            return


        try:
            
            coords = self.gui.image_canvas.coords(self.rect_id)
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

        print(f"[DEBUG] Modification mode ON → df_row: {df_index}, list_index: {self.listbox_index}, rect_id: {self.rect_id}")



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

        # print("DEBUG x list "), self.gui.all_annotations.at[self.selected_annotation_original_index, 'x']

        # print(f"x_list (len {len(x_list)}): {x_list}")
        # print(f"y_list (len {len(y_list)}): {y_list}")
        # print(f"w_list (len {len(w_list)}): {w_list}")
        # print(f"h_list (len {len(h_list)}): {h_list}")
        # print(f"annotation_idx: {annotation_idx}")

        # print(f"Listbox size: {self.gui.img_annotation_listbox.size()}")
        # print(f"Listbox current selection index: {annotation_idx}")


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

        # print("DEBUG x list "), self.gui.all_annotations.at[self.selected_annotation_original_index, 'x']

        # print(f"[DEBUG] Updated annotation index {annotation_idx} in DataFrame row {self.selected_annotation_original_index} → x:{new_x}, y:{new_y}, w:{new_w}, h:{new_h}")

        # Listbox-Eintrag aktualisieren (Klassen-Label hier einzeln aus der Liste holen)
        if self.listbox_index < len(class_list):
            class_label = class_list[self.listbox_index]
        else:
            class_label = "Unknown"

        selected_in_listbox = self.gui.img_annotation_listbox.curselection()
        if selected_in_listbox:
            updated_text = f"{str(class_label).ljust(12)} x:{str(new_x).ljust(5)} y:{str(new_y).ljust(5)} w:{str(new_w).ljust(5)} h:{str(new_h).ljust(5)}"

            self.gui.img_annotation_listbox.delete(self.listbox_index)
            self.gui.img_annotation_listbox.insert(self.listbox_index, updated_text)
            self.gui.img_annotation_listbox.selection_set(self.listbox_index)
            self.gui.img_annotation_listbox.activate(self.listbox_index)
        else:
            print("[Warning] Could not find selected item in listbox to update its text.")

    def update_image_listbox_with_annotation_colors(self):
        """
        Aktualisiert die Image-Listbox mit Hintergrundfarben abhängig vom Annotationsstatus.
        Grün = annotiert (class ≠ NN oder leer), Rot = nicht annotiert.
        """
        self.gui.image_listbox.delete(0, tk.END)

        image_folder = os.path.join(
            self.gui.selected_image_folder,
            self.gui.patient_id,
            self.gui.selected_exam
        )

        images = [
            i for i in os.listdir(image_folder)
            if i.lower().endswith(('.png', '.jpg', '.jpeg')) and "frame" not in i.lower()
        ]

        # Lade Annotationen aus CSV oder gespeicherter Quelle
        annotated_df = self.annotation_loader.load_annotations_from_annotable()

        # Sicherheitsprüfung: Falls img_ID oder class nicht vorhanden
        if 'img_ID' not in annotated_df.columns or 'class' not in annotated_df.columns:
            print("[ERROR] Annotation DataFrame fehlt notwendige Spalten ('img_ID' oder 'class').")
            return

        # Normiere leere oder NN-Klassen
        annotated_df['class'] = annotated_df['class'].apply(
            lambda x: "NN" if pd.isna(x) or x == [] else x
        )

        # Finde alle Bilder mit gültiger Annotation
        annotated_image_ids = set(
            annotated_df.loc[annotated_df['class'] != "NN", 'img_ID'].astype(str).str.strip()
        )

        # Listbox befüllen + einfärben
        for image in images:
            image_id = image.split(".")[0].strip()
            self.gui.image_listbox.insert(tk.END, image)
            index = self.gui.image_listbox.size() - 1

            if image_id in annotated_image_ids:
                self.gui.image_listbox.itemconfig(index, {'bg': '#d4fcd4'})  # grün
            else:
                self.gui.image_listbox.itemconfig(index, {'bg': '#fcd4d4'})  # rot

import tkinter as tk
import pandas as pd
import statistics as stat
import os

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
        self.modify_mode = False

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
            if isinstance(cell, list):
                cell.append(value)
            elif pd.isna(cell) or cell == "NN":
                return [value]
            else:
                return [cell, value]  # Fallback für Einzelwerte
            return cell
    
        for col, val in zip(['x', 'y', 'w', 'h', 'class'], [x, y, w, h, img_selected_class]):
            self.gui.all_annotations.at[idx, col] = append_or_init_list(self.gui.all_annotations.at[idx, col], val)
        print(self.gui.all_annotations["x"])

        self.update_image_listbox_with_annotation_colors()
        print(f"[DEBUG] Add Annotation: new rect with rect_id {self.rect_id} was added to internal list.")

        self.drawn_rect_ids.append(self.rect_id)
        print(f"[DEBUG] Added rect_id {self.rect_id} to drawn_rect_ids. List size now: {len(self.drawn_rect_ids)}")



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

        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(image_id).strip()
        if not match.any():
            print(f"[ERROR] No entry found in annotation_df for img_ID '{image_id}'")
            return

        idx = self.gui.all_annotations[match].index[0]

        try:
            # Werte aus den Spalten holen
            x_list = self.gui.all_annotations.at[idx, 'x']
            y_list = self.gui.all_annotations.at[idx, 'y']
            w_list = self.gui.all_annotations.at[idx, 'w']
            h_list = self.gui.all_annotations.at[idx, 'h']
            class_list = self.gui.all_annotations.at[idx, 'class']

            # Entfernen des Eintrags an Position `index`, nur wenn Listen vorhanden sind
            for col_name, data_list in zip(['x', 'y', 'w', 'h', 'class'], [x_list, y_list, w_list, h_list, class_list]):
                if isinstance(data_list, list) and len(data_list) > index:
                    data_list.pop(index)

            # Entferne Rechteck (falls vorhanden)
            if index < len(self.drawn_rect_ids):
                rect_id = self.drawn_rect_ids[index]
                self.gui.image_canvas.delete(rect_id)
                self.drawn_rect_ids.pop(index)

            # Entferne Resize-Handles
            for handle in self.resize_handles:
                self.gui.image_canvas.delete(handle)
            self.resize_handles.clear()

            # Entferne Eintrag aus Listbox
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
        x_list = row['x'] if isinstance(row['x'], list) else []
        y_list = row['y'] if isinstance(row['y'], list) else []
        w_list = row['w'] if isinstance(row['w'], list) else []
        h_list = row['h'] if isinstance(row['h'], list) else []
        class_list = row['class'] if isinstance(row['class'], list) else []

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
        if not self.modify_mode:
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
        if not self.modify_mode:
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
        if not self.modify_mode:
            if self.is_drawing:
                self.rect_end = (event.x, event.y)
                self.is_drawing = False
        else:
            self.dragging_handle = None
            self.dragging_rectangle = False
            self.last_mouse_pos = None

        self.modify_mode = False
        self.remove_resize_handles()

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
        self.modify_mode = False
        selection = self.gui.img_annotation_listbox.curselection()
        if selection:
            self.selected_annotation_original_index = selection[0]
            self.modify_mode = True


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
        """
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
        """
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
        x, y, w, h = self.calculate_rectangle()

        # Direkt in der DataFrame-Zeile ändern
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'x'] = x
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'y'] = y
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'w'] = w
        self.gui.all_annotations.at[self.selected_annotation_original_index, 'h'] = h

        print(f"[DEBUG] Updated DataFrame row {self.selected_annotation_original_index} → x:{x}, y:{y}, w:{w}, h:{h}")

        # Optional: rect_id prüfen (Debug)
        rect_id = self.gui.all_annotations.at[self.selected_annotation_original_index, 'rect_id']
        if rect_id != self.rect_id:
            print(f"[Warning] rect_id mismatch: DataFrame={rect_id}, Current={self.rect_id}")

        # Listbox-Eintrag aktualisieren
        selected_in_listbox = self.gui.img_annotation_listbox.curselection()
        if selected_in_listbox:
            listbox_index = selected_in_listbox[0]
            class_label = self.gui.all_annotations.at[self.selected_annotation_original_index, 'class']
            updated_text = f"{str(class_label).ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"

            self.gui.img_annotation_listbox.delete(listbox_index)
            self.gui.img_annotation_listbox.insert(listbox_index, updated_text)
            self.gui.img_annotation_listbox.selection_set(listbox_index)
            self.gui.img_annotation_listbox.activate(listbox_index)
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

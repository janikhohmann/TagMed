import os
import ast
import tkinter as tk
import pandas as pd

from config_handler import ConfigHandler
from video_annotation_handler import VideoAnnotationHandler

class  VideoTracking:
    def __init__(self, gui):
        self.gui = gui


        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.selected_anno_table_file = config.get("selected_anno_table_file")

        self.video_annotation_handler = VideoAnnotationHandler(gui)

    def tracking_starter(self):
        tracking_type = self.gui.tracking_type.get()
        
        if tracking_type == "Simple":
            self.simple_tracking_method()

    def simple_tracking_method(self):
        """Copies the annotation data to all subsequent frames in the central list."""

        selected_annotation = self.gui.video_annotation_listbox.curselection()
        if not selected_annotation:
            print("[INFO] Bitte zuerst eine Annotation auswählen.")
            return

        selected_annotation_index = selected_annotation[0]
        current_frames = self.gui.current_frames
        current_frame_index = self.gui.current_frame_index
        current_image_id = current_frames[current_frame_index].split(".")[0]
        selected_class = self.gui.video_selected_class.get()

        # Finde Zeile in DataFrame
        match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == current_image_id
        if not match.any():
            print(f"[ERROR] Kein Eintrag für img_ID = {current_image_id}")
            return
        df_index = self.gui.all_annotations[match].index[0]
        row = self.gui.all_annotations.loc[df_index]

        # Prüfe ob Polygon oder Bounding Box
        selected_text = self.gui.img_annotation_listbox.get(selected_annotation_index)
        is_polygon = "Polygon" in selected_text
        if is_polygon:
            print("[INFO] Polygon-Tracking noch nicht implementiert.")
            return
        
        else:

            # Hole aktuelle BBox-Daten
            x_list = self._safe_parse_list(row.get('x'))
            y_list = self._safe_parse_list(row.get('y'))
            w_list = self._safe_parse_list(row.get('w'))
            h_list = self._safe_parse_list(row.get('h'))
            class_list = self._safe_parse_list(row.get('class'))
            annotype_list = self._safe_parse_list(row.get('bb_annotype'))
            print(annotype_list)

            try:
                x = x_list[selected_annotation_index]
                y = y_list[selected_annotation_index]
                w = w_list[selected_annotation_index]
                h = h_list[selected_annotation_index]
                slected_class = class_list[selected_annotation_index]
                #annotype = annotype_list[selected_annotation_index]

            except IndexError:
                print(f"[ERROR] rect_id Index {selected_annotation_index} out of range.")
                return


            # Tracking-Daten in nachfolgende Frames eintragen
            for i in range(current_frame_index + 1, len(current_frames)):
                next_img_id = current_frames[i].split(".")[0]

                # Existiert schon eine Zeile für diesen Frame?
                match_next = self.gui.all_annotations['img_ID'].astype(str).str.strip() == next_img_id
                if match_next.any():
                    next_df_index = self.gui.all_annotations[match_next].index[0]

                    for col, val in zip(['x', 'y', 'w', 'h', 'class', 'bb_annotype'], [x, y, w, h, slected_class, 'tracking']):
                        self.gui.all_annotations.at[next_df_index, col] = self._append_or_init_list(self.gui.all_annotations.at[next_df_index, col], val)


            print("[INFO] Trackingdaten erfolgreich kopiert.")

    # def simple_tracking_method(self):
    #     """Copies the annotation data to all subsequent frames in the central list."""

    #     selected_annotation = self.gui.video_annotation_listbox.curselection()
    #     if not selected_annotation:
    #         print("hier könnte noch eine funktion rein die erinnert, dass eine annotation ausgewählt werden muss. videotracking, gui in gui")
    #         return
        
    #     selected_annotation_index = selected_annotation[0]
    #     print(selected_annotation_index)

    #     current_frames = self.gui.current_frames
    #     current_frame_index = self.gui.current_frame_index
    #     current_image_id = self.gui.current_frames[self.gui.current_frame_index].split(".")[0]
    #     img_annotation_type = self.gui.video_annotation_type.get()
    #     img_selected_class = self.gui.video_selected_class.get()

    #     match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == current_image_id
    #     if not match.any():
    #         print(f"[ERROR] No row found for img_ID = {current_image_id}")
    #         return
        
    #     df_index = self.gui.all_annotations[match].index[0]
    #     row = self.gui.all_annotations.loc[df_index]
        
    #     # Entscheide, ob es sich um Bounding Box oder Polygon handelt
    #     selected_text = self.gui.img_annotation_listbox.get(selected_annotation_index)
    #     is_polygon = "Polygon" in selected_text

    #     # ==== Polygon Tracking ====
    #     if is_polygon:
    #         pass

    #     else:
    #         x_list = self._safe_parse_list(row.get('x'))
    #         y_list = self._safe_parse_list(row.get('y'))
    #         w_list = self._safe_parse_list(row.get('w'))
    #         h_list = self._safe_parse_list(row.get('h'))
    #         class_list = self._safe_parse_list(row.get('class')) 


    #         for i in range(current_frame_index + 1, len(current_frames)):
    #              next_frame_id = current_frames[i].split(".")[0]
    #         try:
    #             self.rect_id = self.video_annotation_handler.drawn_rect_ids[self.selected_annotation_index]
    #             coords = self.gui.image_canvas.coords(self.rect_id)

    #             # bbox_annotype als Liste parsen
    #             bbox_annotype_list = self._safe_parse_list(row.get('bbox_annotype'))

    #             # Prüfen, ob Index gültig ist
    #             if self.selected_annotation_index < len(bbox_annotype_list):
    #                 bbox_annotype_list[self.selected_annotation_index] = "tracking"
    #             else:
    #                 print(f"[ERROR] listbox_index {self.selected_annotation_index} out of range for bbox_annotype.")

    #             # Geänderte Liste zurückschreiben
    #             row['bbox_annotype'] = bbox_annotype_list

    #             if len(coords) == 4:
    #                 x1, y1, x2, y2 = coords
    #                 self.rect_start = (x1, y1)
    #                 self.rect_end = (x2, y2)
    #                 self.create_resize_handles()  # funktioniert jetzt ohne Argumente
    #             else:
    #                 print(f"[ERROR] Unexpected coords for rect_id: {coords}")
    #                 return
    #         except IndexError:
    #             print(f"[ERROR] rect_id index {self.listbox_index} out of range.")
    #             return




        # if 'img_ID' not in self.gui.all_annotations.columns:
        #     print("[ERROR] 'img_ID' column not found in DataFrame.")
        #     return

        # # Search for row with Image-ID
        # match = self.gui.all_annotations['img_ID'].astype(str).str.strip() == str(image_id).strip()
        # if not match.any():
        #     print(f"[ERROR] No entry found in annotation_df for img_ID '{image_id}'")
        #     return
        # idx = self.gui.all_annotations[match].index[0] # return index of row in dataframe

        # # initialize cells 
        # def append_or_init_list(cell, value):
        #     """ HelperFunction: Appends a value to a list in a DataFrame cell or initializes it if empty."""
        #     if isinstance(cell, float) and pd.isna(cell):
        #         return [value]
        #     if isinstance(cell, str) and cell == "NN":
        #         return [value]
        #     if isinstance(cell, list):
        #         return cell + [value]
        #     try:
        #         parsed = ast.literal_eval(cell)
        #         if isinstance(parsed, list):
        #             return parsed + [value]
        #     except:
        #         pass
        #     return [value]
        
        # # === Bounding Box Annotation ===
        # if img_annotation_type == "Bounding Box":
        #     if self.gui.all_annotations.at[idx, "polygon"] != "NN": # safe exit when already an polygon annotation exists
        #         self.gui.wrong_annotation_warning_gui("Bounding Box")
        #         self.delete_all_bounding_boxes()
        #         return
        #     x, y, w, h = self.calculate_rectangle()
        #     annotation_text = f"{img_selected_class.ljust(12)} x:{str(x).ljust(5)} y:{str(y).ljust(5)} w:{str(w).ljust(5)} h:{str(h).ljust(5)}"
        #     self.gui.video_annotation_listbox.insert(tk.END, annotation_text)

        #     for col, val in zip(['x', 'y', 'w', 'h', 'class', 'bb_annotype'], [x, y, w, h, img_selected_class, 'tracking']):
        #         self.gui.all_annotations.at[idx, col] = append_or_init_list(self.gui.all_annotations.at[idx, col], val)

        #     self.drawn_rect_ids.append(self.rect_id)
        #     print(f"[DEBUG] Added Bounding Box with rect_id {self.rect_id}")





        # print(current_frame_index)

        # if not hasattr(self.video_annotation_handler, 'current_frames') or not current_frames:
        #     print("[WARN] No current frames found.")
        #     return
        
        # get annotation for respective frame
        # selected_class = self.gui.video_selected_class.get()
        # annotation_type = self.gui.video_annotation_type.get()
        # x, y, w, h = self.video_annotation_handler.calculate_rectangle()

        # # add new annotation for each frame when its a new annotation
        # if not self.gui.modify_mode:
























        # # add new annotation for each frame when its a new annotation
        # if not self.video_annotation_handler.modify_mode:
        #     for i in range(current_frame_index + 1, len(current_frames)):
        #         next_frame_id = current_frames[i].split(".")[0]

        #         self.gui.all_annotations.append({
        #             'img_ID': next_frame_id,
        #             'rect_id': None,
        #             'class': selected_class,
        #             'x': x, 'y': y, 'w': w, 'h': h,
        #             'annotation_type': annotation_type
        #         })
        # # modify all existing annotation when in modifying mode
        # else:
        #     for i in range(current_frame_index + 1, len(current_frames)):
        #         next_frame_id = current_frames[i].split(".")[0]

        #         updated = False
        #         for ann in self.gui.all_annotations:
        #             if ann['img_ID'] == next_frame_id and ann['class'] == selected_class:
        #                 ann.update({
        #                     'x': x, 'y': y, 'w': w, 'h': h,
        #                     'annotation_type': annotation_type
        #                 })
        #                 updated = True

        #         # If there is no fitting annotation -> add as a new one
        #         if not updated:
        #             self.gui.all_annotations.append({
        #                 'img_ID': next_frame_id,
        #                 'rect_id': None,
        #                 'class': selected_class,
        #                 'x': x, 'y': y, 'w': w, 'h': h,
        #                 'annotation_type': annotation_type
        #     })


        # print("[INFO] simple_tracking_method completed.")


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
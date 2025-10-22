"""
Export - Annotation data export module for TagMed application.

This module handles the export of annotated images and videos from the TagMed application. The script is called 
via the "Export" button in the main GUI and provides a seamless integration of TagMed with object detection frameworks.

Features:
- Export annotated images and videos in various formats.
- Compatibility with popular object detection frameworks.

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import pandas as pd
import json
import os
from datetime import datetime
import ast
import cv2

from config_handler import ConfigHandler


class ExportHandler:
    def __init__(self, gui):
        self.gui = gui
        self.config = ConfigHandler()

    def coco_export(self):
        """
        Export annotations to the specified export directory in CSV format.
        """
        self.export_dir = self.config.get("export_directory", "../exports")  # Directory to save exported files - should be configurable and absolute
        self.annotation_image_size = self.config.get("image_size", [600, 600])  # Image size for recalculating coordinates
        display_h, display_w = self.annotation_image_size if len(self.annotation_image_size) == 2 else (self.annotation_image_size[1], self.annotation_image_size[0])
        try:
            self.annotation_df = pd.read_csv(self.config.get("selected_anno_table_file"))  # DataFrame containing annotations
        except Exception as e:
            print(f"Error loading annotation data: {e}")
            return
        
        coco = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
               }
        
        # get metadata on gui ?????

        # collect categories
        all_classes = set(self.config.get("class_list", []))

        category_map = {name: i + 1 for i, name in enumerate(sorted(all_classes))}
        coco["categories"] = [{"id": i, "name": n} for n, i in category_map.items()]

        annotation_id = 1
        image_id_map = {}

        for idx, row in self.annotation_df.iterrows():
            img_id = row['img_ID']
            file_path = row['file_path']
            file_name = os.path.basename(file_path)


            height, width, channels = cv2.imread(file_path).shape # abfangen wenn es ein Video ist

            # ==== Image Entry ====
            if img_id not in image_id_map:
                image_id = len(image_id_map) + 1
                image_id_map[img_id] = image_id
                coco["images"].append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height
                })
            else:
                image_id = image_id_map[img_id]

            # === Bounding Box Annotations ===
            if pd.notna(row["x"]):
                try:
                    xs = ast.literal_eval(row["x"]) if isinstance(row["x"], str) else [row["x"]]
                    ys = ast.literal_eval(row["y"]) if isinstance(row["y"], str) else [row["y"]]
                    ws = ast.literal_eval(row["w"]) if isinstance(row["w"], str) else [row["w"]]
                    hs = ast.literal_eval(row["h"]) if isinstance(row["h"], str) else [row["h"]]
                    classes = ast.literal_eval(row["class"]) if isinstance(row["class"], str) else [row["class"]]
                except Exception:
                    continue

                # defensive: handle missing image read
                img = cv2.imread(file_path)
                if img is None:
                    print(f"[WARN] Could not read image {file_path}, skipping bboxes.")
                else:
                    orig_h, orig_w = img.shape[0], img.shape[1]
                    # fall-back: wenn display_w/display_h 0 oder None, skip
                    if display_w == 0 or display_h == 0:
                        print("[WARN] Display size is zero, skipping scaling.")
                        scale_x = scale_y = 1.0
                    else:
                        scale_x = orig_w / float(display_w)
                        scale_y = orig_h / float(display_h)

                    for x, y, w, h, c in zip(xs, ys, ws, hs, classes):
                        if pd.isna(c):
                            continue
                        # stored format: center x,y in display coordinates
                        xmin = float(x) - float(w)/2.0
                        ymin = float(y) - float(h)/2.0

                        # scale from display -> original image coordinates
                        xmin_s = max(0, xmin * scale_x)
                        ymin_s = max(0, ymin * scale_y)
                        w_s = max(0, float(w) * scale_x)
                        h_s = max(0, float(h) * scale_y)

                        # clip to image bounds
                        xmin_s = min(xmin_s, orig_w - 1)
                        ymin_s = min(ymin_s, orig_h - 1)
                        w_s = min(w_s, orig_w - xmin_s)
                        h_s = min(h_s, orig_h - ymin_s)

                        coco["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map.get(c, 0),
                            "bbox": [int(round(xmin_s)), int(round(ymin_s)), int(round(w_s)), int(round(h_s))],
                            "area": int(round(w_s * h_s)),
                            "iscrowd": 0,
                            "segmentation": [] # looking for mask if available
                        })
                        annotation_id += 1

            # === Polygon Annotations ===
            if pd.notna(row["polygon"]):
                try:
                    polygons = ast.literal_eval(row["polygon"]) if isinstance(row["polygon"], str) else row["polygon"]
                    class_poly = ast.literal_eval(row["class_polygon"]) if isinstance(row["class_polygon"], str) else row["class_polygon"]
                except Exception:
                    continue

                img = cv2.imread(file_path)
                if img is None:
                    print(f"[WARN] Could not read image {file_path}, skipping polygons.")
                else:
                    orig_h, orig_w = img.shape[0], img.shape[1]
                    if display_w == 0 or display_h == 0:
                        scale_x = scale_y = 1.0
                    else:
                        scale_x = orig_w / float(display_w)
                        scale_y = orig_h / float(display_h)

                    def polygon_area(points):
                        # Shoelace formula. points = list of (x,y)
                        area = 0.0
                        n = len(points)
                        for i in range(n):
                            x1,y1 = points[i]
                            x2,y2 = points[(i+1) % n]
                            area += x1*y2 - x2*y1
                        return abs(area) / 2.0

                    for poly, c in zip(polygons, class_poly):
                        if not poly or pd.isna(c):
                            continue

                        # scale each point
                        scaled_points = [[float(px) * scale_x, float(py) * scale_y] for px, py in poly]
                        # flat segmentation for COCO
                        flat_poly = [coord for p in scaled_points for coord in p]

                        # compute bbox from scaled polygon
                        xs_poly = [p[0] for p in scaled_points]
                        ys_poly = [p[1] for p in scaled_points]
                        xmin = max(0, min(xs_poly))
                        ymin = max(0, min(ys_poly))
                        w_poly = max(0, max(xs_poly) - xmin)
                        h_poly = max(0, max(ys_poly) - ymin)

                        area_poly = polygon_area(scaled_points)

                        coco["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map.get(c, 0),
                            "bbox": [int(round(xmin)), int(round(ymin)), int(round(w_poly)), int(round(h_poly))],
                            "area": int(round(area_poly)),
                            "iscrowd": 0,
                            "segmentation": [flat_poly]
                        })
                        annotation_id += 1
            # === JSON speichern ===
            output_json_path = os.path.join(self.export_dir, "coco_annotations.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(coco, f, indent=4)

    # def voc_export(self):
    #     """
    #     Export annotations to the specified export directory in Pascal VOC format.
    #     """
    #     self.export_dir = self.config.get("export_directory", "../exports")  # Directory to save exported files - should be configurable and absolute
    #     try:
    #         self.annotation_df = pd.read_csv(self.config.get("selected_anno_table_file"))  # DataFrame containing annotations
    #     except Exception as e:
    #         print(f"Error loading annotation data: {e}")
    #         return

    #     print("Starting VOC export process...")


# ======================================================================
# Helper funkctions


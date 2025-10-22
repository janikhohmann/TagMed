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

            # ==== Bounding Box Annotations ====
            if pd.notna(row["x"]):
                try:
                    xs = ast.literal_eval(row["x"]) if isinstance(row["x"], str) else [row["x"]]
                    ys = ast.literal_eval(row["y"]) if isinstance(row["y"], str) else [row["y"]]
                    ws = ast.literal_eval(row["w"]) if isinstance(row["w"], str) else [row["w"]]
                    hs = ast.literal_eval(row["h"]) if isinstance(row["h"], str) else [row["h"]]
                    classes = ast.literal_eval(row["class"]) if isinstance(row["class"], str) else [row["class"]]
                except Exception:
                    continue

                for x, y, w, h, c in zip(xs, ys, ws, hs, classes):
                    if pd.isna(c):
                        continue
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map.get(c, 0),
                        "bbox": [x, y, w, h], # wrong format and wrong image size
                        "area": w * h,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    annotation_id += 1

            # ==== Polygon Annotations ====
            if pd.notna(row["polygon"]):
                try:
                    polygons = ast.literal_eval(row["polygon"]) if isinstance(row["polygon"], str) else row["polygon"]
                    class_poly = ast.literal_eval(row["class_polygon"]) if isinstance(row["class_polygon"], str) else row["class_polygon"]
                except Exception:
                    continue

                for poly, c in zip(polygons, class_poly):
                    if not poly or pd.isna(c):
                        continue

                    # Flatten polygon points (list of [x,y] -> [x1,y1,x2,y2,...])
                    flat_poly = [coord for point in poly for coord in point]

                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map.get(c, 0),
                        "bbox": [min(p[0] for p in poly),
                                min(p[1] for p in poly),
                                max(p[0] for p in poly) - min(p[0] for p in poly),
                                max(p[1] for p in poly) - min(p[1] for p in poly)],
                        "area": 0,  # Optional: calculate area if needed
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


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
        

    def coco_export(self, export_info_dic = None):
        """
        Export annotations to the specified export directory in COCO format.
        """
        self.config = ConfigHandler()
        self.export_dir = self.config.get("export_directory", "../exports")
        self.annotation_image_size = self.config.get("image_size", [600, 600])

        # interpret display size (assume [width, height] in config; fallback to sane defaults)
        try:
            if len(self.annotation_image_size) == 2:
                display_w, display_h = int(self.annotation_image_size[0]), int(self.annotation_image_size[1])
            else:
                display_w, display_h = 600, 600
        except Exception:
            display_w, display_h = 600, 600

        # try reading CSV (first try semicolon separated, then default)
        csv_path = self.config.get("selected_anno_table_file")
        try:
            self.annotation_df = pd.read_csv(csv_path, sep=";")
        except Exception:
            try:
                self.annotation_df = pd.read_csv(csv_path)
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

        if export_info_dic is not None: # add custom info if provided
            for key in export_info_dic:
                coco["info"][key] = export_info_dic[key]

        # categories
        all_classes = set(self.config.get("class_list", []))
        category_map = {name: i + 1 for i, name in enumerate(sorted(all_classes))}
        for name, cid in category_map.items():
            coco["categories"].append({"id": cid, "name": name})

        # helpers
        def parse_list_cell(cell):
            if pd.isna(cell):
                return []
            if isinstance(cell, list):
                return cell
            if isinstance(cell, str):
                try:
                    parsed = ast.literal_eval(cell)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
                return [cell]
            return [cell]

        def polygon_area(points):
            # Shoelace formula for polygon area
            area = 0.0
            n = len(points)
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return abs(area) / 2.0

        annotation_id = 1
        image_id_map = {}
        frame_size_cache = {}

        for idx, row in self.annotation_df.iterrows():
            img_id = str(row.get('img_ID'))
            file_path = row.get('file_path')
            if not file_path or not isinstance(file_path, str):
                print(f"[WARN] Missing file_path for row {idx}, skipping.")
                continue
            file_name = os.path.basename(file_path)
            #print(file_name)

            # determine whether this is a video frame by convention in img_id
            is_frame = "frame" in img_id

            # determine original image size
            orig_w = orig_h = None

            # reuse cache when possible
            if file_path in frame_size_cache:
                orig_w, orig_h = frame_size_cache[file_path]
            else:
                # if it's a video path, try to read via VideoCapture
                try:
                    if is_frame: # and os.path.exists(file_path) and file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        cap = cv2.VideoCapture(file_path)
                        if cap.isOpened():
                            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            cap.release()
                            frame_size_cache[file_path] = (orig_w, orig_h) # only cache frame sizes for videos
                        else:
                            # fallback: try reading as image
                            img = cv2.imread(file_path)
                            if img is None:
                                print(f"[WARN] Could not open video or image {file_path}, skipping row {idx}.")
                                continue
                            orig_h, orig_w = img.shape[:2]
                    else:
                        img = cv2.imread(file_path)
                        if img is None:
                            print(f"[WARN] Could not read image {file_path}, skipping row {idx}.")
                            continue
                        orig_h, orig_w = img.shape[:2]
                except Exception as e:
                    print(f"[WARN] Error reading file {file_path}: {e}")
                    continue

                

            # compute scale from display coords -> original image coords
            scale_x = orig_w / float(display_w) if display_w else 1.0
            scale_y = orig_h / float(display_h) if display_h else 1.0

            # ensure image entry exists
            if img_id not in image_id_map:
                image_id = len(image_id_map) + 1
                image_id_map[img_id] = image_id
                coco["images"].append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": int(orig_w),
                    "height": int(orig_h)
                })
            else:
                image_id = image_id_map[img_id]

            # === Bounding boxes ===
            if pd.notna(row.get('x')):
                xs = parse_list_cell(row.get('x'))
                ys = parse_list_cell(row.get('y'))
                ws = parse_list_cell(row.get('w'))
                hs = parse_list_cell(row.get('h'))
                classes = parse_list_cell(row.get('class'))

                # ==== MASKS SHOULD BE IMPLEMENTED HERE -> segmentation field for bboxes ====


                for x, y, w, h, c in zip(xs, ys, ws, hs, classes):
                    if pd.isna(c):
                        continue
                    try:
                        cx = float(x)
                        cy = float(y)
                        bw = float(w)
                        bh = float(h)
                    except Exception:
                        continue

                    # convert center (cx,cy) to top-left
                    xmin = cx - bw / 2.0
                    ymin = cy - bh / 2.0

                    # scale to original image
                    xmin_s = max(0.0, xmin * scale_x)
                    ymin_s = max(0.0, ymin * scale_y)
                    w_s = max(0.0, bw * scale_x)
                    h_s = max(0.0, bh * scale_y)

                    # clip
                    xmin_s = min(xmin_s, orig_w - 1)
                    ymin_s = min(ymin_s, orig_h - 1)
                    w_s = min(w_s, orig_w - xmin_s)
                    h_s = min(h_s, orig_h - ymin_s)

                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map.get(str(c), 0),
                        "bbox": [int(round(xmin_s)), int(round(ymin_s)), int(round(w_s)), int(round(h_s))],
                        "area": int(round(w_s * h_s)),
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    annotation_id += 1

            # === Polygons ===
            if pd.notna(row.get('polygon')):
                polygons = parse_list_cell(row.get('polygon'))
                class_polys = parse_list_cell(row.get('class_polygon'))

                for poly, c in zip(polygons, class_polys):
                    if not poly or pd.isna(c):
                        continue
                    # poly is expected as list of [x,y]
                    try:
                        scaled_points = [[float(px) * scale_x, float(py) * scale_y] for px, py in poly]
                    except Exception:
                        # maybe the polygon is stored as a flat list
                        try:
                            flat = [float(v) for v in poly]
                            scaled_points = [[flat[i] * scale_x, flat[i+1] * scale_y] for i in range(0, len(flat), 2)]
                        except Exception:
                            continue

                    flat_poly = [float(coord) for p in scaled_points for coord in p]

                    xs_poly = [p[0] for p in scaled_points]
                    ys_poly = [p[1] for p in scaled_points]
                    xmin = max(0.0, min(xs_poly))
                    ymin = max(0.0, min(ys_poly))
                    w_poly = max(0.0, max(xs_poly) - xmin)
                    h_poly = max(0.0, max(ys_poly) - ymin)

                    area_poly = polygon_area(scaled_points)

                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map.get(str(c), 0),
                        "bbox": [int(round(xmin)), int(round(ymin)), int(round(w_poly)), int(round(h_poly))],
                        "area": int(round(area_poly)),
                        "iscrowd": 0,
                        "segmentation": [ [int(round(v)) for v in flat_poly] ]
                    })
                    annotation_id += 1

        # save JSON
        if export_info_dic is not None: # name dataset custom
            export_file_name = f"{export_info_dic.get('name of dataset')}_{export_info_dic.get('version')}.json"
            output_json_path = os.path.join(self.export_dir, export_file_name)
        else:
            output_json_path = os.path.join(self.export_dir, "coco_annotations.json")
            
        try:
            os.makedirs(self.export_dir, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(coco, f, indent=4)
            print(f"COCO annotations exported to {output_json_path}")
        except Exception as e:
            print(f"[ERROR] Could not write COCO JSON: {e}")





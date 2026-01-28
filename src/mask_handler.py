"""
TagMed Mask Handler - Mask visualization and management for medical annotation

This module provides comprehensive mask handling capabilities for the TagMed medical
annotation application. It manages mask storage, visualization, and manipulation
for both individual images and video frame sequences.

Key Features:
- Mask persistence with compressed NPZ format
- Dual-canvas support (image and video frame canvases)
- Transparent overlay rendering with customizable colors
- Batch mask operations for video sequences
- Memory-efficient mask loading and garbage collection management

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import numpy as np
import os
from PIL import Image, ImageTk


class MaskHandler:
    def __init__(self, gui):
        self.gui = gui
        
        self.mask_dir = "../masks" # path to the directory where masks will be saved - should be configurable and absolute



    def load_mask(self, image_id, mask_class, mask_idx):
        """
        Load a mask from a file with the format:
        <img_id>_<class>_<index>.npz.
        """
        mask_path = f"{self.mask_dir}/{image_id}_{mask_class}_{mask_idx}.np"
        try:
            mask = np.load(mask_path)
            return mask
        except FileNotFoundError:
            print(f"Mask file not found: {mask_path}")
            return None
        

    def save_mask(self, mask, image_id, mask_class, mask_idx):
        """
        Save the mask to a file with following format:
        <img_id>_<class>_<index>.npz.
        """
        os.makedirs(self.mask_dir, exist_ok=True)
        
        # Ensure mask is 2D before saving
        if mask.ndim > 2:
            mask = mask.squeeze()  # Remove dimensions of size 1
        elif mask.ndim == 1:
            print(f"[ERROR] Cannot save 1D mask for {image_id}_{mask_class}_{mask_idx}")
            return None
        
        mask_path = f"{self.mask_dir}/{image_id}_{mask_class}_{mask_idx}.npz"
        np.savez_compressed(mask_path, mask=mask)

        return mask_path
    
    def delete_mask(self, mask_path):
        """
        Delete a mask file     
        """

        try:
            os.remove(mask_path)
            print(f"Mask file deleted: {mask_path}")
        except FileNotFoundError:
            print(f"Mask file not found for deletion: {mask_path}")
        except Exception as e:
            print(f"Error deleting mask file: {e}")
    
    def delete_all_masks_for_one_video(self, frame_id):
        """
        Delete all mask files for a specific video.
        """

        print(f"Deleting masks for video frame: {frame_id}")
        masks = self.searching_for_matching_masks(frame_id)
        print(f"Found masks: {masks}")
        if not masks:
            print(f"No masks found for video {frame_id}.")
            return
    
        for mask in masks:
            mask_path = os.path.join(self.mask_dir, mask)
            self.delete_mask(mask_path)
        
        print(f"All masks for video {frame_id} have been deleted.")

    def searching_for_matching_masks(self, frame_id):
        """
        Search for all mask files that match the given image_id.
        Returns a list of matching file paths.
        """

        relevant_files = []
        for root, dirs, files in os.walk(self.mask_dir):
            for file in files:
                #print(file)
                if frame_id in file and file.endswith('.npz'):
                    relevant_files.append(file)
        return relevant_files


    def load_masks_for_frame(self):
        """Load masks for the selected frame into canvas with zoom and pan transformation."""
        self.clear_all_masks()

        current_frame_id = self.gui.current_frame_id
        #current_frame_index = self.gui.current_frame_index

        masks = self.searching_for_matching_masks(current_frame_id) 
        print(masks)
        for mask in masks:
            mask_path = os.path.join(self.mask_dir, mask)
            try:
                mask_data = np.load(mask_path)['mask']  # Load the mask data
                
                # Ensure mask is 2D
                if mask_data.ndim > 2:
                    mask_data = mask_data.squeeze()  # Remove dimensions of size 1
                elif mask_data.ndim == 1:
                    print(f"[ERROR] Invalid mask dimensions for {mask}: {mask_data.shape}")
                    continue
                
                height, width = mask_data.shape
                #red_color = (255, 0, 0, 70)  # bright, transparent red (alpha=60/255)
                red_color = (255, 165, 0, 150)  # Orange
                rgba_array = np.zeros((height, width, 4), dtype=np.uint8)

                rgba_array[mask_data > 0] = red_color  # only fill mask

                # Convert to PIL Image
                pil_image = Image.fromarray(rgba_array, mode="RGBA")
                
                # Apply zoom transformation - resize the mask image
                if hasattr(self.gui, 'video_zoom_level'):
                    zoom_level = self.gui.video_zoom_level
                    new_width = int(width * zoom_level)
                    new_height = int(height * zoom_level)
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.NEAREST)
                
                tk_image = ImageTk.PhotoImage(pil_image)

                # Apply pan transformation - position the mask with pan offsets
                pan_x = getattr(self.gui, 'video_pan_offset_x', 0)
                pan_y = getattr(self.gui, 'video_pan_offset_y', 0)
                
                # Show Image on Canvas with zoom and pan
                mask_id = self.gui.frame_canvas.create_image(pan_x, pan_y, anchor="nw", image=tk_image, tags=("mask",))

                
                # Prevent garbage collection
                if not hasattr(self.gui, 'mask_image_refs'):
                    self.gui.mask_image_refs = []
                self.gui.mask_image_refs.append(tk_image)

            except Exception as e:
                print(f"[ERROR] Failed to draw mask for {mask} (Error: {e})")
                continue


    def load_masks_for_image(self):
        """Load masks for the selected image into canvas with zoom and pan transformation."""
        self.clear_all_masks()

        current_frame_id = self.gui.selected_image_index.split(".")[0].strip()
        #current_frame_index = self.gui.current_frame_index

        masks = self.searching_for_matching_masks(current_frame_id) 
        print(masks)
        for mask in masks:
            mask_path = os.path.join(self.mask_dir, mask)
            try:
                mask_data = np.load(mask_path)['mask']  # Load the mask data
                
                # Ensure mask is 2D
                if mask_data.ndim > 2:
                    mask_data = mask_data.squeeze()  # Remove dimensions of size 1
                elif mask_data.ndim == 1:
                    print(f"[ERROR] Invalid mask dimensions for {mask}: {mask_data.shape}")
                    continue
                
                height, width = mask_data.shape
                #red_color = (255, 0, 0, 70)  # bright, transparent red (alpha=60/255)
                red_color = (255, 165, 0, 150)  # Orange
                rgba_array = np.zeros((height, width, 4), dtype=np.uint8)

                rgba_array[mask_data > 0] = red_color  # only fill mask

                # Convert to PIL Image
                pil_image = Image.fromarray(rgba_array, mode="RGBA")
                
                # Apply zoom transformation - resize the mask image
                if hasattr(self.gui, 'zoom_level'):
                    zoom_level = self.gui.zoom_level
                    new_width = int(width * zoom_level)
                    new_height = int(height * zoom_level)
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.NEAREST)
                
                tk_image = ImageTk.PhotoImage(pil_image)

                # Apply pan transformation - position the mask with pan offsets
                pan_x = getattr(self.gui, 'pan_offset_x', 0)
                pan_y = getattr(self.gui, 'pan_offset_y', 0)
                
                # Show Image on Canvas with zoom and pan
                mask_id = self.gui.image_canvas.create_image(pan_x, pan_y, anchor="nw", image=tk_image, tags=("mask",))

                
                # Prevent garbage collection
                if not hasattr(self.gui, 'mask_image_refs'):
                    self.gui.mask_image_refs = []
                self.gui.mask_image_refs.append(tk_image)

            except Exception as e:
                print(f"[ERROR] Failed to draw mask for {mask} (Error: {e})")
                continue




    def clear_all_masks(self):
        """Clear all masks."""
        try:
            self.gui.frame_canvas.delete("mask")
        except AttributeError:
            pass  # If frame_canvas does not exist, ignore
        try:
            self.gui.image_canvas.delete("mask")
        except AttributeError:
            pass  # If image_canvas does not exist, ignore

        if not hasattr(self.gui, 'mask_image_refs'):
            self.gui.mask_image_refs = []
        
        self.gui.mask_image_refs.clear()






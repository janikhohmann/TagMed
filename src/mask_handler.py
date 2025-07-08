import numpy as np
import os
from PIL import Image, ImageTk

#from video_annotation_handler import VideoAnnotationHandler




class MaskHandler:
    def __init__(self, gui):
        self.gui = gui
        
        self.mask_dir = "../masks" # path to the directory where masks will be saved - should be configurable and absolute



    def load_mask(self, image_id, mask_class, mask_idx):
        """
        Load a mask from a file with the format:
        <img_id>_<class>_<index>.npz.
        """
        mask_path = f"{self.mask_dir}/{image_id}_{mask_class}_{mask_idx}.npy"
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
        mask_path = f"{self.mask_dir}/{image_id}_{mask_class}_{mask_idx}.npz"
        #np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
        np.savez_compressed(mask_path, mask=mask)
        #np.save(mask_path, mask)

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
        """Load masks for the selected frame into canvas."""
        self.clear_all_masks()

        current_frame_id = self.gui.current_frame_id
        #current_frame_index = self.gui.current_frame_index

        masks = self.searching_for_matching_masks(current_frame_id) 
        print(masks)
        for mask in masks:
            mask_path = os.path.join(self.mask_dir, mask)
            try:
                mask_data = np.load(mask_path)['mask']  # Load the mask data
                height, width = mask_data.shape
                red_color = (255, 0, 0, 70)  # bright, transparent red (alpha=60/255)
                rgba_array = np.zeros((height, width, 4), dtype=np.uint8)

                rgba_array[mask_data > 0] = red_color  # only fill mask

                # Convert in PIL-Image then in PhotoImage
                pil_image = Image.fromarray(rgba_array, mode="RGBA")
                tk_image = ImageTk.PhotoImage(pil_image)

                # Show Image on Canvas
                mask_id = self.gui.frame_canvas.create_image(0, 0, anchor="nw", image=tk_image, tags=("mask",))

                
                # Prevent garbage collection
                if not hasattr(self.gui, 'mask_image_refs'):
                    self.gui.mask_image_refs = []
                self.gui.mask_image_refs.append(tk_image)

            except Exception as e:
                print(f"[ERROR] Failed to draw mask for {mask} (Error: {e})")
                continue


    def load_masks_for_image(self):
        """Load masks for the selected frame into canvas."""
        self.clear_all_masks()

        current_frame_id = self.gui.selected_image_index.split(".")[0].strip()
        #current_frame_index = self.gui.current_frame_index

        masks = self.searching_for_matching_masks(current_frame_id) 
        print(masks)
        for mask in masks:
            mask_path = os.path.join(self.mask_dir, mask)
            try:
                mask_data = np.load(mask_path)['mask']  # Load the mask data
                height, width = mask_data.shape
                red_color = (255, 0, 0, 70)  # bright, transparent red (alpha=60/255)
                rgba_array = np.zeros((height, width, 4), dtype=np.uint8)

                rgba_array[mask_data > 0] = red_color  # only fill mask

                # Convert in PIL-Image then in PhotoImage
                pil_image = Image.fromarray(rgba_array, mode="RGBA")
                tk_image = ImageTk.PhotoImage(pil_image)

                # Show Image on Canvas
                mask_id = self.gui.image_canvas.create_image(0, 0, anchor="nw", image=tk_image, tags=("mask",))

                
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






import numpy as np
import os


class MaskHandler:
    def __init__(self):
        self.mask_dir = "../masks" # path to the directory where masks will be saved - should be configurable and absolute


    def load_mask(self, image_id, mask_class, mask_idx):
        """
        Load a mask from a file with the format:
        <img_id>_<class>_<index>.npy.
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
        <img_id>_<class>_<index>.npy.
        """
        os.makedirs(self.mask_dir, exist_ok=True)
        mask_path = f"{self.mask_dir}/{image_id}_{mask_class}_{mask_idx}.npz"
        #np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
        np.savez_compressed(mask_path, mask=mask)
        #np.save(mask_path, mask)

        return mask_path
    
    def delete_mask(self, image_id, mask_class, mask_idx):
        """
        Delete a mask file     
        """
        mask_path = f"{self.mask_dir}/{image_id}_{mask_class}_{mask_idx}.npz"
        try:
            os.remove(mask_path)
            print(f"Mask file deleted: {mask_path}")
        except FileNotFoundError:
            print(f"Mask file not found for deletion: {mask_path}")
        except Exception as e:
            print(f"Error deleting mask file: {e}")


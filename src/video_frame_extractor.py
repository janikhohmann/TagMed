"""
Video Frame Extractor for TagMed - Intelligent Video Frame Handling

This module provides intelligent "on-the-fly" video frame extraction for the
TagMed annotation tool. It automatically detects whether video frames already
exist or need to be extracted, managing temporary storage during annotation sessions.

Features:
- Automatic detection: existing frames or extract from video
- Temporary storage while patient session is active
- Compatible with existing TagMed structure
- Automatic cleanup on patient switch
- Support for multiple video formats
- Configurable frame extraction intervals

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import os
import cv2
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm


class VideoFrameExtractor:
    """
    Intelligent video frame extractor for TagMed annotation workflow.
    
    This class provides smart frame handling for video annotation by automatically
    detecting whether frames already exist for a video or need to be extracted.
    It manages temporary storage during annotation sessions and provides cleanup
    functionality when switching between patients.
    
    Features:
    - Automatic detection: existing frames vs. video extraction
    - Temporary storage during active patient sessions
    - Compatible with existing TagMed folder structure
    - Automatic cleanup on patient switch
    - Support for multiple video formats
    - Configurable extraction parameters
    
    Attributes:
        temp_frame_dirs (dict): Mapping of video paths to temporary directories
        supported_video_formats (set): Set of supported video file extensions
    """
    
    def __init__(self):
        """
        Initialize VideoFrameExtractor with default settings.
        
        Sets up temporary directory tracking and defines supported
        video formats for the extraction process.
        """
        self.temp_frame_dirs = {}  # Video path -> temp directory mapping
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
    def get_video_frames_intelligent(self, patient_id, selected_exam, selected_video, image_folder):
        """
        Intelligent frame provision for video annotation.
        
        This method automatically determines whether to use existing frames
        or extract new ones from a video file. It first checks for existing
        frames, and if none are found, extracts frames from the video.
        
        Args:
            patient_id (str): Patient identifier
            selected_exam (str): Selected examination folder
            selected_video (str): Selected video file (e.g. "video1.mp4")
            image_folder (str): Base folder containing patient data
            
        Returns:
            list: Sorted list of available frame files
        """
        video_base_name = selected_video.split(".")[0]  # "video1.mp4" -> "video1"
        
        # 1. First check if frames already exist
        existing_frames = self._find_existing_frames(image_folder, video_base_name)
        
        if existing_frames:
            print(f"[INFO] Found existing frames for {selected_video}: {len(existing_frames)}")
            return existing_frames
        
        # 2. If no frames exist, check if video file exists
        video_path = os.path.join(image_folder, selected_video)
        
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            return []
            
        if not self._is_video_file(video_path):
            print(f"[ERROR] Unsupported video format: {selected_video}")
            return []
            
        print(f"[INFO] No frames found, extracting from video: {selected_video}")
        
        # 3. Extract frames from video
        extracted_frames = self._extract_frames_from_video(
            video_path=video_path,
            video_base_name=video_base_name,
            frame_interval=1,  # Every 1st frame (configurable)
            max_frames=2000     # Maximum 2000 frames (configurable)
        )
        
        return extracted_frames
    
    def _find_existing_frames(self, image_folder, video_base_name):
        """
        Search for existing frames corresponding to a video file.
        
        Scans the image folder for frame files that match the video's base name
        and contain frame indicators. Supports common image formats.
        
        Args:
            image_folder (str): Directory to search for existing frames
            video_base_name (str): Base name of the video (without extension)
            
        Returns:
            list: Sorted list of existing frame files, empty if none found
        """
        try:
            all_files = os.listdir(image_folder)
            
            # Search for files that belong to the video and contain "frame"
            video_frames = [
                file for file in all_files 
                if video_base_name in file and "frame" in file.lower()
                and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
            ]
            
            return sorted(video_frames)
            
        except Exception as e:
            print(f"[ERROR] Error searching for frames: {e}")
            return []
    
    def _is_video_file(self, file_path):
        """
        Check if a file is a supported video format.
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            bool: True if the file is a supported video format
        """
        return Path(file_path).suffix.lower() in self.supported_video_formats
    
    def _extract_frames_from_video(self, video_path, video_base_name, frame_interval=1, max_frames=2000):
        """
        Extract frames from video into temporary directory.
        
        Creates a temporary directory and extracts frames from the video
        at specified intervals. Handles progress display and error recovery.
        
        Args:
            video_path (str): Path to the video file
            video_base_name (str): Base name for frame files
            frame_interval (int): Extrahiere jeden N-ten Frame
            max_frames (int): Maximale Anzahl Frames
            
        Returns:
            list: list with extracted frame filenames
        """
        # check if already extracted frames exist for this video
        if video_path in self.temp_frame_dirs:
            existing_frames = self._get_frames_from_temp_dir(video_path)
            if existing_frames:
                print(f"[INFO] Frames already extracted for: {Path(video_path).name}")
                return existing_frames

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"tagmed_frames_{video_base_name}_")
        print(f"[INFO] Extracting frames to: {temp_dir}")

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video: {video_path}")
                return []

            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"[INFO] Video Info: {total_frames} Frames, {fps:.2f} FPS, {duration:.2f}s")
            
            extracted_count = 0
            frame_number = 0
            extracted_frame_names = []
            
            # Progress bar
            expected_frames = min(total_frames // frame_interval, max_frames)
            pbar = tqdm(total=expected_frames, desc=f"extracting {Path(video_path).name}", unit="frames")

            while frame_number < total_frames and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # extract every N-th frame
                if frame_number % frame_interval == 0:
                    # Frame name compatible with SAM2 structure (start with frame 1)
                    frame_display_number = (frame_number // frame_interval) + 1
                    frame_filename = f"{video_base_name}_frame_{frame_display_number:06d}.jpg"
                    frame_path = Path(temp_dir) / frame_filename

                    # Save frame
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frame_names.append(frame_filename)
                    extracted_count += 1
                    pbar.update(1)
                
                frame_number += 1
            
            pbar.close()
            cap.release()

            print(f"[INFO] {extracted_count} frames extracted from {Path(video_path).name}")

            # Save temp directory
            self.temp_frame_dirs[video_path] = temp_dir
            
            return sorted(extracted_frame_names)
            
        except Exception as e:
            print(f"[ERROR] Frame extraction failed: {e}")
            # Cleanup on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return []
    
    def _get_frames_from_temp_dir(self, video_path):
        """
        Gets frame filenames from already extracted temp directory.
        
        Args:
            video_path (str): Path to the original video file

        Returns:
            list: Sorted list of frame filenames
        """
        if video_path not in self.temp_frame_dirs:
            return []
        
        temp_dir = Path(self.temp_frame_dirs[video_path])
        if not temp_dir.exists():
            return []

        # Find all frame files
        frame_files = [
            f.name for f in temp_dir.glob('*_frame_*.jpg')
        ]
        
        return sorted(frame_files)
    
    def get_frame_path(self, patient_id, selected_exam, selected_video, frame_filename, image_folder):
        """
        Gives back the full path to a specific frame.
        
        Args:
            patient_id (str): Patient ID
            selected_exam (str): Exam ID
            selected_video (str): Video Name
            frame_filename (str): Frame filename
            image_folder (str): Base image folder

        Returns:
            str: Full path to the frame
        """
        video_path = os.path.join(image_folder, selected_video)

        # Check if frame comes from temporary directory
        if video_path in self.temp_frame_dirs:
            temp_dir = self.temp_frame_dirs[video_path]
            frame_path = os.path.join(temp_dir, frame_filename)
            if os.path.exists(frame_path):
                return frame_path
        
        # Otherwise normal path in image_folder
        return os.path.join(image_folder, frame_filename)
    
    def cleanup_temp_frames(self, video_path=None):
        """
        Deletes temporary frame directories.

        Args:
            video_path (str, optional): Specific video (None = all)
        """
        if video_path:
            # Cleanup for specific video
            if video_path in self.temp_frame_dirs:
                temp_dir = self.temp_frame_dirs[video_path]
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"[INFO] Temporary frames deleted for: {Path(video_path).name}")
                    except Exception as e:
                        print(f"[WARNING] Could not delete temp directory: {e}")
                del self.temp_frame_dirs[video_path]
        else:
            # Cleanup for all videos
            for video_path, temp_dir in list(self.temp_frame_dirs.items()):
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"[INFO] Temporary frames deleted: {temp_dir}")
                    except Exception as e:
                        print(f"[WARNING] Could not delete temp directory: {e}")
            self.temp_frame_dirs.clear()
    
    def cleanup_on_patient_switch(self):
        """
        Cleanup when switching patients.
        Deletes all temporary frame directories.
        """
        print("[INFO] Cleanup temporary frames when switching patients...")
        self.cleanup_temp_frames()  # Deletes all temp directories

    def get_video_info(self, video_path):
        """
        Gets video information without frame extraction.

        Args:
            video_path (str): Path to the video file

        Returns:
            dict: Video properties
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0
            }
            
            if info['fps'] > 0:
                info['duration'] = info['total_frames'] / info['fps']
            
            cap.release()
            return info
            
        except Exception as e:
            print(f"[ERROR] Error loading video info: {e}")
            return None

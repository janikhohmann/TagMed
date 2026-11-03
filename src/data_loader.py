"""
TagMed Data Loader - Medical image and video data management for annotation workflows

This module provides comprehensive data loading capabilities for the TagMed medical
annotation application. It handles the discovery and loading of medical images and
videos from organized patient directories, supporting various file formats and
flexible directory structures.

Key Features:
- Patient/Exam/File hierarchy navigation for medical data organization
- Multiple image format support (PNG, JPG, JPEG) with medical imaging standards
- Video file detection and loading for temporal annotation workflows
- Frame filtering to exclude pre-extracted frames from listings
- Configuration-based path management for flexible deployment
- Robust error handling for missing directories and file access issues

The loader is designed to work with standard medical imaging directory structures
where patients contain multiple exams, and each exam contains images and/or videos
suitable for medical annotation and analysis.

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import os
import tkinter as tk

from config_handler import ConfigHandler


class DataLoader:
    """
    Handles loading and discovery of medical images and videos from patient directories.
    
    This class provides comprehensive data discovery methods to scan patient exam
    folders and retrieve lists of available images and videos. It integrates with
    the configuration system to locate the base data directory and supports filtering
    to separate original files from processed frames.
    
    The loader is optimized for medical imaging workflows where data is organized
    hierarchically by patient ID and exam ID, with support for both individual
    images and video sequences requiring frame-by-frame annotation.
    
    Attributes:
        config (ConfigHandler): Configuration management instance for data paths
        selected_image_folder (str): Base directory containing patient data structure
    """
    
    def __init__(self):
        """
        Initialize the data loader with configuration settings.
        
        Loads the base image folder path from configuration and prepares
        the loader for scanning patient directories. The configuration
        system allows for flexible specification of data source locations
        across different deployment environments.
        """
        # Initialize configuration management
        self.config = ConfigHandler()
        
        # Load base directory path for patient data from configuration
        self.selected_image_folder = self.config.get("selected_image_folder", "")


    def load_images_in_dir(self, patient_id, selected_exam):
        """
        Load all image files from a specified patient exam directory.
        
        Scans the exam folder for supported image formats while filtering out
        frame files that may have been extracted from videos. This ensures
        only original image files are returned for annotation workflows,
        preventing duplicate listings of video frames alongside source videos.
        
        Args:
            patient_id (str): Unique identifier for the patient (folder name)
            selected_exam (str): Exam identifier/folder name within patient directory
            
        Returns:
            list: List of image filenames (excluding frame extracts) or empty list
                  if folder not found or no valid images exist
                  
        Supported Formats:
            - PNG (.png): Preferred for medical imaging due to lossless compression
            - JPEG (.jpg, .jpeg): Supported for compatibility with various systems
            
        Filtering Logic:
            Files containing "frame" in the filename are excluded to avoid
            listing video frame extracts alongside original images, preventing
            confusion in annotation workflows where both videos and extracted
            frames might coexist in the same directory.
        """
        # Construct the full path to the selected exam folder
        image_folder = os.path.join(self.selected_image_folder, patient_id, selected_exam)
        
        # Verify folder exists before attempting to scan
        if not os.path.exists(image_folder):
            print(f"[ERROR] Exam folder not found: {image_folder}")
            return []
        
        try:
            # Scan directory for image files with filtering
            images = [
                filename for filename in os.listdir(image_folder) 
                if (filename.lower().endswith(('.png', '.jpg', '.jpeg')) and 
                    "frame" not in filename.lower())
            ]
            
            print(f"[INFO] Found {len(images)} images in {patient_id}/{selected_exam}")
            return images
            
        except PermissionError:
            print(f"[ERROR] Permission denied accessing folder: {image_folder}")
            return []
        except Exception as e:
            print(f"[ERROR] Unexpected error loading images from {image_folder}: {e}")
            return []


    def load_videos_in_dir(self, patient_id, selected_exam):
        """
        Load all video files from a specified patient exam directory.
        
        Scans the exam folder for supported video formats that can be used
        for frame extraction and video annotation workflows. The method supports
        common medical imaging video formats and can be easily extended to
        support additional formats as needed.
        
        Args:
            patient_id (str): Unique identifier for the patient (folder name)
            selected_exam (str): Exam identifier/folder name within patient directory
            
        Returns:
            list: List of video filenames or empty list if folder not found
                  or no valid videos exist
                  
        Supported Formats:
            - QuickTime Movie (.mov): Common in medical imaging systems
            
        Extension Capability:
            Additional video formats can be easily added by extending the
            file extension check to include formats like:
            - MP4 (.mp4): Widely supported compressed video format
            - AVI (.avi): Legacy but still used in some medical systems
            - MKV (.mkv): Open standard container format
            
        Integration:
            Video files returned by this method are typically processed by
            the VideoFrameExtractor for intelligent frame extraction and
            annotation workflow integration.
        """
        # Construct the full path to the selected exam folder
        video_folder = os.path.join(self.selected_image_folder, patient_id, selected_exam)
        
        # Verify folder exists before attempting to scan
        if not os.path.exists(video_folder):
            print(f"[ERROR] Exam folder not found: {video_folder}")
            return []
        
        try:
            # Scan directory for video files
            videos = [
                filename for filename in os.listdir(video_folder)
                if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
            ]
            
            print(f"[INFO] Found {len(videos)} videos in {patient_id}/{selected_exam}")
            return videos
            
        except PermissionError:
            print(f"[ERROR] Permission denied accessing folder: {video_folder}")
            return []
        except Exception as e:
            print(f"[ERROR] Unexpected error loading videos from {video_folder}: {e}")
            return []
"""
ConfigHandler - Configuration Management for TagMed

This module handles the persistent storage and retrieval of user settings
for the TagMed annotation tool. It manages configuration data like selected
folders, annotation classes, and file paths in a JSON format.

Features:
- Automatic creation of default configuration
- Persistent storage in JSON format
- Simple get/set interface for configuration values
- UTF-8 encoding support for international characters

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import json
import os

# Resolve the settings file relative to this source file (not the current working
# directory) so TagMed works regardless of where it is launched from.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.abspath(os.path.join(_SRC_DIR, "..", "user_settings.json"))
ABS_SETTINGS_FILE = SETTINGS_FILE  # Absolute path to the settings file

class ConfigHandler:
    """
    Handles configuration management for TagMed application.
    
    This class provides a simple interface for loading, saving, and managing
    user settings and configuration data. All settings are stored persistently
    in a JSON file to maintain user preferences between sessions.
    
    Attributes:
        settings_file (str): Path to the configuration file
        settings (dict): Dictionary containing all configuration values
    """
    def __init__(self, settings_file=SETTINGS_FILE):
        """
        Initialize the ConfigHandler with specified settings file.
        
        Args:
            settings_file (str): Path to the JSON settings file (default: user_settings.json)
        """
        self.settings_file = settings_file
        self.settings = self.load()

    def load(self):
        """
        Load settings from the JSON configuration file.
        
        Creates default configuration if the file doesn't exist.
        Includes default values for all essential TagMed settings.
        
        Returns:
            dict: Configuration dictionary with all settings
        """
        if os.path.exists(self.settings_file):
            with open(self.settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
            
        # fallback to default settings if file doesn't exist
        return {
            "class_list": [],
            "selected_image_folder": "",
            "selected_anno_table_file": "",
            "selected_medical_report_file": "",
            "export_directory": "../exports",
            "image_size": [600, 600]
        }

    def save(self):
        """
        Save current settings to the JSON configuration file.
        
        Writes all configuration data to disk with proper formatting
        and UTF-8 encoding for international character support.
        """
        with open(self.settings_file, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=2)

    def get(self, key, default=None):
        """
        Retrieve a configuration value by key.
        
        Args:
            key (str): Configuration key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value or default if key not found
        """
        return self.settings.get(key, default)

    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): Configuration key to set
            value: Value to store for the given key
        """
        self.settings[key] = value


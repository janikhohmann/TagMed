import json
import os

SETTINGS_FILE = "user_settings.json"

class ConfigHandler:
    def __init__(self, settings_file=SETTINGS_FILE):
        self.settings_file = settings_file
        self.settings = self.load()

    def load(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "class_list": [],
            "selected_image_folder": "",
            "selected_anno_table_file": "",
            "selected_medical_report_file": ""
        }

    def save(self):
        with open(self.settings_file, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=2)

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = value


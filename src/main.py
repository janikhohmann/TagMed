"""
TagMed Main Application Entry Point - Medical Image and Video Annotation System

This module serves as the main entry point for the TagMed application,
a comprehensive medical image and video annotation tool designed for clinical and
research environments. The application provides advanced annotation capabilities
for medical imaging workflows including bounding boxes, polygons, and AI-assisted
segmentation.

Key Features:
- Medical image annotation with multiple annotation types
- Video frame annotation with temporal tracking capabilities
- Integration with SAM2 and MedSAM2 for AI-assisted segmentation
- Patient/exam-based data organization
- Progress tracking and annotation management
- Configurable annotation classes and workflows

The application is built using Tkinter with the Azure theme for a modern,
professional appearance suitable for medical environments.

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""


import tkinter as tk

from annotation_tool import AnnotationTool 



# main function
def main():
    """
    Initialize and launch the TagMed medical annotation application.
    
    """

    root = tk.Tk()
    app = AnnotationTool(root)
    root.title("TagMed - Medical Image and Video Annotation Tool")
    # Set the Theme
    root.tk.call("source", "azure.tcl")
    root.tk.call("set_theme", "light")

    root.geometry("1400x900") 

    root.mainloop()    
 

if __name__ == "__main__":
    main()

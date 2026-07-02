"""
Gallery Navigator - Patient Data Visualization and Annotation Interface

This module provides the main interface for viewing and annotating medical images
and videos within the TagMed application. It creates detailed patient windows
with tabbed interfaces for images and videos, annotation tools, and tracking capabilities.

Features:
- Dual-tab interface for images and videos
- Comprehensive annotation tools (bounding boxes, polygons)
- Video frame extraction and intelligent frame handling
- SAM2/MedSAM2 tracking integration
- Mask visualization and management
- Medical report integration
- Progress saving and session management

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import tkinter as tk
from tkinter import  ttk, messagebox
import os 
from PIL import Image, ImageTk
import threading

from config_handler import ConfigHandler
from annotation_loader import AnnotationLoader
from data_loader import DataLoader
from medical_record_loader import MedicalRecordLoader
from img_annotation_handler import ImgAnnotationHandler
from video_annotation_handler import VideoAnnotationHandler
from video_tracking import VideoTracking
from mask_handler import MaskHandler
from sam2_tracking import SAM2Tracking
from video_frame_extractor import VideoFrameExtractor

class GalleryNavigator:
    """
    Main interface for patient data visualization and medical annotation.
    
    This class creates and manages detailed patient windows with comprehensive
    annotation capabilities for both images and videos. It integrates multiple
    annotation handlers, tracking systems, and visualization tools to provide
    a complete medical annotation workflow.
    
    Attributes:
        root (tk.Tk): Main application window
        patient_window (tk.Toplevel): Patient-specific annotation window
        annotation_loader (AnnotationLoader): Handles annotation data persistence
        video_frame_extractor (VideoFrameExtractor): Manages video frame extraction
        img_annotation_handler (ImgAnnotationHandler): Handles image annotations
        video_annotation_handler (VideoAnnotationHandler): Handles video annotations
        video_tracking (VideoTracking): Provides tracking functionality
        sam2_tracking (SAM2Tracking): Advanced SAM2/MedSAM2 tracking
        mask_handler (MaskHandler): Manages annotation masks
    """
    
    def __init__(self, root):
        """
        Initialize the Gallery Navigator with all required components.
        
        Sets up configuration, annotation handlers, tracking systems,
        and prepares the interface for patient data visualization.
        
        Args:
            root (tk.Tk): Main application window reference
        """
        self.root = root

        # Load configuration settings
        config = ConfigHandler()
        self.refresh_configuration()

        # Initialize data management components
        self.annotation_loader = AnnotationLoader()
        self.all_annotations = self.annotation_loader.load_annotations_in_internal_list() # Load all annotations into an internal list

        self.data_loader = DataLoader()
        self.medical_record_loader = MedicalRecordLoader()
        self.img_annotation_handler = ImgAnnotationHandler(self)
        self.video_annotation_handler = VideoAnnotationHandler(self)

        # Video Frame Extractor for intelligent frame handling
        self.video_frame_extractor = VideoFrameExtractor()

        # Interface state flags
        self.video_mode = False  # Flag to indicate if video mode is active
        self.modify_mode = tk.BooleanVar(value=False)  # Flag to indicate if modify mode is active
        self.bounding_box_mode = False  # Flag to indicate if bounding box mode is active
        self.masks_visible = tk.BooleanVar(value=False)  # Flag to indicate if masks are visible
        self.click_mode = False  # Flag to indicate if click mode is active
        self.good_2_go_mode = False  # Flag to indicate if good-to-go mode is active --> not implemented yet

        # Initialize tracking and mask handling systems
        self.video_tracking = VideoTracking(self)
        self.sam2_tracking = SAM2Tracking(self)

        self.mask_handler = MaskHandler(self)
        self.create_mask_var = tk.BooleanVar(value=False)
        self.create_frame_mask_var = tk.BooleanVar(value=False)
        self.all_masks = []
        self.drawn_mask_ids = []
        self.mask_dir = "../masks"

        self.selected_image_index = None

        # Zoom and Pan variables for image canvas
        self.zoom_level = 1.0  # 1.0 = 100%, range: 1.0 to 2.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Zoom and Pan variables for video frame canvas
        self.video_zoom_level = 1.0  # 1.0 = 100%, range: 1.0 to 2.0
        self.video_pan_offset_x = 0
        self.video_pan_offset_y = 0
        self.is_video_panning = False
        self.video_pan_start_x = 0
        self.video_pan_start_y = 0

    def refresh_configuration(self):
        """
        Loads the latest configuration settings.
        """
        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.selected_anno_table_file = config.get("selected_anno_table_file")
        self.selected_medical_report_file = config.get("selected_medical_report_file")
        self.class_list = config.get("class_list")
        self.image_size = config.get("image_size", (600, 600))
        
        # reinitialize components that depend on configuration
        self.annotation_loader = AnnotationLoader()
        self.all_annotations = self.annotation_loader.load_annotations_in_internal_list()
        self.data_loader = DataLoader()
        self.medical_record_loader = MedicalRecordLoader()




    def open_patient_window(self, patient_id):
        """
        Opens a new window with detailed patient data and annotation interface.
        
        Creates a comprehensive patient window with tabbed interface for images
        and videos, medical report display, and annotation tools. This is the
        main interface for annotating medical data for a specific patient.
        
        Args:
            patient_id (str): Unique identifier for the patient
        """ 

        # refresh configuration before opening the window
        self.refresh_configuration()

        # Creating a new window
        patient_window = tk.Toplevel(self.root)
        patient_window.title(f"Annotation Gallery for {patient_id}")
        patient_window.geometry("1450x900")
        self.patient_window = patient_window

        self.patient_id = patient_id

        # Handle window close event to prompt for saving progress
        self.patient_window.protocol("WM_DELETE_WINDOW", self.saving_progress_question)

        # Shortcut bindings for common actions
        self.patient_window.bind("<Control-z>", lambda event: self.delete_last_polygon_point_manager())
        self.patient_window.bind("<Control-e>", lambda event: self.modify_annotation_manager())
        
        # Zoom and Pan bindings for image canvas
        self.patient_window.bind("<Control-MouseWheel>", self.on_zoom)
        self.patient_window.bind("<Control-Button-4>", self.on_zoom)  # Linux scroll up
        self.patient_window.bind("<Control-Button-5>", self.on_zoom)  # Linux scroll down
        
        # Crosshair control with Ctrl key
        self.patient_window.bind("<KeyPress-Control_L>", self.hide_crosshair)
        self.patient_window.bind("<KeyPress-Control_R>", self.hide_crosshair)
        self.patient_window.bind("<KeyRelease-Control_L>", self.show_crosshair)
        self.patient_window.bind("<KeyRelease-Control_R>", self.show_crosshair)


        # get exams for the patient
        exams = self.annotation_loader.get_exams(patient_id)

        top_frame = tk.Frame(patient_window)
        top_frame.pack(side="top", fill="x", padx=10, pady=10)

        # Label
        label = ttk.Label(top_frame, text="Select Exam:")
        label.pack(side="left")

        # Dropdown (Combobox)
        exam_var = tk.StringVar()
        exam_dropdown = ttk.Combobox(top_frame, textvariable=exam_var, values=exams, state="readonly")
        exam_dropdown.pack(side="left", padx=5)


        exam_dropdown.bind("<<ComboboxSelected>>", lambda event: self.on_exam_selected(event, exam_var, patient_id))

        # Notebooks for gallery
        notebook = ttk.Notebook(patient_window)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook


        """ 
        Set up the Imagetab with List of available Images, 
        Frame for the medical report and for the image itself
        """

        image_tab = tk.Frame(notebook)
        notebook.add(image_tab, text="Images")

        #Configure rows: Row 0 expands, Row 1 with fixed height 
        image_tab.grid_rowconfigure(0, weight=1)  # Row 0 should expand to fill available space
        image_tab.grid_rowconfigure(1, weight=0, minsize=200)  # Row 1 is 200 high

        # Configure columns:
        # Column 0 (left) and Column 3 (right) should have fixed width (200)
        # Column 1 (center-left) and Column 2 (center-right) should expand equally to fill the remaining space
        image_tab.grid_columnconfigure(0, weight=0, minsize=200)  # Fixed width for the left column
        image_tab.grid_columnconfigure(1, weight=1)  # Center-left column expands
        image_tab.grid_columnconfigure(2, weight=0, minsize=620)  # Fixed width for the right column

        # Create top row frames (4 frames in one row)
        image_list_frame = tk.Frame(image_tab)
        image_list_frame.grid(row=0, column=0, padx=2, sticky="nsew")

        # Create Scrollbar fpr the Listbox
        scrollbar_img = tk.Scrollbar(image_list_frame)
        scrollbar_img.pack(side="right", fill="y")

        # Listbox to display images
        self.image_listbox = tk.Listbox(image_list_frame, yscrollcommand=scrollbar_img.set, exportselection=False, selectbackground='#4a90d9')
        self.image_listbox.pack(fill="both", expand=True)

        # Link the scrollbar to the listbox
        scrollbar_img.config(command=self.image_listbox.yview)
 
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_selected)

        # Create the description_frame first
        description_frame = tk.Frame(image_tab)
        description_frame.grid(row=0, column=1, padx=2, sticky="nsew")

        # Create Scrollbar for the description box
        scrollbar_des = tk.Scrollbar(description_frame)
        scrollbar_des.pack(side="right", fill="y")

        # Create the Text widget to display the description and link it with the scrollbar
        self.des_textbox_img = tk.Text(description_frame, yscrollcommand=scrollbar_des.set, state="disabled")
        self.des_textbox_img.pack(fill="both", expand=True)

        # Link the scrollbar to the description box
        scrollbar_des.config(command=self.des_textbox_img.yview)


        image_frame = tk.Frame(image_tab, bg="lightgrey")
        image_frame.grid(row=0, column=2, padx=2, sticky="nsew")

        image_frame.grid_rowconfigure(0, weight=0)
        image_frame.grid_rowconfigure(1, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)

        # Create header frame for image name and zoom percentage
        header_frame = tk.Frame(image_frame, bg=image_frame.cget("bg"))
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 2))
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)

        # load the selected image name for header text
        try:
            header_text = self.selected_image_index.split(".")[0]
        except AttributeError:
             header_text = "No Image selected" # Fallback-Text
        except IndexError:
             header_text = self.selected_image_index 

        self.header_label_img = tk.Label(
            header_frame,
            text=header_text,
            bg=image_frame.cget("bg"),
            font=("Arial", 11, "bold")
            )
        self.header_label_img.grid(row=0, column=0, sticky="w")
        
        # Zoom percentage label
        self.zoom_label_img = tk.Label(
            header_frame,
            text="100%",
            bg=image_frame.cget("bg"),
            font=("Arial", 11, "bold"),
            fg="black"
            )
        self.zoom_label_img.grid(row=0, column=1, sticky="e", padx=(10, 0)) 

        self.image_canvas = tk.Canvas(image_frame, bg="white", highlightthickness=0) 
        self.image_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=(2, 5))

        # Bind crosshair drawing to mouse motion and clear on leave
        self.image_canvas.bind("<Motion>", self.draw_crosshair)
        self.image_canvas.bind("<Leave>", lambda event: self.image_canvas.delete("crosshair_line"))
        
        # Bind pan functionality to image canvas
        self.image_canvas.bind("<Control-ButtonPress-1>", self.start_pan)
        self.image_canvas.bind("<Control-B1-Motion>", self.pan_image)
        self.image_canvas.bind("<Control-ButtonRelease-1>", self.end_pan)

        self.setup_img_bottom_frame(image_tab)

        """ 
        Set up the Videotab with List of available Videos, 
        Frame for the medical report and for the Frames itself
        """

        video_tab = tk.Frame(notebook)
        notebook.add(video_tab, text="Videos")

        # Configure rows: Row 0 expands, Row 1 is fixed height of 40px
        video_tab.grid_rowconfigure(0, weight=1)  # Row 0 should expand to fill available space
        video_tab.grid_rowconfigure(1, weight=0, minsize=200)  # Row 1 is 200 high

        # Configure columns:
        # Column 0 (left) and Column 3 (right) should have fixed width (200)
        # Column 1 (center-left) and Column 2 (center-right) should expand equally to fill the remaining space
        video_tab.grid_columnconfigure(0, weight=0, minsize=200)  # Fixed width for the left column
        video_tab.grid_columnconfigure(1, weight=1)  # Center-left column expands
        video_tab.grid_columnconfigure(2, weight=0, minsize=620)  # Fixed width for the right column

        # Create top row frames (4 frames in one row)
        video_list_frame = tk.Frame(video_tab)
        video_list_frame.grid(row=0, column=0, padx=2, sticky="nsew")

        # Create Scrollbar fpr the Listbox
        scrollbar_video = tk.Scrollbar(video_list_frame)
        scrollbar_video.pack(side="right", fill="y")

        # Listbox to display videos
        self.video_listbox = tk.Listbox(video_list_frame, yscrollcommand=scrollbar_video.set, exportselection=False, selectbackground='#4a90d9')
        self.video_listbox.pack(fill="both", expand=True)

        # Link the scrollbar to the listbox
        scrollbar_video.config(command=self.video_listbox.yview)
 
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_selected)

        # Create the description_frame first
        description_frame = tk.Frame(video_tab)
        description_frame.grid(row=0, column=1, padx=2, sticky="nsew")

        # Create Scrollbar for the description box
        scrollbar_des = tk.Scrollbar(description_frame)
        scrollbar_des.pack(side="right", fill="y")

        # Create the Text widget to display the description and link it with the scrollbar
        self.des_textbox_video = tk.Text(description_frame, yscrollcommand=scrollbar_des.set, state="disabled")
        self.des_textbox_video.pack(fill="both", expand=True)

        # Link the scrollbar to the description box
        scrollbar_des.config(command=self.des_textbox_video.yview)


        frame_frame = tk.Frame(video_tab, bg="lightgrey")
        frame_frame.grid(row=0, column=2, padx=2, sticky="nsew")

        frame_frame.grid_rowconfigure(0, weight=0)
        frame_frame.grid_rowconfigure(1, weight=1)
        frame_frame.grid_columnconfigure(0, weight=1)

        # Create header frame for video name and zoom percentage
        video_header_frame = tk.Frame(frame_frame, bg=frame_frame.cget("bg"))
        video_header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 2))
        video_header_frame.grid_columnconfigure(0, weight=1)
        video_header_frame.grid_columnconfigure(1, weight=0)

        # load the selected video name for header text
        try:
            header_text = self.selected_video_index.split(".")[0]
        except AttributeError:
             header_text = "No Video selected" # Fallback-Text
        except IndexError:
             header_text = self.selected_video_index 

        self.header_label_video = tk.Label(
            video_header_frame,
            text=header_text,
            bg=frame_frame.cget("bg"),
            font=("Arial", 11, "bold")
            )
        self.header_label_video.grid(row=0, column=0, sticky="w")
        
        # Zoom percentage label for video
        self.zoom_label_video = tk.Label(
            video_header_frame,
            text="100%",
            bg=frame_frame.cget("bg"),
            font=("Arial", 11, "bold"),
            fg="black"
            )
        self.zoom_label_video.grid(row=0, column=1, sticky="e", padx=(10, 0)) 

        self.frame_canvas = tk.Canvas(frame_frame, bg="white", highlightthickness=0) 
        self.frame_canvas.grid(row=1, column=0, sticky="nsew")

        # Bind crosshair drawing to mouse motion on the frame canvas as well
        self.frame_canvas.bind("<Motion>", self.draw_crosshair)
        self.frame_canvas.bind("<Leave>", lambda event: self.frame_canvas.delete("crosshair_line"))
        
        # Bind zoom functionality to frame canvas (Control + Mouse Wheel)
        self.frame_canvas.bind("<Control-MouseWheel>", self.on_video_zoom)
        self.frame_canvas.bind("<Control-Button-4>", self.on_video_zoom)  # Linux scroll up
        self.frame_canvas.bind("<Control-Button-5>", self.on_video_zoom)  # Linux scroll down
        
        # Bind pan functionality to frame canvas
        self.frame_canvas.bind("<Control-ButtonPress-1>", self.start_video_pan)
        self.frame_canvas.bind("<Control-B1-Motion>", self.pan_video)
        self.frame_canvas.bind("<Control-ButtonRelease-1>", self.end_video_pan)

        self.setup_video_bottom_frame(video_tab)


    def on_exam_selected(self, event, exam_var, patient_id):
        """
        Handles selection of a medical examination from the dropdown.
        
        Updates both image and video listboxes with files from the selected exam,
        loads corresponding medical reports, and refreshes annotation colors
        to show current annotation status.
        
        Args:
            event: Tkinter event object (ComboboxSelected)
            exam_var (tk.StringVar): Variable containing selected exam name
            patient_id (str): Patient identifier for data loading
        """
        selected_exam = exam_var.get()
        self.selected_exam = selected_exam
        images = self.data_loader.load_images_in_dir(patient_id, selected_exam)
        videos = self.data_loader.load_videos_in_dir(patient_id, selected_exam)

        # clean up the listboxes before filling them and then fillign them with new data
        self.image_listbox.delete(0, tk.END)
        self.video_listbox.delete(0, tk.END)
        for image in images:
            self.image_listbox.insert(tk.END, image)
        for video in videos:
            self.video_listbox.insert(tk.END, video)

        # Clean up and update textboxes with medical reports
        # Image description textbox
        self.des_textbox_img.config(state="normal")        # Temporarily enable editing
        self.des_textbox_img.delete("1.0", tk.END)         # Clear existing content
        medical_record = self.medical_record_loader.fill_description(selected_exam)
        self.des_textbox_img.insert(tk.END, medical_record) # Insert new content
        self.des_textbox_img.config(state="disabled")      # Re-disable editing
        
        # Video description textbox
        self.des_textbox_video.config(state="normal")      # Temporarily enable editing
        self.des_textbox_video.delete("1.0", tk.END)       # Clear existing content
        self.des_textbox_video.insert(tk.END, medical_record) # Insert new content
        self.des_textbox_video.config(state="disabled")    # Re-disable editing

        self.img_annotation_handler.update_image_listbox_with_annotation_colors()
        self.video_annotation_handler.update_video_listbox_with_annotation_colors()

    def on_image_selected(self, event):
        """
        Handles selection and display of a medical image.
        
        Loads the selected image, resizes it for display, updates the header,
        and loads any existing annotations. Also refreshes the annotation
        listbox colors and mask visibility based on current settings.
        
        Args:
            event: Tkinter listbox selection event
        """

        self.image_canvas.delete("temp_boundingbox") # clean up any temporary bounding box

        self.video_mode = False
        
        # Reset zoom and pan when selecting new image
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.update_zoom_label()

        selection = self.image_listbox.curselection()
        self.selected_image = selection
        if not selection:
            return 

        selected_image = self.image_listbox.get(selection[0])

        try:
            base_name, _ = os.path.splitext(selected_image)
            header_text = base_name
        except Exception as e:
            print(f"Error. Can't create Header.'{selected_image}': {e}")
            header_text = selected_image # Fallback 

        if hasattr(self, 'header_label_img') and self.header_label_img:
            self.header_label_img.config(text=header_text)
        else:
            print("Warning: self.header_label does not exist.")

        self.selected_image_index = selected_image
        image_path = os.path.join(self.selected_image_folder, self.patient_id, self.selected_exam, selected_image)

        try:
            pil_image = Image.open(image_path)
        except (FileNotFoundError, OSError) as e:
            messagebox.showerror(
                "Image error",
                f"Could not open image:\n{image_path}\n\n{e}"
            )
            return

        width, height = self.image_size
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        self.pil_image_for_processing = pil_image.copy()
        self.original_pil_image = pil_image.copy()  # Store original for zoom operations

        self.tk_image = ImageTk.PhotoImage(pil_image)
        x = 0 
        y = 0

        if not hasattr(self, 'image_on_canvas'):
            self.image_on_canvas = self.image_canvas.create_image(x, y, anchor="nw", image=self.tk_image)
        else:
            self.image_canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

        if hasattr(self, 'img_annotation_handler'):
            self.img_annotation_handler.load_annotations_for_image(base_name) 
        else:
            print("Warning: self.img_annotation_handler not found.")
        
        self.img_annotation_handler.update_image_listbox_with_annotation_colors()
        self.toggle_mask_visibility()


    def on_video_selected(self, event):
        """
        Handles selection of a video for annotation.
        
        Intelligently provides video frames by either using existing frames
        or extracting them from the video file. Sets up the video annotation
        interface with frame navigation and displays the first frame.
        
        Uses the VideoFrameExtractor for intelligent frame handling:
        - Checks for existing extracted frames first
        - Extracts frames from video if none exist
        - Sets up frame navigation and current frame tracking
        
        Args:
            event: Tkinter listbox selection event
        """

        self.video_mode = True
        self.selected_image_index = None
        self.inference_state = None

        selection = self.video_listbox.curselection()
        if not selection:
            return 

        selected_video = self.video_listbox.get(selection[0])

        try:
            base_name, _ = os.path.splitext(selected_video)
            header_text = base_name
        except Exception as e:
            print(f"Error. Can't create Header.'{selected_video}': {e}")
            header_text = selected_video  # Fallback 

        if hasattr(self, 'header_label_img') and self.header_label_video:
            self.header_label_video.config(text=header_text)
        else:
            print("Warning: self.header_label does not exist.")

        self.selected_video_index = selected_video
        
        image_folder = os.path.join(
            self.selected_image_folder,
            self.patient_id,
            self.selected_exam
        )

        # get frames: either existing or extracted from video
        current_frames = self.video_frame_extractor.get_video_frames_intelligent(
            patient_id=self.patient_id,
            selected_exam=self.selected_exam,
            selected_video=selected_video,
            image_folder=image_folder
        )
        

        self.current_frames = current_frames
        self.current_frame_index = 0
        self.current_frame_id = current_frames[self.current_frame_index].split(".")[0].strip() if current_frames else None
        self.video_slider.set(0) # set slider back to first position
        
        # Reset zoom and pan when selecting new video
        self.video_zoom_level = 1.0
        self.video_pan_offset_x = 0
        self.video_pan_offset_y = 0
        self.update_video_zoom_label()

        self.video_annotation_handler.display_current_frame()

    def saving_progress_question(self):
        """
        Shows save dialog when closing the patient window.
        
        Presents a dialog asking whether to save annotations before closing.
        Handles three responses:
        - Yes: Saves annotations then closes window
        - No: Closes window without saving
        - Cancel: Keeps window open
        
        Also performs cleanup of temporary video frames when closing.
        """

        answer = messagebox.askyesnocancel(
            "Unsaved Progress",
            "Do you want to save your annotations before closing?",
            parent=self.patient_window
        )

        if answer is True:
            def on_saved():
                # Cleanup temporary video frames on patient switch with saving
                self.video_frame_extractor.cleanup_on_patient_switch()
                self.patient_window.destroy()
            
            self.save_annotation_gui(on_complete=on_saved)
        elif answer is False:
            # Cleanup temporary video frames on patient switch withut saving
            self.video_frame_extractor.cleanup_on_patient_switch()
            self.patient_window.destroy()



    def setup_img_bottom_frame(self, parent):
        """
        Sets up the bottom control panel for image annotation.
        
        Creates a comprehensive control interface including:
        - Annotation type dropdown (Bounding Box, Polygon)
        - Class selection dropdown
        - Action buttons (Add, Delete, Save)
        - Modify mode toggle
        - Annotation objects listbox
        - Mask visibility controls
        
        Also binds mouse events to the image canvas for interactive annotation.
        
        Args:
            parent: Parent tkinter widget to contain the bottom frame
        """
        bottom_frame = tk.Frame(parent, height=200, relief=tk.SUNKEN, borderwidth=1)
        bottom_frame.grid(row=1, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")

        # Grid configuration for bottom_frame:
        # Column 0 (Dropdowns) and 1 (Buttons) have fixed width (weight=0)
        # Column 2 (Listbox) expands (weight=1)
        bottom_frame.grid_columnconfigure(0, weight=0)
        bottom_frame.grid_columnconfigure(1, weight=0)
        bottom_frame.grid_columnconfigure(2, weight=0)
        bottom_frame.grid_columnconfigure(3, weight=0)  # Slider
        bottom_frame.grid_rowconfigure(1, weight=1) # Allows the listbox to expand vertically

        # --- Header over Controls ---
        header = tk.Label(bottom_frame, text="Annotation Properties", font=("Arial", 12, "bold"))
        header.grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(10, 5))

        # --- Column 0: Dropdowns ---
        dropdown_frame = tk.Frame(bottom_frame)
        dropdown_frame.grid(row=1, column=0, sticky="nw", padx=5, pady=5)

        tk.Label(dropdown_frame, text="Type:").pack(anchor="w", padx=5)
        self.img_annotation_type = tk.StringVar()
        type_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.img_annotation_type, values=["Bounding Box", "Polygon"], width=15, state="readonly")
        type_dropdown.pack(anchor="w", padx=5, pady=(0, 10))
        type_dropdown.current(0)

        tk.Label(dropdown_frame, text="Class:").pack(anchor="w", padx=5)
        self.img_selected_class = tk.StringVar()
        class_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.img_selected_class, values=self.class_list, width=15, state="readonly")
        class_dropdown.pack(anchor="w", padx=5)
        if self.class_list: # Set default only if list is not empty
            class_dropdown.current(0)

        # --- Column 1: Buttons and Toggle ---
        controls_frame = tk.Frame(bottom_frame)
        controls_frame.grid(row=1, column=1, sticky="n", padx=5, pady=5) # sticky="n" for top alignment

        add_button = tk.Button(controls_frame, text="Add", width=12, command=self.img_annotation_handler.add_annotation)
        add_button.pack(pady=2, fill=tk.X)

        delete_button = tk.Button(controls_frame, text="Delete", width=12, command=self.img_annotation_handler.delete_annotation)
        delete_button.pack(pady=2, fill=tk.X)

        save_button = tk.Button(controls_frame, text="Save", width=12,
                               command=lambda: self.save_annotation_gui(on_complete=None),
                               bg="#d4fcd4")
        save_button.pack(pady=2, fill=tk.X)

        # Modify Mode Toggle (Checkbutton)
        modify_toggle = ttk.Checkbutton(controls_frame, text="Modify Mode", 
                                        variable=self.modify_mode, 
                                        command=self.img_annotation_handler.modify_annotation)
        modify_toggle.pack(pady=(10, 2), anchor="w") # Slightly more space above, left-aligned

        # --- Column 2: Listbox for Objects ---
        listbox_area_frame = tk.Frame(bottom_frame)
        listbox_area_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 10), pady=5)
        listbox_area_frame.grid_columnconfigure(0, weight=1)
        listbox_area_frame.grid_rowconfigure(1, weight=1)

        tk.Label(listbox_area_frame, text="Objects:").grid(row=0, column=0, sticky="sw")
        self.img_annotation_listbox = tk.Listbox(listbox_area_frame, height=6, width=50, selectmode=tk.SINGLE)
        self.img_annotation_listbox.grid(row=1, column=0, sticky="nw", pady=(2, 0))


        self.img_annotation_listbox.bind("<<ListboxSelect>>", lambda event: self.img_annotation_handler.on_annotation_selected(event))

        if self.image_canvas: # Only bind if canvas exists
            self.image_canvas.bind("<ButtonPress-1>", self.img_annotation_handler.on_press)
            self.image_canvas.bind("<B1-Motion>", self.img_annotation_handler.on_drag)
            self.image_canvas.bind("<ButtonRelease-1>", self.img_annotation_handler.on_release)
        else:
            print("Warning: self.image_canvas is not initialized. Bindings not set.")

        # # Slider + additional controls (Column 3) - stays empty for images
        slider_frame = tk.Frame(bottom_frame, width=600, height=150)
        slider_frame.grid(row=1, column=4, sticky="nsew", padx=(10, 10), pady=0)
        slider_frame.grid_propagate(False)


        # Navigation Buttons below (left bottom)
        tracker_controls_row = tk.Frame(slider_frame)
        tracker_controls_row.pack(side="bottom", anchor="w", fill="x", pady=(15, 10))

        # Create Mask from Annotation Checkbox
        self.create_mask_toggle_button = ttk.Checkbutton(
            tracker_controls_row,
            text="Create Mask from Annotation",
            variable=self.create_mask_var
        )
        self.create_mask_toggle_button.pack(side="left", padx=5)

        # Mask Checkbox
        self.mask_toggle_button = ttk.Checkbutton(
            tracker_controls_row,
            text="Show Masks",
            variable=self.masks_visible,
            command=self.toggle_mask_visibility
        )
        self.mask_toggle_button.pack(side="left", padx=5) 






    def setup_video_bottom_frame(self, parent):
        """
        Sets up the bottom control panel for video annotation and tracking.
        
        Creates a comprehensive video annotation interface including:
        - Annotation type dropdown (Bounding Box, Polygon)
        - Class selection dropdown
        - Action buttons (Add, Delete, Save)
        - Modify mode toggle
        - Annotation objects listbox
        - Video frame navigation slider with controls
        - Tracking system dropdown (Simple, SAM2, MedSAM2 variants)
        - Start tracking button
        - Mask visibility and creation controls
        
        Also binds mouse events to the frame canvas for interactive annotation.
        
        Args:
            parent: Parent tkinter widget to contain the bottom frame
        """
        bottom_frame = tk.Frame(parent, height=200, relief=tk.SUNKEN, borderwidth=1)
        bottom_frame.grid(row=1, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")

        bottom_frame.grid_columnconfigure(0, weight=0)  # Dropdowns
        bottom_frame.grid_columnconfigure(1, weight=0)  # Buttons + Toggles
        bottom_frame.grid_columnconfigure(2, weight=0)  # Listbox
        bottom_frame.grid_columnconfigure(3, weight=0)  # Slider
        bottom_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = tk.Label(bottom_frame, text="Annotation Properties", font=("Arial", 12, "bold"))
        header.grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(10, 5))

        # Dropdowns (Column 0)
        dropdown_frame = tk.Frame(bottom_frame)
        dropdown_frame.grid(row=1, column=0, sticky="nw", padx=5, pady=5)

        tk.Label(dropdown_frame, text="Type:").pack(anchor="w", padx=5)
        self.video_annotation_type = tk.StringVar()
        type_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.video_annotation_type,
                                    values=["Bounding Box", "Polygon"], width=15, state="readonly")
        type_dropdown.pack(anchor="w", padx=5, pady=(0, 10))
        type_dropdown.current(0)

        tk.Label(dropdown_frame, text="Class:").pack(anchor="w", padx=5)
        self.video_selected_class = tk.StringVar()
        class_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.video_selected_class,
                                    values=self.class_list, width=15, state="readonly")
        class_dropdown.pack(anchor="w", padx=5, pady=(0, 10))
        if self.class_list:
            class_dropdown.current(0)

        # Buttons (Column 1)
        controls_frame = tk.Frame(bottom_frame)
        controls_frame.grid(row=1, column=1, sticky="n", padx=5, pady=5)

        add_button = tk.Button(controls_frame, text="Add", width=12,
                            command=self.video_annotation_handler.add_annotation)
        add_button.pack(pady=2, fill=tk.X)

        delete_button = tk.Button(controls_frame, text="Delete", width=12)
        delete_button.pack(pady=2, fill=tk.X)
        delete_button.bind("<Button-1>", lambda event: self.video_annotation_handler.delete_annotation())
        delete_button.bind("<Double-Button-1>", lambda event: self.delete_annotations_for_all_frames_question())

        save_button = tk.Button(controls_frame, text="Save", width=12,
                                command=lambda: self.save_annotation_gui(on_complete=None),
                                bg="#d4fcd4")
        save_button.pack(pady=2, fill=tk.X)

        modify_toggle = ttk.Checkbutton(controls_frame, text="Modify Mode",
                                        variable=self.modify_mode,
                                        command=self.video_annotation_handler.modify_annotation)
        modify_toggle.pack(pady=(10, 2), anchor="w")
        

        # Listbox (Column 2)
        listbox_area_frame = tk.Frame(bottom_frame)
        listbox_area_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 10), pady=5)
        listbox_area_frame.grid_columnconfigure(0, weight=1)
        listbox_area_frame.grid_rowconfigure(1, weight=1)

        tk.Label(listbox_area_frame, text="Objects:").grid(row=0, column=0, sticky="sw")
        self.video_annotation_listbox = tk.Listbox(listbox_area_frame, height=6, width=50, selectmode=tk.SINGLE)
        self.video_annotation_listbox.grid(row=1, column=0, sticky="nw", pady=(2, 0))

        self.video_annotation_listbox.bind("<<ListboxSelect>>",
                                        lambda event: self.video_annotation_handler.on_annotation_selected(event))

        if self.frame_canvas:
            self.video_annotation_handler.frame_canvas = self.frame_canvas
            self.frame_canvas.bind("<ButtonPress-1>", self.video_annotation_handler.on_press)
            self.frame_canvas.bind("<B1-Motion>", self.video_annotation_handler.on_drag)
            self.frame_canvas.bind("<ButtonRelease-1>", self.video_annotation_handler.on_release)
        else:
            print("Warning: self.frame_canvas is not initialized. Bindings not set.")

        # Slider + additional controls (Column 3)
        slider_frame = tk.Frame(bottom_frame, width=600, height=150)
        slider_frame.grid(row=1, column=3, sticky="nsew", padx=(10, 10), pady=0)
        slider_frame.grid_propagate(False)

        self.frame_index_label = tk.Label(slider_frame, text="Frame 1 / 1", font=("Arial", 10))
        self.frame_index_label.pack(pady=(5, 0))

        self.video_slider = ttk.Scale(
            slider_frame,
            from_=0, to=10,
            orient='horizontal',
            length=600,
            command=lambda val: self.on_slider_changed(int(float(val)))
        )
        self.video_slider.pack(pady=(5, 5))


        # New horizontal buttons below the slider
        button_row = tk.Frame(slider_frame)
        button_row.pack(pady=(0, 0))

        left_button = tk.Button(button_row, text="←", width=4,
                                command=lambda: self.video_slider.set(self.video_slider.get() - 1))
        left_button.pack(side="left", padx=5)

        self.play_stop_button = tk.Button(button_row, text="▶", width=6,
                                command=self.toggle_play_stop)
        self.play_stop_button.pack(side="left", padx=5)

        right_button = tk.Button(button_row, text="→", width=4,
                                command=lambda: self.video_slider.set(self.video_slider.get() + 1))
        right_button.pack(side="left", padx=5)

        # Separator under slider unit
        separator = tk.Frame(slider_frame, height=1, bg="black")  
        separator.pack(fill="x",  padx=0, pady=4)

        # Navigation Buttons below
        tracker_controls_row = tk.Frame(slider_frame)
        tracker_controls_row.pack(pady=(15, 2), fill="x")

        # Tracking-Dropdown
        self.tracking_type = tk.StringVar()
        tracking_dropdown = ttk.Combobox(tracker_controls_row, textvariable=self.tracking_type,
                                        values=["Simple", "SAM3","SAM2 large", "SAM2 tiny","MedSAM2", "MedSAM2 US Heart", "MedSAM2 MRI Liver Lesion"], width=20, state="readonly")
        tracking_dropdown.pack(side="left", padx=5)
        tracking_dropdown.current(0)

        # Start Tracking Button
        start_tracker_button = tk.Button(tracker_controls_row, text="Start Tracking", width=16, bg="#d4fcd4",
                                         command=self.video_tracking.tracking_starter)
        start_tracker_button.pack(side="left", padx=5)

        # # Mask-Checkbox
        self.mask_toggle_button = ttk.Checkbutton(
            tracker_controls_row,
            text="Show Masks",
            variable=self.masks_visible,
            command=self.toggle_mask_visibility
        )
        self.mask_toggle_button.pack(side="left", padx=5)

        # Create Mask from Annotation Checkbox
        self.create_frame_mask_toggle_button = ttk.Checkbutton(
            tracker_controls_row,
            text="Create Mask for single frame",
            variable=self.create_frame_mask_var
        )
        self.create_frame_mask_toggle_button.pack(side="left", padx=5)

        # Tracker Prompt Button (Placeholder)
        # tracker_prompt_button = tk.Button(tracker_controls_row, text="Tracker Prompts", width=16)
        # tracker_prompt_button.pack(side="left", padx=5)






    def on_slider_changed(self, value):
        """
        Handles video frame navigation slider changes.
        
        Updates the current frame index and refreshes the display when
        the user moves the frame navigation slider. Ensures the value
        is within valid range and updates the video annotation handler.
        
        Args:
            value (float): New slider position (converted to int frame index)
        """
        if not hasattr(self, 'current_frames') or not self.current_frames:
            return
        value = int(value)
        if 0 <= value < len(self.current_frames):
            self.current_frame_index = value
            self.current_frame_id = self.current_frames[self.current_frame_index].split(".")[0].strip() if self.current_frames else None
            self.video_annotation_handler.display_current_frame()



    def save_annotation_gui(self, on_complete=None):
        """
        Shows progress spinner while saving annotations in background.
        
        Creates a modal progress dialog and saves annotations in a separate
        thread to prevent UI freezing. Automatically closes the dialog when
        saving is complete and executes optional completion callback.
        
        Args:
            on_complete (callable, optional): Function to call after saving completes
        """

        popup = tk.Toplevel()
        popup.title("Saving Annotations")
        popup.geometry("300x100")
        popup.resizable(False, False)
        popup.transient(self.patient_window)
        popup.grab_set()

        tk.Label(popup, text="Please wait... Saving annotations").pack(pady=10)

        spinner = ttk.Progressbar(popup, mode="indeterminate", length=250)
        spinner.pack(pady=10)
        spinner.start(10)

        def worker():
            try:
                self.annotation_loader.save_annotations_to_anno_table()
            finally:
                def cleanup():
                    spinner.stop()
                    popup.destroy()
                    if on_complete:
                        on_complete()

                self.patient_window.after(0, cleanup)

        # Start the worker thread
        threading.Thread(target=worker, daemon=True).start()


    def wrong_annotation_warning_gui(self, status):
        """
        Shows warning dialog for conflicting annotation types.
        
        Displays appropriate warning message when user tries to mix
        different annotation types (bounding boxes and polygons) on
        the same image, which is not allowed in TagMed.
        
        Args:
            status (str): Type of conflict - "Polygon" or "Bounding Box"
        """

        if status == "Polygon":
            info = messagebox.showinfo(
                "Complication with annotation types",
                "There are already bounding box annotations saved for this image. Please use only one annotation type per image.",
                parent=self.patient_window
            )

        if status == "Bounding Box":
            info = messagebox.showinfo(
                "Complication with annotation types",
                "There are already polygon annotations saved for this image. Please use only one annotation type per image.",
                parent=self.patient_window
            )

    def select_annotation_before_tracking_gui(self):
        """
        Shows error dialog when tracking is attempted without annotation selection.
        
        Displays an informational message box reminding the user to select
        an annotation before starting the tracking process.
        """
        info = messagebox.showinfo(
            "Tracking Error",
            "Please select an annotation to be tracked.",
            parent=self.patient_window
        )

    def delete_annotations_for_all_frames_question(self):
        """
        Shows confirmation dialog for deleting all video annotations.
        
        Asks user to confirm deletion of all annotations for the currently
        selected video. If confirmed, delegates to video tracking system
        to perform the bulk deletion operation.
        """

        answer = messagebox.askyesno(
            "Deleting all Annotations",
            f"Do you really want to delete all Annotations for video {self.selected_video_index}?",
            parent=self.patient_window
        )

        if answer is True:
            self.video_tracking.delete_all_annotations_for_one_video()

    def ask_for_sam2_download(self):
        """
        Shows dialog asking user permission to download SAM2 model.
        
        Prompts user when SAM2 model is not available locally and needs
        to be downloaded for tracking functionality. Returns user's choice.
        
        Returns:
            bool: True if user agrees to download, False otherwise
        """
        answer = messagebox.askyesno(
            "Downloading SAM2",
            f"The SAM2 model is not yet loaded. Should this model be loaded?",
            parent=self.patient_window
        )
        
        return answer
    
    def ask_for_medsam2_download(self):
        """
        Shows dialog asking user permission to download MedSAM2 model.
        
        Prompts user when MedSAM2 model is not available locally and needs
        to be downloaded for medical-specific tracking functionality. 
        Returns user's choice.
        
        Returns:
            bool: True if user agrees to download, False otherwise
        """
        answer = messagebox.askyesno(
            "Downloading MedSAM2",
            f"The MedSAM2 model is not yet loaded. Should this model be loaded?",
            parent=self.patient_window
        )
        
        return answer
    


    def toggle_mask_visibility(self):
        """
        Toggles the visibility of annotation masks on images/videos.
        
        Determines whether currently viewing image or video tab, then
        either shows or hides masks based on the current toggle state.
        Delegates to mask_handler for actual mask loading/clearing operations.
        """

        current_tab = self.notebook.select()
        if str(self.image_canvas).startswith(current_tab):
            self.showing_video = False
        elif str(self.frame_canvas).startswith(current_tab):
            self.showing_video = True


        if self.masks_visible.get():
            if self.showing_video:
                #print("[DEBUG] Masks will be shown.")
                self.mask_handler.load_masks_for_frame()
            else:
                self.mask_handler.load_masks_for_image()
        else:
            #print("[DEBUG] Mask will not be shown.")
            self.mask_handler.clear_all_masks()



    def delete_last_polygon_point_manager(self):
        """
        Manages deletion of the last polygon point across different tabs.
        
        Determines which tab (image or video) is currently active and
        delegates the polygon point deletion to the appropriate annotation
        handler. Bound to Ctrl+Z keyboard shortcut.
        """

        current_tab = self.notebook.select()

        if str(self.image_canvas).startswith(current_tab):
            # print("[DEBUG] Using img_annotation_handler")
            self.img_annotation_handler.delete_last_polygon_point()
        elif str(self.frame_canvas).startswith(current_tab):
            # print("[DEBUG] Using video_annotation_handler")
            self.video_annotation_handler.delete_last_polygon_point()

    def modify_annotation_manager(self):

        """
        Manages modification mode toggle across different tabs.
        
        Determines which tab (image or video) is currently active and
        delegates the modify mode action to the appropriate annotation
        handler. Bound to Ctrl+E keyboard shortcut. E for edit.
        """

        current_tab = self.notebook.select()

        if self.modify_mode.get():
            self.modify_mode.set(False)
            if str(self.image_canvas).startswith(current_tab):
                # print("[DEBUG] Using img_annotation_handler")
                self.img_annotation_handler.modify_annotation()
            elif str(self.frame_canvas).startswith(current_tab):
                # print("[DEBUG] Using video_annotation_handler")
                self.video_annotation_handler.modify_annotation()
            
        else:
            self.modify_mode.set(True)
            if str(self.image_canvas).startswith(current_tab):
                # print("[DEBUG] Using img_annotation_handler")
                self.img_annotation_handler.modify_annotation()
            elif str(self.frame_canvas).startswith(current_tab):
                # print("[DEBUG] Using video_annotation_handler")
                self.video_annotation_handler.modify_annotation()

    def wait_for_tracking_gui(self, on_complete=None):
        """
        Shows progress spinner while tracking runs in background.
        
        Creates a modal progress dialog with indeterminate progress bar
        while tracking operations execute in a separate thread. Prevents
        UI freezing during potentially long-running tracking processes.
        
        Args:
            on_complete (callable, optional): Function to call when tracking completes.
                                            If None, defaults to SAM2 tracking method.
        """

        popup = tk.Toplevel()
        popup.title("Tracking in progress")
        popup.geometry("300x100")
        popup.resizable(False, False)
        popup.transient(self.patient_window)
        popup.grab_set()

        tk.Label(popup, text="Please wait... Your Annotation is being tracked").pack(pady=10)

        spinner = ttk.Progressbar(popup, mode="indeterminate", length=250)
        spinner.pack(pady=10)
        spinner.start(10)

        def worker():
            try:
                # Execute the callback function if provided, otherwise default to SAM2
                if on_complete:
                    on_complete()
                else:
                    self.sam2_tracking.sam2_tracking_method()
            finally:
                def cleanup():
                    spinner.stop()
                    popup.destroy()

                self.patient_window.after(0, cleanup)

        # Start the worker thread
        threading.Thread(target=worker, daemon=True).start()


    def hide_crosshair(self, event):
        """
        Hides the crosshair when Ctrl key is pressed.
        
        Args:
            event: Keyboard event
        """
        if hasattr(self, 'image_canvas'):
            self.image_canvas.delete("crosshair_line")
        if hasattr(self, 'frame_canvas'):
            self.frame_canvas.delete("crosshair_line")
    
    def show_crosshair(self, event):
        """
        Re-enables crosshair drawing when Ctrl key is released.
        
        Args:
            event: Keyboard event
        """
        # Crosshair will be redrawn on next mouse motion
        pass

    def draw_crosshair(self,event):
        """
        Draws a crosshair on the image canvas at the mouse position.
        
        When the mouse moves over the image canvas, this function draws
        horizontal and vertical lines intersecting at the cursor position,
        creating a crosshair effect. Previous crosshair lines are removed
        before drawing new ones to avoid clutter.
        
        Args:
            event: Tkinter mouse motion event containing cursor coordinates
        """

        # Use the actual canvas that triggered the event (works for image_canvas and frame_canvas)
        canvas = event.widget

        # Remove existing crosshair lines on this canvas only
        try:
            canvas.delete("crosshair_line")
        except Exception:
            pass

        # Determine which annotation-type variable to check depending on the canvas
        annotation_type_var = None
        try:
            if hasattr(self, 'image_canvas') and canvas == self.image_canvas:
                annotation_type_var = getattr(self, 'img_annotation_type', None)
            elif hasattr(self, 'frame_canvas') and canvas == self.frame_canvas:
                annotation_type_var = getattr(self, 'video_annotation_type', None)
        except Exception:
            annotation_type_var = None

        # Resolve the actual string value (supports tk.StringVar or plain str)
        ann_type = None
        if isinstance(annotation_type_var, tk.StringVar):
            try:
                ann_type = annotation_type_var.get()
            except Exception:
                ann_type = None
        elif isinstance(annotation_type_var, str):
            ann_type = annotation_type_var

        # Only draw crosshair when the current annotation type for this canvas is "Bounding Box"
        if ann_type == "Bounding Box" and not bool(event.state & 0x0004):  # dont draw crosshair while zooming (Ctrl pressed)
            x = event.x
            y = event.y
            width = canvas.winfo_width()
            height = canvas.winfo_height()

            # Draw new crosshair lines
            canvas.create_line(0, y, width, y, fill="red", dash=(4, 2), tags="crosshair_line")
            canvas.create_line(x, 0, x, height, fill="red", dash=(4, 2), tags="crosshair_line")
        else:
            # Don't draw anything for other annotation types
            return


    # ========== Zoom and Pan Methods ==========

    def on_zoom(self, event):
        """
        Handles zoom events with Ctrl+MouseWheel.
        
        Zooms in or out on the image canvas while keeping coordinates
        relative to the original image unchanged. Limits zoom range
        between 100% (1.0) and 300% (3.0).
        
        Args:
            event: Mouse wheel event with delta information
        """
        # # Use the actual canvas that triggered the event (works for image_canvas and frame_canvas)
        # canvas = event.widget
        # # Remove existing crosshair lines on this canvas only
        # try:
        #     canvas.delete("crosshair_line")
        # except Exception:
        #     pass

        # Only zoom if cursor is over image canvas
        if not hasattr(self, 'image_canvas') or not hasattr(self, 'original_pil_image'):
            return
            
        # Check if event is from image canvas area
        try:
            widget = event.widget
            # Navigate up to find if we're in image_canvas context
            while widget:
                if widget == self.image_canvas:
                    break
                widget = widget.master if hasattr(widget, 'master') else None
            else:
                # Not over image canvas
                return
        except:
            return

        # Determine zoom direction
        if event.num == 4 or event.delta > 0:  # Scroll up - zoom in
            zoom_factor = 1.1
        elif event.num == 5 or event.delta < 0:  # Scroll down - zoom out
            zoom_factor = 0.9
        else:
            return

        # Calculate new zoom level
        new_zoom = self.zoom_level * zoom_factor
        
        # Limit zoom range: 100% to 300%
        new_zoom = max(1.0, min(3.0, new_zoom))
        
        if new_zoom == self.zoom_level:
            return  # No change needed
        
        self.zoom_level = new_zoom
        
        # Reset pan when zooming out to 100%
        if self.zoom_level == 1.0:
            self.pan_offset_x = 0
            self.pan_offset_y = 0
        
        self.update_image_display()
        self.update_zoom_label()

    def update_image_display(self):
        """
        Updates the image display with current zoom and pan settings.
        
        Applies zoom transformation to the original image and positions
        it according to pan offsets. Redraws all annotations at correct
        positions.
        """
        if not hasattr(self, 'original_pil_image'):
            return
            
        # Calculate new image size based on zoom
        width, height = self.image_size
        new_width = int(width * self.zoom_level)
        new_height = int(height * self.zoom_level)
        
        # Resize image
        zoomed_image = self.original_pil_image.resize(
            (new_width, new_height), 
            Image.Resampling.LANCZOS
        )
        
        self.tk_image = ImageTk.PhotoImage(zoomed_image)
        
        # Update canvas image with pan offset
        if hasattr(self, 'image_on_canvas'):
            self.image_canvas.coords(
                self.image_on_canvas, 
                self.pan_offset_x, 
                self.pan_offset_y
            )
            self.image_canvas.itemconfig(self.image_on_canvas, image=self.tk_image)
        
        # Redraw all annotations with new coordinates
        self.redraw_annotations_with_zoom()

    def redraw_annotations_with_zoom(self):
        """
        Redraws all annotations with zoom and pan transformation applied.
        
        Clears existing annotation visuals and reloads them from the
        annotation data, applying current zoom and pan transformations.
        """
        if not hasattr(self, 'img_annotation_handler'):
            return
            
        # Store current selections and state
        current_selection = None
        if hasattr(self.img_annotation_handler, 'listbox_index'):
            current_selection = self.img_annotation_handler.listbox_index
        
        # Clear visual annotations (but keep data)
        for rect_id in self.img_annotation_handler.drawn_rect_ids:
            try:
                self.image_canvas.delete(rect_id)
            except:
                pass
        self.img_annotation_handler.drawn_rect_ids.clear()
        
        # Reload annotations (will use transform_coordinates)
        if self.selected_image_index:
            base_name = self.selected_image_index.split(".")[0]
            self.img_annotation_handler.load_annotations_for_image(base_name)
        
        # Reload masks with zoom and pan transformation
        if self.masks_visible.get() and hasattr(self, 'mask_handler'):
            self.mask_handler.load_masks_for_image()

    def update_zoom_label(self):
        """
        Updates the zoom percentage label display.
        """
        if hasattr(self, 'zoom_label_img'):
            percentage = int(self.zoom_level * 100)
            self.zoom_label_img.config(text=f"{percentage}%")

    def start_pan(self, event):
        """
        Starts panning mode when Ctrl+Click on image.
        
        Args:
            event: Mouse button press event
        """
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.image_canvas.config(cursor="fleur")  # Change cursor to indicate panning

    def pan_image(self, event):
        """
        Handles image panning during Ctrl+Drag.
        
        Args:
            event: Mouse motion event
        """
        if not self.is_panning:
            return
            
        # Calculate movement delta
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        # Update pan offsets
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        
        # Update start position for next delta
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        # Update display
        if hasattr(self, 'image_on_canvas'):
            self.image_canvas.coords(
                self.image_on_canvas,
                self.pan_offset_x,
                self.pan_offset_y
            )
        
        # Redraw annotations with new pan offset
        self.redraw_annotations_with_zoom()

    def end_pan(self, event):
        """
        Ends panning mode when releasing Ctrl+Click.
        
        Args:
            event: Mouse button release event
        """
        self.is_panning = False
        self.image_canvas.config(cursor="")  # Reset cursor

    def toggle_play_stop(self):
        """
        Toggles between play and stop for video playback.
        
        Changes button appearance and functionality based on playback state:
        - If not playing: Shows ▶ (play), starts video playback
        - If playing: Shows ⏸ (pause), stops video playback
        """
        if not hasattr(self.video_annotation_handler, 'is_playing'):
            self.video_annotation_handler.is_playing = False
        
        if self.video_annotation_handler.is_playing:
            # Currently playing - stop it
            self.video_annotation_handler.stop_video()
            self.play_stop_button.config(text="▶")
        else:
            # Not playing - start it
            self.video_annotation_handler.play_video()
            self.play_stop_button.config(text="⏸")

    def transform_coordinates(self, image_x, image_y):
        """
        Transforms image coordinates to canvas coordinates with zoom and pan.
        
        Converts coordinates from the original image space (100% zoom)
        to the current canvas display space with zoom and pan applied.
        
        Args:
            image_x: X coordinate in original image space
            image_y: Y coordinate in original image space
            
        Returns:
            tuple: (canvas_x, canvas_y) in current display space
        """
        canvas_x = image_x * self.zoom_level + self.pan_offset_x
        canvas_y = image_y * self.zoom_level + self.pan_offset_y
        return canvas_x, canvas_y

    def inverse_transform_coordinates(self, canvas_x, canvas_y):
        """
        Transforms canvas coordinates back to original image coordinates.
        
        Converts coordinates from the current canvas display space
        back to the original image space (100% zoom), removing the
        effects of zoom and pan transformations.
        
        Args:
            canvas_x: X coordinate in current canvas space
            canvas_y: Y coordinate in current canvas space
            
        Returns:
            tuple: (image_x, image_y) in original image space
        """
        image_x = (canvas_x - self.pan_offset_x) / self.zoom_level
        image_y = (canvas_y - self.pan_offset_y) / self.zoom_level
        return image_x, image_y
    
    # ========== Video Zoom and Pan Methods ==========
    
    def on_video_zoom(self, event):
        """
        Handles zoom events for video frame canvas.
        """
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:  # Scroll up - zoom in
            zoom_factor = 1.1
        elif event.num == 5 or event.delta < 0:  # Scroll down - zoom out
            zoom_factor = 0.9
        else:
            return

        # Calculate new zoom level
        new_zoom = self.video_zoom_level * zoom_factor
        
        # Limit zoom range: 100% to 300%
        new_zoom = max(1.0, min(3.0, new_zoom))
        
        if new_zoom == self.video_zoom_level:
            return  # No change needed
        
        self.video_zoom_level = new_zoom
        
        # Reset pan when zooming out to 100%
        if self.video_zoom_level == 1.0:
            self.video_pan_offset_x = 0
            self.video_pan_offset_y = 0
        
        self.update_video_display()
        self.update_video_zoom_label()
    
    def update_video_display(self):
        """
        Updates the video frame display with current zoom and pan settings.
        """
        if not hasattr(self, 'original_video_frame'):
            return
            
        # Calculate new image size based on zoom
        width, height = self.image_size
        new_width = int(width * self.video_zoom_level)
        new_height = int(height * self.video_zoom_level)
        
        # Resize frame
        zoomed_frame = self.original_video_frame.resize(
            (new_width, new_height), 
            Image.Resampling.LANCZOS
        )
        
        self.tk_frame = ImageTk.PhotoImage(zoomed_frame)
        
        # Update canvas with pan offset
        if hasattr(self.video_annotation_handler, 'frame_on_canvas'):
            self.frame_canvas.coords(
                self.video_annotation_handler.frame_on_canvas, 
                self.video_pan_offset_x, 
                self.video_pan_offset_y
            )
            self.frame_canvas.itemconfig(self.video_annotation_handler.frame_on_canvas, image=self.tk_frame)
        
        # Redraw all annotations with new coordinates
        self.redraw_video_annotations_with_zoom()
    
    def redraw_video_annotations_with_zoom(self):
        """
        Redraws all video frame annotations with zoom and pan transformation.
        """
        if not hasattr(self, 'video_annotation_handler'):
            return
            
        # Clear visual annotations (but keep data)
        for rect_id in self.video_annotation_handler.drawn_rect_ids:
            try:
                self.frame_canvas.delete(rect_id)
            except:
                pass
        self.video_annotation_handler.drawn_rect_ids.clear()
        
        # Reload annotations (will use transform_video_coordinates)
        self.video_annotation_handler.load_annotations_for_frame()
        
        # Reload masks with zoom and pan transformation
        if self.masks_visible.get() and hasattr(self, 'mask_handler'):
            self.mask_handler.load_masks_for_frame()
    
    def update_video_zoom_label(self):
        """
        Updates the video zoom percentage label display.
        """
        if hasattr(self, 'zoom_label_video'):
            percentage = int(self.video_zoom_level * 100)
            self.zoom_label_video.config(text=f"{percentage}%")
    
    def start_video_pan(self, event):
        """
        Starts panning mode for video frame canvas.
        """
        self.is_video_panning = True
        self.video_pan_start_x = event.x
        self.video_pan_start_y = event.y
        self.frame_canvas.config(cursor="fleur")
    
    def pan_video(self, event):
        """
        Handles video frame panning during Ctrl+Drag.
        """
        if not self.is_video_panning:
            return
            
        # Calculate movement delta
        dx = event.x - self.video_pan_start_x
        dy = event.y - self.video_pan_start_y
        
        # Update pan offsets
        self.video_pan_offset_x += dx
        self.video_pan_offset_y += dy
        
        # Update start position for next delta
        self.video_pan_start_x = event.x
        self.video_pan_start_y = event.y
        
        # Update display
        if hasattr(self.video_annotation_handler, 'frame_on_canvas'):
            self.frame_canvas.coords(
                self.video_annotation_handler.frame_on_canvas,
                self.video_pan_offset_x,
                self.video_pan_offset_y
            )
        
        # Redraw annotations with new pan offset
        self.redraw_video_annotations_with_zoom()
    
    def end_video_pan(self, event):
        """
        Ends panning mode for video frame canvas.
        """
        self.is_video_panning = False
        self.frame_canvas.config(cursor="")
    
    def transform_video_coordinates(self, image_x, image_y):
        """
        Transforms video frame image coordinates to canvas coordinates with zoom and pan.
        """
        canvas_x = image_x * self.video_zoom_level + self.video_pan_offset_x
        canvas_y = image_y * self.video_zoom_level + self.video_pan_offset_y
        return canvas_x, canvas_y
    
    def inverse_transform_video_coordinates(self, canvas_x, canvas_y):
        """
        Transforms video canvas coordinates back to original frame coordinates.
        """
        image_x = (canvas_x - self.video_pan_offset_x) / self.video_zoom_level
        image_y = (canvas_y - self.video_pan_offset_y) / self.video_zoom_level
        return image_x, image_y



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


class GalleryNavigator:
    def __init__(self, root):
        self.root = root

        config = ConfigHandler()
        self.selected_image_folder = config.get("selected_image_folder")
        self.selected_anno_table_file = config.get("selected_anno_table_file")
        self.selected_medical_report_file = config.get("selected_medical_report_file")
        self.class_list = config.get("class_list")
        self.image_size = config.get("image_size", (600, 600))  # Default image size if not set

        self.annotation_loader = AnnotationLoader()
        self.all_annotations = self.annotation_loader.load_annotations_in_internal_list() # Load all annotations into an internal list

        self.data_loader = DataLoader()
        self.medical_record_loader = MedicalRecordLoader()
        self.img_annotation_handler = ImgAnnotationHandler(self)
        self.video_annotation_handler = VideoAnnotationHandler(self)

        self.video_mode = False  # Flag to indicate if video mode is active
        self.modify_mode = tk.BooleanVar(value=False)  # Flag to indicate if modify mode is active
        self.bounding_box_mode = False  # Flag to indicate if bounding box mode is active
        self.mask_mode = False  # Flag to indicate if mask mode is active
        self.masks_visible = tk.BooleanVar(value=False)  # Flag to indicate if masks are visible
        self.click_mode = False  # Flag to indicate if click mode is active
        self.good_2_go_mode = False  # Flag to indicate if good-to-go mode is active

        self.video_tracking = VideoTracking(self)
        self.all_masks = []



    def open_patient_window(self, patient_id):
        """
        Opens new window with Details and Gallery for a patient.
        Gets executetd by double clining on a patient - on_double_click().
        """ 

        # Creating a new window
        patient_window = tk.Toplevel(self.root)
        patient_window.title(f"Annotation Gallery for {patient_id}")
        patient_window.geometry("1400x900")
        self.patient_window = patient_window

        self.patient_id = patient_id

        self.patient_window.protocol("WM_DELETE_WINDOW", self.saving_progress_question)

        exams = self.annotation_loader.get_exams(patient_id)
        patient_annotations = self.annotation_loader.filter_annotations_for_patient(self.all_annotations, patient_id) # filter annotations for the patient from internal list

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

        """ 
        Set up the Imagetab with List of available Images, 
        Frame for the medical report and for the image itself
        """

        image_tab = tk.Frame(notebook)
        notebook.add(image_tab, text="Images")

        #Configure rows: Row 0 expands, Row 1 is fixed height of 40px
        image_tab.grid_rowconfigure(0, weight=1)  # Row 0 should expand to fill available space
        image_tab.grid_rowconfigure(1, weight=0, minsize=200)  # Row 1 is 200 high

        # Configure columns:
        # Column 0 (left) and Column 3 (right) should have fixed width (200)
        # Column 1 (center-left) and Column 2 (center-right) should expand equally to fill the remaining space
        image_tab.grid_columnconfigure(0, weight=0, minsize=200)  # Fixed width for the left column
        image_tab.grid_columnconfigure(1, weight=1)  # Center-left column expands
        image_tab.grid_columnconfigure(2, weight=0, minsize=620)  # Fixed width for the right column

        # Create top row frames (4 frames in one row)
        image_list_frame = tk.Frame(image_tab, bg="blue")
        image_list_frame.grid(row=0, column=0, padx=2, sticky="nsew")

        # Create Scrollbar fpr the Listbox
        scrollbar_img = tk.Scrollbar(image_list_frame)
        scrollbar_img.pack(side="right", fill="y")

        # Listbox to display images
        self.image_listbox = tk.Listbox(image_list_frame, yscrollcommand=scrollbar_img.set)
        self.image_listbox.pack(fill="both", expand=True)

        # Link the scrollbar to the listbox
        scrollbar_img.config(command=self.image_listbox.yview)
 
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_selected)

        # Create the description_frame first
        description_frame = tk.Frame(image_tab, bg="black")
        description_frame.grid(row=0, column=1, padx=2, sticky="nsew")

        # Create Scrollbar for the description box
        scrollbar_des = tk.Scrollbar(description_frame)
        scrollbar_des.pack(side="right", fill="y")

        # Create the Text widget to display the description and link it with the scrollbar
        self.des_textbox_img = tk.Text(description_frame, yscrollcommand=scrollbar_des.set)
        self.des_textbox_img.pack(fill="both", expand=True)

        # Link the scrollbar to the description box
        scrollbar_des.config(command=self.des_textbox_img.yview)


        image_frame = tk.Frame(image_tab, bg="lightgrey")
        image_frame.grid(row=0, column=2, padx=2, sticky="nsew")

        image_frame.grid_rowconfigure(0, weight=0)
        image_frame.grid_rowconfigure(1, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)

        try:
            header_text = self.selected_image_index.split(".")[0]
        except AttributeError:
             header_text = "No Image selected" # Fallback-Text
        except IndexError:
             header_text = self.selected_image_index 

        self.header_label_img = tk.Label(
            image_frame,
            text=header_text,
            bg=image_frame.cget("bg"),
            font=("Arial", 11, "bold")
            )
        self.header_label_img.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 2)) 

        self.image_canvas = tk.Canvas(image_frame, bg="white", highlightthickness=0) 
        self.image_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=(2, 5))

        self.setup_img_bottom_frame(image_tab)

        """ 
        Set up the Videotab with List of available Videos, 
        Frame for the medical report and for the Frames itself
        """

        video_tab = tk.Frame(notebook)
        notebook.add(video_tab, text="Videos")

                #Configure rows: Row 0 expands, Row 1 is fixed height of 40px
        video_tab.grid_rowconfigure(0, weight=1)  # Row 0 should expand to fill available space
        video_tab.grid_rowconfigure(1, weight=0, minsize=200)  # Row 1 is 200 high

        # Configure columns:
        # Column 0 (left) and Column 3 (right) should have fixed width (200)
        # Column 1 (center-left) and Column 2 (center-right) should expand equally to fill the remaining space
        video_tab.grid_columnconfigure(0, weight=0, minsize=200)  # Fixed width for the left column
        video_tab.grid_columnconfigure(1, weight=1)  # Center-left column expands
        video_tab.grid_columnconfigure(2, weight=0, minsize=620)  # Fixed width for the right column

        # Create top row frames (4 frames in one row)
        video_list_frame = tk.Frame(video_tab, bg="blue")
        video_list_frame.grid(row=0, column=0, padx=2, sticky="nsew")

        # Create Scrollbar fpr the Listbox
        scrollbar_video = tk.Scrollbar(video_list_frame)
        scrollbar_video.pack(side="right", fill="y")

        # Listbox to display videos
        self.video_listbox = tk.Listbox(video_list_frame, yscrollcommand=scrollbar_video.set)
        self.video_listbox.pack(fill="both", expand=True)

        # Link the scrollbar to the listbox
        scrollbar_video.config(command=self.video_listbox.yview)
 
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_selected)

        # Create the description_frame first
        description_frame = tk.Frame(video_tab, bg="black")
        description_frame.grid(row=0, column=1, padx=2, sticky="nsew")

        # Create Scrollbar for the description box
        scrollbar_des = tk.Scrollbar(description_frame)
        scrollbar_des.pack(side="right", fill="y")

        # Create the Text widget to display the description and link it with the scrollbar
        self.des_textbox_video = tk.Text(description_frame, yscrollcommand=scrollbar_des.set)
        self.des_textbox_video.pack(fill="both", expand=True)

        # Link the scrollbar to the description box
        scrollbar_des.config(command=self.des_textbox_video.yview)


        frame_frame = tk.Frame(video_tab, bg="lightgrey")
        frame_frame.grid(row=0, column=2, padx=2, sticky="nsew")

        frame_frame.grid_rowconfigure(0, weight=0)
        frame_frame.grid_rowconfigure(1, weight=1)
        frame_frame.grid_columnconfigure(0, weight=1)

        #print("Video selected: ",self.selected_video_index)
        try:
            header_text = self.selected_video_index.split(".")[0]
        except AttributeError:
             header_text = "No Video selected" # Fallback-Text
        except IndexError:
             header_text = self.selected_video_index 

        self.header_label_video = tk.Label(
            frame_frame,
            text=header_text,
            bg=frame_frame.cget("bg"),
            font=("Arial", 11, "bold")
            )
        self.header_label_video.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 2)) 

        self.frame_canvas = tk.Canvas(frame_frame, bg="white", highlightthickness=0) 
        self.frame_canvas.grid(row=1, column=0, sticky="nsew")

        self.setup_video_bottom_frame(video_tab)


    def on_exam_selected(self, event, exam_var, patient_id):
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

        #self.create_internal_mask_list(patient_id, selected_exam)

        # clean up the textboxes before filling them and then filling them with new medical reports
        self.des_textbox_img.delete("1.0", tk.END)
        self.des_textbox_video.delete("1.0", tk.END)
        medical_record = self.medical_record_loader.fill_description(selected_exam)
        self.des_textbox_img.insert(tk.END, medical_record)
        self.des_textbox_video.insert(tk.END, medical_record)

        self.img_annotation_handler.update_image_listbox_with_annotation_colors()
        self.video_annotation_handler.update_video_listbox_with_annotation_colors()

    def on_image_selected(self, event):
        """
        Function to display the selected image on canvas and update the header.
        """

        self.video_mode = False

        selection = self.image_listbox.curselection()
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

        pil_image = Image.open(image_path)

        width, height = self.image_size
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        self.pil_image_for_processing = pil_image.copy() 

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


    def on_video_selected(self, event):
        """
        Function to select one video from the listbox.
        Searches for frames of respective video and shows the first one.
        """

        self.video_mode = True

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
        current_frames = self.video_annotation_handler.get_all_video_frames(self.patient_id, self.selected_exam)
        self.current_frames = current_frames
        self.current_frame_index = 0
        self.current_frame_id = current_frames[self.current_frame_index].split(".")[0].strip() if current_frames else None

        self.video_annotation_handler.display_current_frame()

    def saving_progress_question(self):
        """
        Opens Popup window when you want to close the patient window.
        Asks if you want to save your progress, saves, and closes window afterwards. 
        Pops up everytime you want to close the patient window. Does not check for changes. 
        """

        answer = messagebox.askyesnocancel(
            "Unsaved Progress",
            "Do you want to save your annotations before closing?",
            parent=self.patient_window
        )

        if answer is True:
            def on_saved():
                self.patient_window.destroy()
            
            self.save_annotation_gui(on_complete=on_saved)
        elif answer is False:
            self.patient_window.destroy()



    def setup_img_bottom_frame(self, parent):
        """
        Setup for bottom frame - sets grid configuration, calls img_annotation_handler
        """
        bottom_frame = tk.Frame(parent, height=200, relief=tk.SUNKEN, borderwidth=1) # Etwas Relief zum Debuggen
        bottom_frame.grid(row=1, column=0, columnspan=3, padx=2, pady=2, sticky="nsew")

        # Grid-Konfiguration für bottom_frame:
        # Spalte 0 (Dropdowns) und 1 (Buttons) haben feste Breite (weight=0)
        # Spalte 2 (Listbox) dehnt sich aus (weight=1)
        bottom_frame.grid_columnconfigure(0, weight=0)
        bottom_frame.grid_columnconfigure(1, weight=0)
        bottom_frame.grid_columnconfigure(2, weight=1)
        bottom_frame.grid_rowconfigure(1, weight=1) # Erlaubt der Listbox, sich vertikal auszudehnen

        # --- Header über den Steuerelementen ---
        header = tk.Label(bottom_frame, text="Annotation Properties", font=("Arial", 12, "bold"))
        header.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5))

        # --- Spalte 0: Dropdowns ---
        dropdown_frame = tk.Frame(bottom_frame)
        dropdown_frame.grid(row=1, column=0, sticky="nw", padx=5, pady=5)

        tk.Label(dropdown_frame, text="Type:").pack(anchor="w", padx=5)
        self.img_annotation_type = tk.StringVar()
        type_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.img_annotation_type, values=["Bounding Box", "Polygon", "Magic Wand"], width=15)
        type_dropdown.pack(anchor="w", padx=5, pady=(0, 10))
        type_dropdown.current(0)

        tk.Label(dropdown_frame, text="Class:").pack(anchor="w", padx=5)
        self.img_selected_class = tk.StringVar()
        class_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.img_selected_class, values=self.class_list, width=15)
        class_dropdown.pack(anchor="w", padx=5)
        if self.class_list: # Setze Default nur, wenn Liste nicht leer
            class_dropdown.current(0)

        # --- Spalte 1: Buttons und Toggle ---
        controls_frame = tk.Frame(bottom_frame)
        controls_frame.grid(row=1, column=1, sticky="n", padx=5, pady=5) # sticky="n" für Top-Alignment

        add_button = tk.Button(controls_frame, text="Add", width=12, command=self.img_annotation_handler.add_annotation)
        add_button.pack(pady=2, fill=tk.X)

        delete_button = tk.Button(controls_frame, text="Delete", width=12, command=self.img_annotation_handler.delete_annotation)
        delete_button.pack(pady=2, fill=tk.X)

        save_button = tk.Button(controls_frame, text="Save", width=12,
                               command=lambda: self.save_annotation_gui(on_complete=None),
                               bg="#d4fcd4")
        save_button.pack(pady=2, fill=tk.X)

        # Modify Mode Toggle (Checkbutton)
        modify_toggle = ttk.Checkbutton(controls_frame, text="Modify Mode", variable=self.modify_mode, command=self.img_annotation_handler.modify_annotation)
        modify_toggle.pack(pady=(10, 2), anchor="w") # Etwas Abstand nach oben, linksbündig

        # --- Spalte 2: Listbox für Objects ---
        listbox_area_frame = tk.Frame(bottom_frame)
        listbox_area_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 10), pady=5)
        listbox_area_frame.grid_columnconfigure(0, weight=1)
        listbox_area_frame.grid_rowconfigure(1, weight=1)

        tk.Label(listbox_area_frame, text="Objects:").grid(row=0, column=0, sticky="sw", padx=(0, 0))

        self.img_annotation_listbox = tk.Listbox(listbox_area_frame, height=6, width=50, selectmode=tk.SINGLE)
        self.img_annotation_listbox.grid(row=1, column=0, sticky="nw", pady=(2, 0))


        self.img_annotation_listbox.bind("<<ListboxSelect>>", lambda event: self.img_annotation_handler.on_annotation_selected(event))

        if self.image_canvas: # Nur binden, wenn Canvas existiert
            self.image_canvas.bind("<ButtonPress-1>", self.img_annotation_handler.on_press)
            self.image_canvas.bind("<B1-Motion>", self.img_annotation_handler.on_drag)
            self.image_canvas.bind("<ButtonRelease-1>", self.img_annotation_handler.on_release)
        else:
            print("Warnung: self.image_canvas ist nicht initialisiert. Bindings nicht gesetzt.")

        # Slider + zusätzliche Steuerungen (Spalte 3) - stays empty for images
        slider_frame = tk.Frame(bottom_frame, width=600, height=150)
        slider_frame.grid(row=1, column=3, sticky="nsew", padx=(10, 10), pady=0)
        slider_frame.grid_propagate(False)

        # Optional: Referenz zum Frame speichern
        self.bottom_frame = bottom_frame


    def setup_video_bottom_frame(self, parent):
        """
        Setup for bottom frame - sets grid configuration, calls video_annotation_handler
        """
        bottom_frame = tk.Frame(parent, height=200, relief=tk.SUNKEN, borderwidth=1)
        bottom_frame.grid(row=1, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")

        bottom_frame.grid_columnconfigure(0, weight=0)  # Dropdowns
        bottom_frame.grid_columnconfigure(1, weight=0)  # Buttons + Toggles
        bottom_frame.grid_columnconfigure(2, weight=1)  # Listbox
        bottom_frame.grid_columnconfigure(3, weight=0)  # Slider
        bottom_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = tk.Label(bottom_frame, text="Annotation Properties", font=("Arial", 12, "bold"))
        header.grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(10, 5))

        # Dropdowns (Spalte 0)
        dropdown_frame = tk.Frame(bottom_frame)
        dropdown_frame.grid(row=1, column=0, sticky="nw", padx=5, pady=5)

        tk.Label(dropdown_frame, text="Type:").pack(anchor="w", padx=5)
        self.video_annotation_type = tk.StringVar()
        type_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.video_annotation_type,
                                    values=["Bounding Box", "Polygon"], width=15)
        type_dropdown.pack(anchor="w", padx=5, pady=(0, 10))
        type_dropdown.current(0)

        tk.Label(dropdown_frame, text="Class:").pack(anchor="w", padx=5)
        self.video_selected_class = tk.StringVar()
        class_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.video_selected_class,
                                    values=self.class_list, width=15)
        class_dropdown.pack(anchor="w", padx=5, pady=(0, 10))
        if self.class_list:
            class_dropdown.current(0)

        # Buttons (Spalte 1)
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

        # Listbox (Spalte 2)
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
            print("Warnung: self.frame_canvas ist nicht initialisiert. Bindings nicht gesetzt.")

        # Slider + zusätzliche Steuerungen (Spalte 3)
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

        
        # Neue horizontale Leiste unter dem Slider
        button_row = tk.Frame(slider_frame)
        button_row.pack(pady=(0, 0))

        left_button = tk.Button(button_row, text="←", width=4,
                                command=lambda: self.video_slider.set(self.video_slider.get() - 1))
        left_button.pack(side="left", padx=5)

        right_button = tk.Button(button_row, text="→", width=4,
                                command=lambda: self.video_slider.set(self.video_slider.get() + 1))
        right_button.pack(side="left", padx=5)

        # Seperator under slider unit
        separator = tk.Frame(slider_frame, height=1, bg="black")  # Hellgrau
        separator.pack(fill="x",  padx=0, pady=4)

        # Navigation Buttons unterhalb
        tracker_controls_row = tk.Frame(slider_frame)
        tracker_controls_row.pack(pady=(15, 2), fill="x")

        # Tracking-Dropdown
        self.tracking_type = tk.StringVar()
        tracking_dropdown = ttk.Combobox(tracker_controls_row, textvariable=self.tracking_type,
                                        values=["Simple", "SAM 2"], width=12)
        tracking_dropdown.pack(side="left", padx=5)
        tracking_dropdown.current(0)

        # Start Tracking Button
        start_tracker_button = tk.Button(tracker_controls_row, text="Start Tracking", width=16, bg="#d4fcd4",
                                         command=self.video_tracking.tracking_starter)
        start_tracker_button.pack(side="left", padx=5)

        # Masken-Checkbox
        self.mask_toggle_button = ttk.Checkbutton(
            tracker_controls_row,
            text="Show Masks",
            variable=self.masks_visible,
            command=self.video_annotation_handler.toggle_mask_visibility
        )
        self.mask_toggle_button.pack(side="left", padx=5)

        # Tracker Prompt Button (Platzhalter)
        tracker_prompt_button = tk.Button(tracker_controls_row, text="Tracker Prompts", width=16)
        tracker_prompt_button.pack(side="left", padx=5)



    def on_slider_changed(self, value):
        """
        Triggered when slider is moved, updates current frame and image.
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
        shows spinner while saving runs in background.
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
            info = messagebox.showinfo(
                "Tracking Error",
                "Please select an annotation to be tracked.",
                parent=self.patient_window
            )

    def delete_annotations_for_all_frames_question(self):

        answer = messagebox.askyesno(
            "Deleting all Annotations",
            f"Do you really want to delete all Annotations for video {self.selected_video_index}?",
            parent=self.patient_window
        )

        if answer is True:
            self.video_tracking.delete_all_annotations_for_one_video()

    def ask_for_sam2_download(self):
        answer = messagebox.askyesno(
            "Downloading SAM2",
            f"The SAM2 model is not yet loaded. Should this model be loaded?",
            parent=self.patient_window
        )
        
        return answer



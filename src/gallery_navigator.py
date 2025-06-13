import tkinter as tk
from tkinter import  ttk, messagebox
import os 
from PIL import Image, ImageTk

from config_handler import ConfigHandler
from annotation_loader import AnnotationLoader
from data_loader import DataLoader
from medical_record_loader import MedicalRecordLoader


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

        self.video_mode = False  # Flag to indicate if video mode is active
        self.modify_mode = False  # Flag to indicate if modify mode is active
        self.bounding_box_mode = False  # Flag to indicate if bounding box mode is active
        self.mask_mode = False  # Flag to indicate if mask mode is active
        self.click_mode = False  # Flag to indicate if click mode is active
        self.good_2_go_mode = False  # Flag to indicate if good-to-go mode is active



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

        self.patient_window.protocol("WM_DELETE_WINDOW", self.saving_progress_question)

        exams = self.annotation_loader.get_exams(patient_id)
        patient_annotations = self.annotation_loader.filter_annotations_for_patient(self.all_annotations, patient_id) # filter annotations for the patient from internal list

        print(patient_annotations)

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
        self.frame_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=(2, 5))

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

        image_path = os.path.join(self.image_folder, selected_image)

        pil_image = Image.open(image_path)

        width, height = self.image_size
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

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
            
            self.annotation_loader.save_annotations_to_anno_table(on_complete=on_saved)
        elif answer is False:
            self.patient_window.destroy()
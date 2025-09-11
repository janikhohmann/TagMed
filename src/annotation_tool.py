"""
TagMed Annotation Tool - Main class for medical image annotation

This class represents the main application of TagMed, a tool for annotating 
medical images and videos. It provides a user-friendly GUI 
for managing patients, images, and annotations.

Features:
- Patient overview with progress display
- Configuration of image folders, annotation databases, and medical reports
- Class management for annotations
- Integration with Gallery Navigator for detailed annotation

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Text
import pandas as pd

from config_handler import ConfigHandler
from annotation_loader import AnnotationLoader
from gallery_navigator import GalleryNavigator

class AnnotationTool:
    """
    Main class of the TagMed annotation tool application.
    
    This class creates and manages the main user interface of TagMed.
    It displays an overview of all available patients with their annotation progress
    and allows configuration of the system.
    
    Attributes:
        root (tk.Tk): Main window of the application
        tree (ttk.Treeview): Table for displaying patient data
        full_data (list): Complete patient data for search function
        config (ConfigHandler): Configuration manager
        annotation_loader (AnnotationLoader): Loads and manages annotation data
    """

    def __init__(self, root):
        """
        Initializes the TagMed Annotation Tool application.

        Creates the entire user interface including menu bar, search bar,
        patient list, and loads the configuration.

        Args:
            root (tk.Tk): Main window of the Tkinter application
        """

        # === INITIAL CONFIGURATION ===
        self.root = root
        self.root.title("TagMed - Annotation Tool")
        self.root.geometry("1400x900")

        # === CREATE MENU BAR ===
        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.helpmenu = tk.Menu(self.menubar, tearoff=0)

        # File menu: Configuration of folders and files
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(label="Add Image Folder", command=self.select_image_folder)
        self.filemenu.add_command(label="Add Annotation Database", command=self.select_anno_table)
        self.filemenu.add_command(label="Add Medical Reports", command=self.select_medical_reports)
        self.filemenu.add_command(label="Manage Classes", command=self.open_class_manager)
        self.filemenu.add_command(label="Close", command=exit)

        # Help menu: Documentation and imprint
        self.menubar.add_cascade(label="Other", menu=self.helpmenu)
        self.helpmenu.add_command(label="Help", command=self.open_help_window)
        self.helpmenu.add_command(label="Imprint", command=self.open_imprint_window)

        # Add menu bar to window
        self.root.config(menu=self.menubar)

        # === CONTROL BAR (Search + Refresh Button) ===
        bt_frame = tk.Frame(root, height=50)
        bt_frame.pack(padx=10, pady=5, fill="x", expand=False)
        bt_frame.pack_propagate(False)  # Prevent frame from resizing to fit its children

        # Data buffer for search function
        self.full_data = []

        # === SEARCH BAR ===
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(bt_frame, textvariable=self.search_var, width=40)
        search_entry.pack(side="left", padx=10, pady=5, fill="x", expand=False)
        search_entry.bind("<KeyRelease>", self.filter_treeview)  # Search on key release

        # === REFRESH BUTTON ===
        self.bt_refresh_text = tk.StringVar(value="Load Data")
        self.bt_refresh = tk.Button(bt_frame, textvariable=self.bt_refresh_text, command=self.populate_tree)
        self.bt_refresh.pack(side="right", pady=5, padx=10)

        # === PATIENT LIST (TREEVIEW) ===
        frame = tk.Frame(root)
        frame.pack(padx=10, fill="both", expand=True)

        # Define table columns
        columns = ("ID", "Exams",  "annotated_images", "total_img","progress")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)

        # Enable sorting by clicking on column headers
        for col in columns:
            self.tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(_col, False))

        # Define column headers
        self.tree.heading("ID", text="Patient-ID")
        self.tree.heading("Exams", text="Exams")
        self.tree.heading("annotated_images", text="Annotated Images")
        self.tree.heading("total_img", text="Images (total)")
        self.tree.heading("progress", text="Progress")

        # Adjust column widths
        self.tree.column("ID", width=100, anchor="center")
        self.tree.column("Exams", width=150, anchor="center")
        self.tree.column("annotated_images", width=150, anchor="center")
        self.tree.column("total_img", width=200, anchor="center")
        self.tree.column("progress", width=200, anchor="center")

        # === SCROLLBAR FOR TABLE ===
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Layout arrangement
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === EVENT-BINDINGS ===
        self.tree.bind("<Double-1>", self.on_double_click)  # Double-click opens patient
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Save on close

        # === LOAD CONFIGURATION ===
        self.config = ConfigHandler()
        self.class_list = self.config.get("class_list", [])
        self.selected_image_folder = self.config.get("selected_image_folder", "")
        self.selected_anno_table_file = self.config.get("selected_anno_table_file", "")
        self.selected_medical_report_file = self.config.get("selected_medical_report_file", "")

        # Persist configuration
        self.config.set("class_list", self.class_list)
        self.config.set("selected_image_folder", self.selected_image_folder)
        self.config.set("selected_anno_table_file", self.selected_anno_table_file)
        self.config.set("selected_medical_report_file", self.selected_medical_report_file)
        self.config.save()

        # === INITIALIZE ANNOTATION LOADER ===
        self.annotation_loader = AnnotationLoader()


    def populate_tree(self):
        """
        Loads all patient data and fills the main table.

        Shows a progress dialog during the loading process and calculates
        for each patient:
        - Number of exams and total files
        - Number of annotated images
        - Progress bar based on annotation degree

        The data is displayed in the table and stored in
        self.full_data for the search function.
        """

        # === CREATE PROGRESS POPUP ===
        popup = tk.Toplevel()
        popup.geometry("300x100")
        tk.Label(popup, text="Load Data...").pack(pady=10)
        progress = ttk.Progressbar(popup, orient="horizontal", length=250, mode="determinate")
        progress.pack(pady=5)
        percent_label = tk.Label(popup, text="0%")
        percent_label.pack()

        # === CLEAR TABLE ===
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.full_data.clear()

        # === LOAD PATIENT DATA ===
        all_patients = self.annotation_loader.available_patients()
        total = len(all_patients)
        # print(all_patients)

        # ==== READ ANNOTATION TABLE AND CHECK IF IT EXISTS ===
        try:
            anno_table = pd.read_csv(self.selected_anno_table_file, sep=";")
        except FileNotFoundError:
            print(f"[ERROR] Annotation table file not found: {self.selected_anno_table_file}")
            messagebox.showinfo("ERROR", f"Annotation table file not found: {self.selected_anno_table_file}\n\nPlease check the file path or create a new one.")


        # === PROCESS EACH PATIENT ===
        for i, patient_id in enumerate(all_patients, 1):
            # Collect statistics
            exams, total_files = self.annotation_loader.count_exams_and_files(patient_id)
            annotated_images = self.annotation_loader.count_annotated_images(patient_id)

            # Calculate progress
            try:
                progress_ratio = annotated_images / total_files
            except ZeroDivisionError:
                progress_ratio = 0.0

            # Create visual progress bar
            percent_value = int(progress_ratio * 100)
            bar_length = 20
            filled_length = int(bar_length * progress_ratio)
            progress_bar = "█" * filled_length + "░" * (bar_length - filled_length)
            progress_display = f"{progress_bar} {percent_value}%"

            # Create row and add to table
            row = (patient_id, exams, annotated_images, total_files, progress_display)
            self.tree.insert("", "end", values=row)
            self.full_data.append(row)

            # Update progress popup
            progress["value"] = (i / total) * 100
            percent_label.config(text=f"{int((i / total) * 100)}%")
            popup.update_idletasks()

        # === CLEANUP ===
        popup.destroy()
        self.bt_refresh_text.set("Refresh")  # Change button text after initial loading


    def treeview_sort_column(self, col, reverse):
        """
        Sorts the patient list according to a specific column.
        
        Triggered by clicking on a column header. Can handle both numerical
        and alphabetical sorting.
        
        Args:
            col (str): Name of the column to be sorted
            reverse (bool): True for descending, False for ascending sorting
        """
        # extract data from the table
        data = [(self.tree.set(child, col), child) for child in self.tree.get_children('')]

        # sort intelligently: numerically if possible, otherwise alphabetically
        try:
            data.sort(key=lambda t: float(t[0]) if t[0] != "NN" else float("-inf"), reverse=reverse)
        except ValueError:
            data.sort(key=lambda t: t[0], reverse=reverse)

        # Apply new sorted order to table
        for index, (val, child) in enumerate(data):
            self.tree.move(child, '', index)

        # Next click should use reverse sorting
        self.tree.heading(col, command=lambda: self.treeview_sort_column(col, not reverse))

    def filter_treeview(self, event=None):
        """
        Filters the patient list based on the search input.
        
        Only searches the patient ID column and displays only matching entries.
        Triggered automatically each time text is entered in the search bar.
        
        Args:
            event: Tkinter event (not used, required for event binding)
        """
        search_term = self.search_var.get().lower()

        # Clear the table
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Show only matching entries
        for row in self.full_data:
            if search_term in str(row[0]).lower():  # Search only in patient ID (Index 0)
                self.tree.insert("", "end", values=row)

    def on_double_click(self, patient_id, event=None):
        """
        Is Triggered on double-clicking a patient in the list.

        Opens the Gallery Navigator for the selected patient,
        enabling detailed annotation.

        Args:
            patient_id: Automatically passed (not used)
            event: Tkinter Event (not used, required for event binding)
        """
        item = self.tree.selection()
        if item:
            # extract patient ID from selected item
            patient_id = self.tree.item(item, "values")[0]
            self.patient_id = patient_id

            # Open Gallery Navigator for detailed annotation
            navigator = GalleryNavigator(self.root)
            navigator.open_patient_window(patient_id)

    def select_image_folder(self):
        """
        Opens a dialog to select the image folder.

        The selected folder is saved in the configuration and
        contains the patient folders with their medical images/videos.
        """
        folder_path = filedialog.askdirectory(title="Choose a directory with your image data")
        if folder_path:
            self.selected_image_folder = folder_path
            self.config.set("selected_image_folder", folder_path)
            self.config.save()  # save immediately

            # Update Annotation Loader with new path
            self.annotation_loader = AnnotationLoader()

            # Automatically reload data
            self.populate_tree()
            
            print(f"[INFO] File directory updated: {folder_path}")
            messagebox.showinfo("Success", f"Image folder updated to:\n{folder_path}\n\nData refreshed automatically.")

    def select_medical_reports(self):
        """
        Opens a dialog to select the medical reports.

        The file (CSV) contains additional medical information
        about the patients and is displayed in the description.
        """
        medical_report_file = filedialog.askopenfilename(
            title="Choose a file with your medical reports",
            filetypes=[("table files", "*.csv *.xlsx")]
        )
        if medical_report_file:
            self.selected_medical_report_file = medical_report_file
            self.config.set("selected_medical_report_file", medical_report_file)
            self.config.save()  # save immediately
            
            print(f"[INFO] Medical Reports updated: {medical_report_file}")
            messagebox.showinfo("Success", f"Medical reports file updated to:\n{medical_report_file}")

    def select_anno_table(self):
        """
        Opens a dialog to select the annotation database.

        The CSV file stores all annotations. If no file is
        selected, a new annotation table is automatically created.
        """
        anno_table_file = filedialog.askopenfilename(
            title="Choose a annotation database",
            filetypes=[("table files", "*.csv *.xlsx")]
        )
        if anno_table_file:
            self.selected_anno_table_file = anno_table_file
            self.config.set("selected_anno_table_file", anno_table_file)
            self.config.save()  # Sofort speichern
            
            # Annotation Loader mit neuer Datenbank aktualisieren
            self.annotation_loader = AnnotationLoader()
            
            # Automatisch Daten neu laden
            self.populate_tree()
            
            print(f"[INFO] Annotation Database updated: {anno_table_file}")
            messagebox.showinfo("Success", f"Annotation database updated to:\n{anno_table_file}\n\nData refreshed automatically.")
        else:
            # automatic creation of a new annotation table --> [TBD]should also be done when starting without one
            messagebox.showinfo("Info", "If you do not select a file, a new file will be created for you.")
            self.annotation_loader.create_default_anno_table()

            # load data after creating the new table
            self.populate_tree()



    def open_class_manager(self):
        """
        Opens a window for managing annotation classes.

        Allows adding and deleting classes that should be available for
        annotation (e.g. "Tumor", "Vessel", etc.).
        The class list is automatically saved in the configuration.
        """
        # === CLASS MANAGEMENT WINDOW CREATION ===
        class_window = tk.Toplevel()
        class_window.title("Manage Classes")

        # Ensure class list exists
        self.class_list = getattr(self, "class_list", [])

        # === LISTBOX FOR EXISTING CLASSES ===
        listbox = tk.Listbox(class_window, height=8, width=30)
        listbox.pack(padx=10, pady=5)

        def update_listbox():
            """Updates the display of classes in the listbox."""
            listbox.delete(0, tk.END)
            for cls in self.class_list:
                listbox.insert(tk.END, cls)

        update_listbox()

        # === ENTRY FOR NEW CLASS ===
        entry = tk.Entry(class_window, width=25)
        entry.pack(padx=10)

        def add_class():
            """Adds a new class to the list."""
            new_class = entry.get().strip()
            if new_class and new_class not in self.class_list:
                self.class_list.append(new_class)
                update_listbox()
                entry.delete(0, tk.END)

        def delete_class():
            """Deletes the selected class from the list."""
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                del self.class_list[index]
                update_listbox()

        def on_close_class_window():

            """Saves the class list when closing the window."""
            # Save configuration
            self.config.set("class_list", self.class_list)
            self.config.save()
            
            print(f"[INFO] Following classes selected: {self.class_list}")
            messagebox.showinfo("Success", f"Class list updated!\nClasses: {', '.join(self.class_list) if self.class_list else 'None'}")
            class_window.destroy()

        # === BUTTONS ===
        button_frame = tk.Frame(class_window)
        button_frame.pack(pady=5)

        class_window.protocol("WM_DELETE_WINDOW", on_close_class_window)

        tk.Button(button_frame, text="Add", command=add_class).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Delete", command=delete_class).grid(row=0, column=1, padx=5)

    def on_close(self):
        """
        Called when the application is closed.
        
        Saves all configuration changes and exits the application cleanly.
        """
        self.config.save()
        self.root.destroy()

    def open_help_window(self):
        """
        Opens a help window with instructions.
        
        Displays basic information on how to use TagMed in a
        scrollable text area.
        """
        # === HELP WINDOW ===
        help_window = tk.Toplevel()
        help_window.geometry("500x500")
        help_window.title("Help")

        # === SCROLLED TEXT AREA ===
        text_box = scrolledtext.ScrolledText(help_window, wrap="word", width=60, height=20)
        text_box.pack(padx=10, pady=10, fill="both", expand=True)

        # define text formatting
        text_box.tag_config("title", font=("Arial", 14, "bold"))
        text_box.tag_config("bold", font=("Arial", 12, "bold"))

        # === INSERT HELP TEXT ===
        text_box.insert("end", "Welcome to Help\n", "title")
        text_box.insert("end", "\nGeneral:\n", "bold")
        text_box.insert("end", "This tool was designed to annotate sensitive images that are difficult to interpret.\n\n")
        text_box.insert("end", "For further information please contact us.\n")
        text_box.insert("end", "Contact details can be found in the imprint.\n")

        # Make text non-editable
        text_box.config(state="disabled")

        # Close button
        button_close = tk.Button(help_window, text="Close", command=help_window.destroy)
        button_close.pack(pady=10)

    def open_imprint_window(self):
        """Opens a window with the imprint.

        Displays contact information and legal details of the developing
        institution (University Hospital Düsseldorf).
        """
        # === IMPRINT WINDOW ===
        imprint_window = tk.Toplevel()
        imprint_window.geometry("400x400")
        imprint_window.title("Imprint")

        # === TEXT AREA ===
        text_box = Text(imprint_window, wrap="word", width=60, height=10)
        text_box.pack(padx=10, pady=10, fill="both", expand=True)

        # define text formatting
        text_box.tag_config("title", font=("Arial", 12, "bold"))
        text_box.tag_config("bold", font=("Arial", 10, "bold"))
        text_box.tag_config("normal", font=("Arial", 10))

        # === INSERT IMPRINT TEXT ===
        text_box.insert("end", "Imprint:\n\n", "title")
        text_box.insert("end", "M.Sc., Janik Hohmann\n", "normal")
        text_box.insert("end", "Clinic for Gastroenterology, Hepatology and Infectiology\n", "normal")
        text_box.insert("end", "Director: Prof. Dr. med. Tom Lüdde\n\n", "normal")
        text_box.insert("end", "University Hospital Düsseldorf\n", "bold")
        text_box.insert("end", "Building 13.58\n", "normal")
        text_box.insert("end", "Moorenstr. 5\n", "normal")
        text_box.insert("end", "40225 Düsseldorf\n", "normal")
        text_box.insert("end", "Germany\n\n", "normal")
        text_box.insert("end", "Phone:\t+49 211 81-05083\n", "normal")
        text_box.insert("end", "E-mail:\tJanik.Hohmann@med.uni-duesseldorf.de\n", "normal")
        text_box.insert("end", "Internet:\twww.uniklinik-duesseldorf.de\n", "normal")

        # Make text non-editable
        text_box.config(state="disabled")

        # Close button
        button_close = tk.Button(imprint_window, text="Close", command=imprint_window.destroy)
        button_close.pack(pady=10)
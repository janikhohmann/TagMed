"""

"""

import tkinter as tk 
from tkinter import ttk, filedialog, messagebox, scrolledtext, Text

from config_handler import ConfigHandler
from annotation_loader import AnnotationLoader
from gallery_navigator import GalleryNavigator

class AnnotationTool:
    """
    
    """

    def __init__(self, root):

        self.root = root
        self.root.title("TagMed - Annotation Tool")
        self.root.geometry("1400x900")
        
        # Create menubar
        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.helpmenu = tk.Menu(self.menubar, tearoff=0)

        # Add cascades to menubar
        self.menubar.add_cascade(label="File", menu=self.filemenu) # first cascade
        self.filemenu.add_command(label="Add Image Folder", command=self.select_image_folder)
        self.filemenu.add_command(label="Add Annotation Database", command=self.select_anno_table)
        self.filemenu.add_command(label="Add Medical Reports", command=self.select_medical_reports)
        self.filemenu.add_command(label="Manage Classes", command=self.open_class_manager)
        self.filemenu.add_command(label="Close", command=exit)

        self.menubar.add_cascade(label="Other", menu=self.helpmenu) # second cascade
        self.helpmenu.add_command(label="Help", command=self.open_help_window)
        self.helpmenu.add_command(label="Imprint", command=self.open_imprint_window)


        # add menubar to window
        self.root.config(menu=self.menubar)

        # Create frame for refresh button, Progressbar and Searchterm
        bt_frame = tk.Frame(root, height=50)  # Set a specific height and width
        bt_frame.pack(padx=10, pady=5, fill="x", expand=False)
        bt_frame.pack_propagate(False)  # Prevent frame from resizing to fit its children

        self.full_data = [] # here the data in the table is stored

        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(bt_frame, textvariable=self.search_var, width=40)
        search_entry.pack(side="left", padx=10, pady=5, fill="x", expand=False)
        search_entry.bind("<KeyRelease>", self.filter_treeview)

        # StringVar for button text
        self.bt_refresh_text = tk.StringVar(value="Load Data")

        # Button with textvariable
        self.bt_refresh = tk.Button(bt_frame, textvariable=self.bt_refresh_text, command=self.populate_tree)
        self.bt_refresh.pack(side="right", pady=5, padx=10)  # Place button on the right side of the frame

        # Frame for Treeview and Scrollbar
        frame = tk.Frame(root)
        frame.pack(padx=10, fill="both", expand=True)

        # Create Treeview
        columns = ("ID", "Exams",  "annotated_images", "total_img","progress")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)
        for col in columns: # Connects sort column function with table
            self.tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(_col, False))


        # Define column headings
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
             

        # Add Scrollbar 
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # pack
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # add double click event
        self.tree.bind("<Double-1>", self.on_double_click)

        # bind close event 
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # load configurated options
        self.config = ConfigHandler()
        self.class_list = self.config.get("class_list", [])
        self.selected_image_folder = self.config.get("selected_image_folder", "")
        self.selected_anno_table_file = self.config.get("selected_anno_table_file", "")
        self.selected_medical_report_file = self.config.get("selected_medical_report_file", "")

        self.config.set("class_list", self.class_list)
        self.config.set("selected_image_folder", self.selected_image_folder)
        self.config.set("selected_anno_table_file", self.selected_anno_table_file)
        self.config.set("selected_medical_report_file", self.selected_medical_report_file)
        self.config.save()

        # Initialize AnnotationLoader
        self.annotation_loader = AnnotationLoader()


    def populate_tree(self):
        """
        Fills the table with patient data and shows Progress Pop-up.
        """
    
        popup = tk.Toplevel()
        popup.geometry("300x100")
        tk.Label(popup, text="Load Data...").pack(pady=10)
        progress = ttk.Progressbar(popup, orient="horizontal", length=250, mode="determinate")
        progress.pack(pady=5)
        percent_label = tk.Label(popup, text="0%")
        percent_label.pack()
        #self.update()

        # Delete content to prevent double entries
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.full_data.clear()

        all_patients = self.annotation_loader.available_patients()

        total = len(all_patients)
        print(all_patients)

        for i, patient_id in enumerate(all_patients, 1):
            exams, total_files = self.annotation_loader.count_exams_and_files(patient_id)
            annotated_images = self.annotation_loader.count_annotated_images(patient_id)

            try:
                progress_ratio = annotated_images / total_files
            except ZeroDivisionError:
                progress_ratio = 0.0

            #progress_ratio = round(progress_ratio)
            percent_value = int(progress_ratio * 100)
            bar_length = 20  # lenghts of progressbar
            filled_length = int(bar_length * progress_ratio)

            progress_bar = "█" * filled_length + "░" * (bar_length - filled_length)
            progress_display = f"{progress_bar} {percent_value}%"

            row = (patient_id, exams, annotated_images, total_files, progress_display)
            self.tree.insert("", "end", values=row)
            self.full_data.append(row) # Append data to full_data - buffer for search function

            # Update Progress
            progress["value"] = (i / total) * 100
            percent_label.config(text=f"{int((i / total) * 100)}%")
            popup.update_idletasks()

        popup.destroy()
        self.bt_refresh_text.set("Refresh")


    def treeview_sort_column(self, col, reverse):
        """
        Sorts the table with double clicking on a specific column.
        """
        data = [(self.tree.set(child, col), child) for child in self.tree.get_children('')]

        try:
            data.sort(key=lambda t: float(t[0]) if t[0] != "NN" else float("-inf"), reverse=reverse)
        except ValueError:
            data.sort(key=lambda t: t[0], reverse=reverse)

        for index, (val, child) in enumerate(data):
            self.tree.move(child, '', index)

        self.tree.heading(col, command=lambda: self.treeview_sort_column(col, not reverse))

    def filter_treeview(self, event=None):
        search_term = self.search_var.get().lower()

        # Clear treeview
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Show inly relevant data
        for row in self.full_data:
            #if any(search_term in str(cell).lower() for cell in row):
            if search_term in str(row[0]).lower(): # only searching for patient ID
                self.tree.insert("", "end", values=row)

    def on_double_click(self, patient_id ,event=None):
        """
        On double click on a patient the patient window will be opend and you can start annotate.
        """
        item = self.tree.selection() 
        if item:
            patient_id = self.tree.item(item, "values")[0] # gets Patient ID from selected Row
            self.patient_id = patient_id

            navigator = GalleryNavigator(self.root)
            navigator.open_patient_window(patient_id)

    def select_image_folder(self):
        folder_path = filedialog.askdirectory(title="Choose a directory with your image data")
        if folder_path:
            self.selected_image_folder = folder_path # safe 
            self.config.set("selected_image_folder", folder_path)
            print(f"[INFO] File directory: {folder_path}")
        

    def select_medical_reports(self):
        medical_report_file = filedialog.askopenfilename(title="Choose a file with your medical reports",
                        filetypes=[("table files", "*.csv *.xlsx")])
        if medical_report_file:
            self.selected_medical_report_file = medical_report_file
            self.config.set("selected_medical_report_file", medical_report_file)
            print(f"[INFO] Medical Reports: {medical_report_file}")

    def select_anno_table(self):
        anno_table_file = filedialog.askopenfilename(title="Choose a annotation database",
                        filetypes=[("table files", "*.csv *.xlsx")])
        if anno_table_file:
            self.selected_anno_table_file = anno_table_file
            self.config.set("selected_anno_table_file", anno_table_file)
            print(f"[INFO] Annotation Database: {anno_table_file}")
        else:
            messagebox.showinfo("Info", "If you do not select a file, a new file will be created for you.")
            self.annotation_loader.create_default_anno_table()



    def open_class_manager(self):
        class_window = tk.Toplevel()
        class_window.title("Manage Classes")

        self.class_list = getattr(self, "class_list", [])  # make sure that the list exists

        # listbox to show classes
        listbox = tk.Listbox(class_window, height=8, width=30)
        listbox.pack(padx=10, pady=5)

        # funkction to refresh classes
        def update_listbox():
            listbox.delete(0, tk.END)
            for cls in self.class_list:
                listbox.insert(tk.END, cls)

        update_listbox()

        # adding classes
        entry = tk.Entry(class_window, width=25)
        entry.pack(padx=10)

        def add_class():
            new_class = entry.get().strip()
            if new_class and new_class not in self.class_list:
                self.class_list.append(new_class)
                update_listbox()
                entry.delete(0, tk.END)

        def delete_class():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                del self.class_list[index]
                update_listbox()

        def on_close_class_window():
            print(f"[INFO] Following classes selected: {self.class_list}")
            class_window.destroy()

        # Buttons
        button_frame = tk.Frame(class_window)
        button_frame.pack(pady=5)

        # bind close event 
        class_window.protocol("WM_DELETE_WINDOW", on_close_class_window)

        tk.Button(button_frame, text="Add", command=add_class).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Delete", command=delete_class).grid(row=0, column=1, padx=5)

    def on_close(self):
        self.config.save()
        self.root.destroy()


    def open_help_window():
        """
        This function is used to open and display a new window to display a help text or a readMe.
        Is called up via the menu bar on the start page using the Help button.
        """
        # Create new Toplevel window 
        help_window = tk.Toplevel()
        help_window.geometry("500x500")
        help_window.title("Help")

        # Scrollable Textbox to display Help
        text_box = scrolledtext.ScrolledText(help_window, wrap="word", width=60, height=20)
        text_box.pack(padx=10, pady=10, fill="both", expand=True)

        # Configure fonts as tags
        text_box.tag_config("title", font=("Arial", 14, "bold"))
        text_box.tag_config("bold", font=("Arial", 12, "bold"))

        
        # Define Text elements
        text_box.insert("end", "Welcome to Help\n", "title")
        text_box.insert("end", "\nGeneral:\n", "bold")
        text_box.insert("end", "This tool was designed to annotate sensitive images that are difficult to interpret.\n\n")
        text_box.insert("end", "For further information please contact us\n")
        text_box.insert("end", "Contact details can be found in the imprint.\n")

        # Set the text field to not editable
        text_box.config(state="disabled")

        # create close button at the bottom center
        button_close = tk.Button(help_window, text="Close", command=help_window.destroy)
        button_close.pack(pady=10)

    def open_imprint_window():
        """
        This function is used to open and display a new window to display the imprint.
        Is called up via the menu bar on the start page using the Imprint button.
        """

        # Create new Toplevel window 
        imprint_window = tk.Toplevel()
        imprint_window.geometry("400x400")
        imprint_window.title("Imprint")

        # Textbox 
        text_box = Text(imprint_window, wrap="word", width=60, height=10)
        text_box.pack(padx=10, pady=10, fill="both", expand=True)

        # Configure fonts as tags
        text_box.tag_config("title", font=("Arial", 12, "bold"))
        text_box.tag_config("bold", font=("Arial", 10, "bold"))
        text_box.tag_config("normal", font=("Arial", 10))
        
        # Define Text elements
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

        # Set the text field to not editable
        text_box.config(state="disabled")

        # create close button at the bottom center
        button_close = tk.Button(imprint_window, text="Close", command=imprint_window.destroy)
        button_close.pack(pady=10)
    
    def bind_listbox_events(self, listbox):
        listbox.bind("<<ListboxSelect>>", lambda event: self.on_select(event, listbox))
        listbox.bind("<MouseWheel>", lambda event: self.on_mousewheel(event, listbox))

    def on_select(self, event, listbox):
        selected = listbox.curselection()
        if selected:
            print("Ausgewählt:", listbox.get(selected[0]))
            self.patient_id = self.patient_listbox.get(selected[0])

    def on_mousewheel(self, event, listbox):
        listbox.yview_scroll(-1 * (event.delta // 120), "units")
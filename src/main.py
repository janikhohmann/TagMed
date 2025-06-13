import tkinter as tk

from annotation_tool import AnnotationTool 



# main function
def main():
    """
    Calls the Annotation Tool
    """

    root = tk.Tk()
    app = AnnotationTool(root)
    # Set the Theme
    root.tk.call("source", "azure.tcl")
    root.tk.call("set_theme", "light")

    root.geometry("1400x900") 

    root.mainloop()    
 

if __name__ == "__main__":
    main()

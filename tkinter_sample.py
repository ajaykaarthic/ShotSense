import tkinter as tk
from tkinter import Label, filedialog
from tkinter import messagebox
import predict_shot

root = tk.Tk()
root.geometry("450x300")

def show_message(title, msg):
    messagebox.showinfo(title, msg)

def open_file():
    filename = filedialog.askopenfilename(
        title='Select a video file',
        filetypes=[('Video files', '*.mp4 *.avi *.mov')],
        initialdir='/',
        multiple=False
    )
    if filename:
        print(filename)
        show_message("file path", f"file path: {filename}")
        try:
            predict_shot.main(filename)
            show_message("result", "Success: check the results folder")
        except:
            print("error occured")
            show_message("error",  "error occured")

        # You can perform further actions with the selected file here

button = tk.Button(
    root,
    text='Select a video file',
    command=open_file,
    padx=10,
    pady=5,
    fg='white',
    bg='#4CAF50',  # Green color
    font=('Arial', 12),
    borderwidth=2,
    relief='raised',  # Raised border
)
button.pack(pady=60)

root.mainloop()

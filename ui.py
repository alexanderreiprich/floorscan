import tkinter as tk
from tkinter import filedialog, Button, Label, Canvas, Frame, Scrollbar, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import os
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FloorScan")
        
        main_frame = Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.label = Label(control_frame, text="Bild auswählen")
        self.label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.upload_button = Button(control_frame, text="Bilder auswählen", command=self.open_images)
        self.upload_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.segment_button = Button(control_frame, text="Segmentieren", command=self.segment_images)
        self.segment_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_button = Button(control_frame, text="Ergebnisse speichern", command=self.save_results)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.save_button.config(state=tk.DISABLED)  # To avoid save before segmenting
        
        # Canvas
        canvas_frame = Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        v_scrollbar = Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        h_scrollbar = Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas = Canvas(canvas_frame, width=800, height=600, 
                             yscrollcommand=v_scrollbar.set,
                             xscrollcommand=h_scrollbar.set)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Images
        self.image_frame = Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW)
        
        self.image_paths = []
        self.image_objects = []  # Instances of images
        self.processed_images = []  # Processed Images (Thumbnails)
        self.full_processed_images = []  # Process Images (Full resolution)
        
        self.model_yolo = YOLO("./model/yolo.pt")
        self.model_sam, self.predictor = self.load_sam()
        
        self.image_frame.bind("<Configure>", self.on_frame_configure)
                
    # Setup SAM
    def load_sam(self):
        model_type = "vit_l"
        checkpoint_path = "./model/sam.pth"

        sam = sam_model_registry[model_type]()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        sam.load_state_dict(checkpoint, strict=False)

        predictor = SamPredictor(sam)
        
        return sam, predictor
    
    # Refresh scrolling region
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # Open Prompt to select images from device
    def open_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;")])
        if file_paths:
            # remove previous images
            for widget in self.image_frame.winfo_children():
                widget.destroy()
            
            self.image_paths = list(file_paths)
            self.image_objects = []
            self.processed_images = []
            self.full_processed_images = []
            
            # Set save button to disabled
            self.save_button.config(state=tk.DISABLED)
            
            self.display_images()
            self.label.config(text=f"{len(self.image_paths)} Bilder geladen")
    
    # Draw images
    def display_images(self):
        if not self.image_paths:
            return
        
        # Calculate canvas size
        num_images = len(self.image_paths)
        cols = min(2, num_images)  # 2 columns max
        
        # Max thumbnail (!) size
        max_width = 400
        max_height = 300
        
        # Load selected images
        for i, path in enumerate(self.image_paths):
            row = i // cols
            col = i % cols
            
            img_container = Frame(self.image_frame)
            img_container.grid(row=row, column=col, padx=10, pady=10)
            
            # Display Filename
            filename = os.path.basename(path)
            img_label = Label(img_container, text=filename)
            img_label.pack()
            
            # Load singular image and draw thumbnail
            try:
                img = Image.open(path)
                img.thumbnail((max_width, max_height))
                photo_img = ImageTk.PhotoImage(img)

								# Draw image on canvas
                img_canvas = Canvas(img_container, width=photo_img.width(), height=photo_img.height())
                img_canvas.pack()
                img_canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
                
                # Add to array, avoids garbage collection
                self.image_objects.append((photo_img, img_canvas))
                
            except Exception as e:
                error_label = Label(img_container, text=f"Fehler beim Laden: {str(e)}", fg="red")
                error_label.pack()
    
    def segment_images(self):
        if not self.image_paths:
            self.label.config(text="Bitte zuerst Bilder laden!")
            return
        
        # Refresh status
        self.label.config(text="Verarbeite Bilder...")
        self.root.update()
        
        # Remove widgets
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        # Calculate canvas size (like in display_images)
        num_images = len(self.image_paths)
        cols = min(2, num_images)
        
        max_width = 400
        max_height = 300
        
        # Initialize saving
        self.processed_images = []
        self.full_processed_images = []
        
        # Go through every image and segment it
        for i, img_path in enumerate(self.image_paths):
            row = i // cols
            col = i % cols
            
            # Container for image
            img_container = Frame(self.image_frame)
            img_container.grid(row=row, column=col, padx=10, pady=10)
            
            # Label (filename)
            filename = os.path.basename(img_path)
            img_label = Label(img_container, text=filename)
            img_label.pack()
            
            try:
                img_path = os.path.normpath(img_path)
                
                if not os.path.exists(img_path):
                    error_label = Label(img_container, text=f"Datei nicht gefunden: {img_path}", fg="red")
                    error_label.pack()
                    continue

                # Load image
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img)
                
                # Use YOLO for detection of elements
                results = self.model_yolo(img_path)
        
                # Setup SAM
                self.predictor.set_image(img_array)
        
                # Transparent image for mask drawing
                mask_image = Image.new("RGBA", img.size, (0, 0, 0, 0))
                
                # Iterate through results
                for result in results:
                    for box in result.boxes.data:
                        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                        label = result.names[int(cls)]

                        if label in ["window", "door"]:
                            # Feed SAM the bounding box of the detected element
                            masks, scores, logits = self.predictor.predict(
                                box=np.array([x1, y1, x2, y2]),
                                multimask_output=True
                            )
                            mask = masks[0].astype(np.uint8) * 255
                            mask_pil = Image.fromarray(mask).convert("L")
                            color = (0, 0, 255, 128) if label == "window" else (255, 0, 0, 128)
            
                            # Display colored masks
                            mask_image.paste(Image.new("RGBA", mask_pil.size, color), mask=mask_pil)
        
                # Combine original and mask and save
                combined = Image.alpha_composite(img.convert("RGBA"), mask_image.convert("RGBA"))
                
                self.full_processed_images.append((filename, combined.copy()))
                
                # Create thumbnail from combined image
                combined.thumbnail((max_width, max_height))
                combined_tk = ImageTk.PhotoImage(combined)
                
                # Canvas for combined image
                img_canvas = Canvas(img_container, width=combined_tk.width(), height=combined_tk.height())
                img_canvas.pack()
                img_canvas.create_image(0, 0, anchor=tk.NW, image=combined_tk)
                
                # Save reference
                self.processed_images.append(combined_tk)
                
            except Exception as e:
                error_label = Label(img_container, text=f"Fehler bei der Verarbeitung: {str(e)}", fg="red")
                error_label.pack()
        
        # Activate "save" button after processing and update UI
        if self.full_processed_images:
            self.save_button.config(state=tk.NORMAL)
            
        self.label.config(text="Segmentierung abgeschlossen")
    
    def save_results(self):
           
        # Create output dir
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create subfolder with session metadata
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_dir, f"session_{timestamp}")
        os.makedirs(session_dir)
            
        # Save all images (full resolution)
        saved_count = 0
        for filename, img in self.full_processed_images:
            try:
                base_name, ext = os.path.splitext(filename)
                output_path = os.path.join(session_dir, f"{base_name}_segmented.png")

                img.save(output_path)
                saved_count += 1
                
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Speichern von {filename}: {str(e)}")
                
        # Update UI
        
        self.label.config(text=f"Segmentierte Bilder gespeichert in: {session_dir}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
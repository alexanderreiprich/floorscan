import tkinter as tk
from tkinter import filedialog, Button, Label, Canvas, Entry
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from transformers import SamModel
from clip_interrogator import Interrogator, Config
from ultralytics import YOLO
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageSegmentationApp:
		def __init__(self, root):
				self.root = root
				self.root.title("Image Segmentation App")
				
				self.canvas = Canvas(root, width=800, height=600)
				self.canvas.pack()
				
				self.label = Label(root, text="Bild auswählen")
				self.label.pack()
				
				self.upload_button = Button(root, text="Bild auswählen", command=self.open_images)
				self.upload_button.pack()		
				
				self.segment_button = Button(root, text="Segmentieren", command=self.segment_image)
				self.segment_button.pack()
				
				self.image_paths = []
				self.images = []
				self.model_yolo = YOLO("./runs/detect/train5/weights/best.pt")
				self.model_sam, self.predictor = self.load_sam()
				
		def load_sam(self):
				model_type = "vit_b"
				checkpoint_path = "./model/sam_vit_b_01ec64.pth"
		
				sam = sam_model_registry[model_type]()
				checkpoint = torch.load(checkpoint_path, map_location="cpu")
				sam.load_state_dict(checkpoint, strict=False)
		
				predictor = SamPredictor(sam)
				
				return sam, predictor

		def open_images(self):
			file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
			if file_paths:
					for path in file_paths:
							if len(self.image_paths) < 4:  # Maximal 4 Bilder erlauben
									self.image_paths.append(path)
					
					self.label.config(text=f"{len(self.image_paths)} Bilder geladen")
				
		# def display_image(self, path):
		# 		img = Image.open(path)
		# 		img.thumbnail((800, 600))
		# 		self.image = ImageTk.PhotoImage(img)
		# 		self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
		
		def segment_image(self):
				if not self.image_paths:
					self.label.config(text="Bitte zuerst Bilder laden!")
					return
		 
				for img_path in self.image_paths:
						img_path = os.path.normpath(img_path)  # Pfad normalisieren

						if not os.path.exists(img_path):
								print(f"Datei nicht gefunden: {img_path}")
								self.label.config(text=f"Fehler: Datei nicht gefunden\n{img_path}")
								continue

						try:
								img = Image.open(img_path).convert("RGB")
						except Exception as e:
								print(f"Fehler beim Öffnen des Bildes: {e}")
								self.label.config(text=f"Fehler beim Öffnen des Bildes\n{str(e)}")
								continue
						img_path = os.path.normpath(img_path)
						img = Image.open(img_path).convert("RGB")
						img_array = np.array(img)
						
						results = self.model_yolo(img_path)
			
						# Bild in den Predictor laden
						self.predictor.set_image(img_array)
			
						mask_image = Image.new("RGBA", img.size, (0, 0, 0, 0))
						draw = ImageDraw.Draw(mask_image)
						
						for result in results:
							for box in result.boxes.data:
								x1, y1, x2, y2, conf, cls = box.cpu().numpy()
								label = result.names[int(cls)]
								
								if label in ["window", "door"]:
									masks, scores, logits = self.predictor.predict(
										box=np.array([x1, y1, x2, y2]),
										multimask_output=True
									)
									mask = masks[0].astype(np.uint8) * 255
									mask_pil = Image.fromarray(mask).convert("L")
									color = (0, 0, 255, 128) if label == "window" else (255, 0, 0, 128)
					
									mask_image.paste(Image.new("RGBA", mask_pil.size, color), mask=mask_pil)
				
						combined = Image.alpha_composite(img.convert("RGBA"), mask_image.convert("RGBA"))
						combined.thumbnail((800, 600))
						combined_tk = ImageTk.PhotoImage(combined)
				
						self.canvas.create_image(0, 0, anchor=tk.NW, image=combined_tk)
						self.images.append(combined_tk)
			
				self.label.config(text="Segmentierung abgeschlossen")

if __name__ == "__main__":
		root = tk.Tk()
		app = ImageSegmentationApp(root)
		root.mainloop()

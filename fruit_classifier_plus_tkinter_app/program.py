import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models\FRUITCNN1.h5')

# Define the class labels (if applicable)
class_labels = ['apple fruit', 'banana fruit', 'cherry fruit', 'chickoo fruit', 'grapes fruit', 'kiwi fruit', 'mango fruit', 'orange fruit', 'strawberry fruit']

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.ANTIALIAS)  # Resize the image
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo  # Keep a reference to the image to prevent garbage collection

        # Perform prediction using the loaded model
        img_array = np.array(image.resize((224, 224))) / 255.0  # Preprocess the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        prediction_label.configure(text=f"Predicted class: {predicted_class}")
        confidence_percentage = (np.max(prediction)*100)
        output.configure(text=f"{prediction}")
        if confidence_percentage > 90.0 and confidence_percentage<=96:
            confidence_label.configure(text=f"Confidence: {confidence_percentage} %")

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Create a button to open an image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display the prediction
prediction_label = tk.Label(root, text="Predicted class: ")
confidence_label = tk.Label(root,text="Confidence: " )
prediction_label.pack()
confidence_label.pack()

output = tk.Label(root, text="output")
output.pack()

# Run the Tkinter event loop
root.mainloop()

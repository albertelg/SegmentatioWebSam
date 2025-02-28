# pip install flask

from flask import Flask, redirect,url_for,render_template,request
from app import process_image
from traditionalAlgorithms import waterShed
from traditionalAlgorithms import kmeans
from traditionalAlgorithms import canny_edge

app=Flask(__name__)

method_map = {
    'Watershed': waterShed,
    'K-means': kmeans,
    'SAM - Meta': process_image,
    'Canny': canny_edge
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=["GET", "POST"])
def submit():
    
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    image_path = f"./uploads/{file.filename}"
    file.save(image_path)  # Save the image
    
    processed_images = []
    
    if request.method == "POST":
        selected_methods = request.form.getlist("segmentation_methods")
        dev = 'cuda' if request.form.get("device") == 'on' else 'cpu'
    
    for method in selected_methods:
        if method in method_map:
            if method == "SAM - Meta":
                processed_image_path = process_image(image_path, dev)
            else:
                processed_image_path = method_map[method](image_path)
                
            processed_images.append({
                "path": processed_image_path,
                "label": method  # Store method name
            })

    return render_template('segmentat.html', processed_images=processed_images)


if __name__=='__main__':
    app.run(debug=True)
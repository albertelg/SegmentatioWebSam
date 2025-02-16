# pip install flask

from flask import Flask, redirect,url_for,render_template,request
from app import process_image

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    image_path = f"./uploads/{file.filename}"
    file.save(image_path)  # Save the image

    # Call the function from process_image.py
    processed_image_path = process_image(image_path)

    return render_template('segmentat.html', processed_image='static/'+processed_image_path)


if __name__=='__main__':
    app.run(debug=True)
import os
import io
import cv2
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from main import Detection

app = Flask(__name__)
app.config['INPUT_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['INPUT_FOLDER']):
    os.makedirs(app.config['INPUT_FOLDER'], exist_ok=True)

detection = Detection()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    file = request.files.get('image')

    if not file or file.filename == '':
        return 'No file uploaded', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['INPUT_FOLDER'], filename)
    file.save(file_path)

    # Process the image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img, _ = detection.predict(img)
    output = Image.fromarray(processed_img)

    # Convert the image to PNG and return it
    buf = io.BytesIO()
    output.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)

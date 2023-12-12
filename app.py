from flask import Flask, render_template, request, flash, jsonify
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flashing messages

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return jsonify({'status': 'success', 'message': 'File uploaded successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)

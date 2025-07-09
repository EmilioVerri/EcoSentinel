
from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, DecimalRangeField, IntegerRangeField
from os import listdir
from os.path import join
# Importa la funzione secure_filename per gestire i nomi dei file
from werkzeug.utils import secure_filename

from wtforms.validators import InputRequired, NumberRange
import os

# Importa il modulo cv2 per lavorare con immagini e video utilizzando OpenCV
import cv2

# Importa la funzione video_detection dal modulo YOLO_Video
from YOLO_Video import video_detection

app = Flask(__name__)

app.config['SECRET_KEY'] = 'emilio'
app.config['UPLOAD_FOLDER'] = 'file/videos'
class UploadFileForm(FlaskForm):  # Definisce una classe per il form per caricare file
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x=''):  # Definisce una funzione per generare i frame del video
    yolo_output = video_detection(path_x)

    for detection_ in yolo_output:  # Itera sui frame elaborati
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()  # Converte il frame elaborato in byte
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_web(path_x):  # Definisce una funzione per generare i frame della webcam
    yolo_output = video_detection(path_x)  # Esegue la funzione di rilevazione degli oggetti YOLO sulla webcam
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()  # Cancella la sessione corrente
    return render_template('index.html')



@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()  # Cancella lasessione corrente
    return render_template('webcam.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()  # Crea un'istanza del form di caricamento file
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))

        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename))

        # Mostra il template video.html con il form
    return render_template('video.html', form=form)

@app.route('/video')
def video():
    # Restituisce i frame del video
    return Response(generate_frames(path_x=session.get('video_path', None)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    # Restituisce i frame della webcam
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/databaseimmagini')
def database_images():
  """
  This function retrieves a list of image filenames from the "images" folder
  and renders the "databaseimmagini.html" template with the list.
  """
  image_filenames = [f for f in listdir("static/images") if f.endswith((".jpg", ".png", ".jpeg"))]
  return render_template('databaseimmagini.html', image_filenames=image_filenames)



@app.route('/database', methods=['GET', 'POST'])
def database():
    session.clear()
    rows = []
    try:
        with open('log.txt', 'r') as file:
            for line in file:
                values = line.strip().split(', ')
                if len(values) == 2:
                    rows.append(values)
    except FileNotFoundError:
        pass
    return render_template('database.html', rows=rows)

@app.route('/delete_logs', methods=['POST'])
def delete_logs():
    try:
        with open('log.txt', 'w') as file:
            file.write('')
        return '', 204
    except FileNotFoundError:
        return '', 404
    except Exception as e:
        print('Errore durante la cancellazione dei log:', e)
        return '', 500  



from os import listdir, remove
from os.path import join

@app.route('/delete_images', methods=['POST'])
def delete_images():
  try:
    images_folder = join(app.static_folder, 'images')

    files = listdir(images_folder)

    for filename in files:
      if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'): 
        file_path = join(images_folder, filename)
        remove(file_path)

    return '', 204

  except FileNotFoundError:

    return '', 404

  except Exception as e:

    print('Errore durante la cancellazione delle immagini:', e)
    return '', 500





if __name__ == "__main__":
    app.run(debug=True)

import cv2
import math
import datetime
import os
from ultralytics import YOLO


def video_detection(path_x):
    # Assegna il valore del parametro path_x alla variabile video_capture
    video_capture = path_x

    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Carica il modello YOLO preaddestrato per rilevare fuoco e fumo
    model = YOLO("../Modelli-YOLO/fireSmoke.pt")

    classNames = ['evidente_fumata', 'importante_fumata', 'lieve_fumata']

    # Crea la cartella "images" se non esiste già
    if not os.path.exists("static/images"):
        os.makedirs("static/images")

    # File di log per salvare le classi trovate e le date/orari
    log_file_path = "log.txt"

    while True:
        success, img = cap.read()  # Legge un frame dal video

        results = model(img, stream=True)  # Esegue il rilevamento degli oggetti sul frame

        for r in results:  # Itera sui risultati della rilevazione
            boxes = r.boxes  # Ottiene le coordinate dei rettangoli delimitatori

            for box in boxes:  # Itera sui rettangoli delimitatori
                x1, y1, x2, y2 = box.xyxy[0]  # Ottiene le coordinate del rettangolo

                # Converte le coordinate in valori interi
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Disegna un rettangolo intorno all'oggetto rilevato
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                conf = math.ceil((box.conf[0] * 100)) / 100  # Calcola la confidenza

                # Ottiene l'indice della classe dell'oggetto
                cls = int(box.cls[0])

                class_name = classNames[cls]  # Ottiene il nome della classe

                # Crea la label senza spazi, usando underscore per la separazione
                label = f"{class_name}_{conf}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

                # Calcola la dimensione del testo per la label
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                # Disegna un rettangolo per la label
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)

                # Scrive la label sull'immagine
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # Salva l'immagine nella cartella "images" con nome senza spazi
                image_path = os.path.join("static/images", label + ".jpg")
                cv2.imwrite(image_path, img)

                # Aggiungi la classe e la data/ora al file di log
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"{class_name}, {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")

        # Restituisce l'immagine elaborata
        yield img

    cv2.destroyAllWindows()

import json
import os
import shutil
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, send_from_directory, request, Response

app = Flask(__name__)

current_frame = None

birds_choices_file_path = 'static/data/birds_choices.json'
birds_photos_stats_file_path = 'static/data/birds_photos_stats.json'
birds_visit_hours_file_path = 'static/data/birds_visit_hours.json'
wrong_detections_file_path = 'static/data/wrong_detections.json'
stream_mode_file_path = 'static/data/stream_mode.json'

website_pic_path = '/home/piotrek/Desktop/website/static/'

save_path = '/home/piotrek/Desktop/website/detections/'
detekcje_save_path = save_path + 'detekcje'
detekcje_no_box_save_path = save_path + 'detekcje_no_box'
detekcje_bbox_txt_save_path = save_path + 'detekcje_bbox_txt'
dobra_detekcja_save_path = save_path + 'dobra_detekcja'
zla_etykieta_save_path = save_path + 'zla_etykieta'
brak_ptaka_save_path = save_path + 'brak_ptaka'
sprawdzone_detekcje_save_path = save_path + 'sprawdzone_detekcje'

def map_to_bird_name(number):
    bird_names = {0: "sikorka", 1: "rudzik", 2: "sojka"}
    bird_name = bird_names.get(number, "inne")
    return bird_name

def wybierz_zdjecie(folder_path):
    pliki_w_folderze = [plik for plik in os.listdir(folder_path) if plik.endswith('.jpg')]

    if not pliki_w_folderze:
        return None

    pic = min(pliki_w_folderze, key=lambda x: int(os.path.splitext(x)[0]))
    
    return pic


def load_next_pic():
    global detekcje_save_path
    global next_pic_path
    next_pic = wybierz_zdjecie(detekcje_save_path)

    if next_pic:
        next_pic_path = os.path.join(detekcje_save_path, next_pic)
        shutil.copy(next_pic_path, os.path.join(website_pic_path, 'next_pic.jpg'))

load_next_pic()

def load_data(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        return None

def save_data(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def calculate_pie_chart_data():
    global birds_photos_stats
    global wrong_detections
    global pie_chart_data
    birds_photos_stats = load_data(birds_photos_stats_file_path) or {
        "sikorka": 0,
        "rudzik": 0,
        "sojka": 0
    }

    pie_chart_data = {bird: birds_photos_stats[bird] - wrong_detections[bird] for bird in birds_photos_stats}

def add_wrong_detections(label_path):
    global wrong_detections
    with open(label_path, "r") as file:
        lines = file.readlines()
    for l in lines:
        bird_number = int(l.split(' ')[0])
        wrong_detections[map_to_bird_name(bird_number)] += 1
    with open(wrong_detections_file_path, "w") as file:
        json.dump(wrong_detections, file)

def change_mode_to_stream():
    with open(stream_mode_file_path, 'w') as file:
        file.write('{"stream": 1}')

def change_mode_to_default():
    with open(stream_mode_file_path, 'w') as file:
        file.write('{"stream": 0}')
            
    
birds_choices = {}
birds_photos_stats = {}
birds_visit_hours = {}
wrong_detections = {}
def load_jsons():
    global birds_choices
    global birds_photos_stats
    global birds_visit_hours
    global wrong_detections
    birds_choices = load_data(birds_choices_file_path) or {
    "sikorka": 0,
    "rudzik": 0,
    "sojka": 1
}

    birds_photos_stats = load_data(birds_photos_stats_file_path) or {
    "sikorka": 0,
    "rudzik": 0,
    "sojka": 0
}

    birds_visit_hours = load_data(birds_visit_hours_file_path) or {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
    "10": 0,
    "11": 0,
    "12": 0,
    "13": 0,
    "14": 0,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 0,
    "19": 0,
    "20": 0,
    "21": 0,
    "22": 0,
    "23": 0
}

    wrong_detections = load_data(wrong_detections_file_path) or {
    "sikorka": 0,
    "rudzik": 0,
    "sojka": 0
}

load_jsons()
pie_chart_data = {}
calculate_pie_chart_data()

def generate_frames():
    while True:
        frame = get_frame()
        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def home():
    change_mode_to_default()
    load_jsons()
    calculate_pie_chart_data()
    load_next_pic()
    return render_template('home.html', birds_choices=birds_choices, pie_chart_data=pie_chart_data, birds_visit_hours=birds_visit_hours)

@app.route('/update_status/<bird>', methods=['POST'])
def update_status(bird):
    if bird in birds_choices:
        birds_choices[bird] = 1 - birds_choices[bird]
        save_data(birds_choices_file_path, birds_choices)
        return jsonify({"status": "success", "bird": bird, "value": birds_choices[bird]})
    else:
        return jsonify({"status": "error", "message": "Invalid bird name"})

@app.route('/<action>', methods=['POST'])
def przenies_zdjecie(action):
    current_pic = wybierz_zdjecie(detekcje_save_path)

    if not current_pic:
        return jsonify({"status": "error", "message": "Brak dostępnych zdjęć."})

    if action == 'dobrze':
        dest_path = dobra_detekcja_save_path
    elif action == 'zlaEtykieta':
        dest_path = zla_etykieta_save_path
    elif action == 'brakPtaka':
        dest_path = brak_ptaka_save_path
    else:
        return jsonify({"status": "error", "message": "Nieznana akcja."})

    try:
        current_pic_path = os.path.join(detekcje_save_path, current_pic)
        
        shutil.move(current_pic_path, os.path.join(sprawdzone_detekcje_save_path, current_pic))
        
        pic_name = current_pic
        pic_path = os.path.join(detekcje_no_box_save_path, pic_name)
        shutil.move(pic_path, os.path.join(dest_path, pic_name))
        
        label_name = current_pic.split('.')[0] + '.txt'
        
        label_path = os.path.join(detekcje_bbox_txt_save_path, label_name)

        if action != 'dobrze':
            add_wrong_detections(label_path)
            calculate_pie_chart_data()

        if action == 'dobrze' or action == 'zlaEtykieta':
            shutil.move(label_path, os.path.join(dest_path, label_name))
        else:
            shutil.move(label_path, os.path.join(dest_path + '/labels', label_name))
        
        next_pic = wybierz_zdjecie(detekcje_save_path)

        if next_pic:
            next_pic_path = os.path.join(detekcje_save_path, next_pic)
            shutil.copy(next_pic_path, os.path.join(website_pic_path, 'next_pic.jpg'))

            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Brak kolejnych zdjęć do wyboru."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/static/<path:filename>')
def serve_static(filename):
    response = send_from_directory(app.static_folder, filename)
    response.cache_control.max_age = 0
    return response
    
@app.route('/get_pie_chart_data')
def get_pie_chart_data():
    global pie_chart_data
    calculate_pie_chart_data()
    
    return jsonify(pie_chart_data)

@app.route("/stream")
def stream():
    change_mode_to_stream()
    return render_template('stream.html')
    
@app.route('/update_frame', methods=['POST'])
def update_frame():
    global current_frame
    image_file = request.files['image']
    image_array = np.frombuffer(image_file.read(), np.uint8)
    current_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return jsonify({'message': 'Frame received'})

@app.route('/get_frame')
def get_frame():
    ret, jpeg = cv2.imencode('.jpg', current_frame)
    return jpeg.tobytes()

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

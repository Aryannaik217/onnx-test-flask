from flask import Flask, render_template, request, jsonify , send_file
import cv2
import base64
import numpy as np
import detect_boxes
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    result_img , count , (conf1 , conf2) , avg_conf = process_image(img)
    _, buffer = cv2.imencode('.jpg', result_img)
    result_img_str = base64.b64encode(buffer).decode('utf-8')
    if count != 0:
        range_str = "" + str(conf1) + " to " + str(conf2) + " %"
        avg_conf = str(avg_conf) + "%"
    else:
        range_str = "NA"
        avg_conf = "NA"
    return jsonify({'result': result_img_str , 'Count':count , 'Confidence_range':range_str , 'confidence_avg':avg_conf})

@app.route('/capture', methods=['POST'])
def capture():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    result_img , count , (conf1 , conf2) , avg_conf = process_image(img)
    _, buffer = cv2.imencode('.jpg', result_img)
    result_img_str = base64.b64encode(buffer).decode('utf-8')
    if count != 0:
        range_str = "" + str(conf1) + " to " + str(conf2) + " %"
        avg_conf = str(avg_conf) + "%"
    else:
        range_str = "NA"
        avg_conf = "NA"
   
    return jsonify({'result': result_img_str , 'Count':count , 'Confidence_range':range_str , 'confidence_avg':avg_conf})



@app.route('/sw.js')
def serve_sw():
    return send_file('sw.js', mimetype='application/javascript')

@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')

def process_image(img):
    try:
        image_path = img
        onnx_model_path = "./PCM5-IOU0.1.onnx" 
        classes = ['pipe']
 
        result_image , count , result_tuple , average_confidence = detect_boxes.predict_with_onnx(image_path, onnx_model_path, classes)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        print("Count:" , count)
        print("Range:" , result_tuple)
        print("Average confidence:" , average_confidence)
        return result_image , count , result_tuple , average_confidence
    
    except Exception:
        print('Error Occurred')
        traceback.print_exc()
        return img , 0 , (0,0) , 0

if __name__ == '__main__':
    app.run(debug=True)

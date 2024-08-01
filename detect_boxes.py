import onnxruntime as ort
import cv2
import numpy as np
 
def load_onnx_model(onnx_model_path):
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path)
    return session
 
def preprocess_image_onnx(image_path, img_size):
    # Load and preprocess the image
    img = image_path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = img_resized / 255.0  # Normalize to [0, 1]
    img_resized = np.transpose(img_resized, (2, 0, 1))  # Change from HWC to CHW
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_resized = img_resized.astype(np.float32)
    return img_resized
 
def draw_boxes(image, detections, classes , original_w , original_h):
    print('Drawing bounding boxes')
    for detection in detections:
        print([round(i,2) for i in detection])
        x1, y1, x2, y2, cls, conf = detection[1:]
        x1 = int(x1 * original_w / 640)                 
        y1 = int(y1 * original_h / 640)                 
        x2 = int(x2 * original_w / 640)
        y2 = int(y2 * original_h / 640)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        label = f"{classes[int(cls)]}: {conf:.2f}"
        print(conf)
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(image, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, label_y - 7), font, font_scale, (0, 0, 0), thickness)
    return image
 
 
def predict_with_onnx(image_path, onnx_model_path, classes, img_size=640, conf_thresh=0.02, iou_thresh=0.2):
    # Load ONNX model
    session = load_onnx_model(onnx_model_path)
    print("model loaded")
    # Preprocess image
    img_input = preprocess_image_onnx(image_path, img_size)
 
    print("Preprocessing completed")
    print("image:" , img_input.shape)

    #img_input = cv2.imread("d:/Video Analtycis/Pothole Detection/yolov7-main/dataset/train/images/img-1_jpg.rf.04766deb9036fc43721c26f431c3eb3d.jpg")
    #print("CV2 Image is being used" , img_input.shape)
    # Perform inference
    outputs = session.run(None, {"images":img_input})
    detections = np.array(outputs[0])

    print("detections found in the image" , detections[:6])
    #print([round(i,2) for i in outputs[0][0]])
    # Filter detections by confidence threshold
    detections = detections[detections[:, 6] >= conf_thresh]
 
    
    #print("Model prediction done")
    # Apply non-maximum suppression
    keep = cv2.dnn.NMSBoxes(
        bboxes=detections[:, 1:5].tolist(),
        scores=detections[:, 6].tolist(),
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh
    )
    detections = detections[keep]
 
    # Load image for drawing boxes
    image = image_path
    print(image.shape)
    original_w , original_h , _ = image.shape
    #image = cv2.resize(image , (640,640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    # Draw bounding boxes
    image_with_boxes = draw_boxes(image, detections, classes , original_w , original_h)
 
    return image_with_boxes
 

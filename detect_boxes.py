import onnxruntime as ort
import cv2
import numpy as np


def resize_and_pad_image(image, target_width, target_height):
    (h, w) = image.shape[:2]
    aspect_ratio = w / h
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    top_padding = (target_height - new_height) // 2
    bottom_padding = target_height - new_height - top_padding
    left_padding = (target_width - new_width) // 2
    right_padding = target_width - new_width - left_padding

    padded_image = cv2.copyMakeBorder(
        resized_image,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return padded_image, (new_width, new_height), (left_padding, top_padding)


def load_onnx_model(onnx_model_path):
    
    session = ort.InferenceSession(onnx_model_path)
    return session
 
def preprocess_image_onnx(image_path, img_size):
   
    img = image_path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized , (n_w , n_h) , (l_p , t_p) = resize_and_pad_image(img, img_size ,img_size)
    img_resized = img_resized / 255.0  
    img_resized = np.transpose(img_resized, (2, 0, 1))  
    img_resized = np.expand_dims(img_resized, axis=0) 
    img_resized = img_resized.astype(np.float32)
    return img_resized
 
def draw_boxes(image, detections, classes , original_w , original_h , new_w , new_h, left_padding , top_padding):
  
    count = 0
    confidences = []
    resized_predictions = []
    for detection in detections:
        
        x1, y1, x2, y2, cls, conf = detection[1:]

        confidences.append(conf)

        # Adjusting Bounding boxes so that it can fit to the original image
        x1 = x1 - left_padding
        y1 = y1 - top_padding
        x2 = x2 - left_padding
        y2 = y2 - top_padding

        x1 = int(x1 * (original_w / new_w)) 
        y1 = int(y1 * (original_h / new_h))              
        x2 = int(x2 * (original_w / new_w)) 
        y2 = int(y2 * (original_h / new_h)) 

        resized_predictions.append([0 , x1 , y1 , x2 , y2 , int(cls) , int(conf*100)])
 
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,255), 1)
    
        count += 1
    if count == 0:
        range_tuple = (0,0)    
        average_confidence = 0.0
    else:
        range_tuple = (min(confidences) , max(confidences))
        average_confidence = sum(confidences) / len(confidences)

    return image , count , range_tuple , average_confidence
 
 
def predict_with_onnx(image_path, onnx_model_path, classes, img_size=640, conf_thresh=0.30, iou_thresh=0.1):
    # Load ONNX model
    session = load_onnx_model(onnx_model_path)
    print("model loaded")

    # Preprocess image
    img_input = preprocess_image_onnx(image_path, img_size)
    print("Preprocessing completed")
    print("image:" , img_input.shape)
 
    # Perform inference
    outputs = session.run(None, {"images":img_input})
    detections = np.array(outputs[0])
    
    # Filter detections by confidence threshold
    detections = detections[detections[:, 6] >= conf_thresh]
 
    # Load image for drawing boxes
    image = image_path
    original_h , original_w , _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2, (n_w , n_h) , (l_p , t_p) = resize_and_pad_image(image , 640 , 640)

    # Draw bounding boxes
    image_with_boxes, count , r_tuple , average_confidence = draw_boxes(image, detections, classes , original_w , original_h , n_w , n_h , l_p , t_p)
    
    range_tuple = [round(r_tuple[0] , 2) , round(r_tuple[1] , 2)]
    range_tuple = tuple(range_tuple)
    average_confidence = round(average_confidence , 2)
    return image_with_boxes,count , range_tuple , average_confidence

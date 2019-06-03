import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import urllib
import urllib.request as urllib2
import base64
import ast
import math

sys.path.append("D:/Projects/Object-Detection-API-Project/models-master/models-master/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util


def distance(a, b, landmark):
    x = landmark[a]["x"] - landmark[b]["x"]
    y = landmark[a]["y"] - landmark[b]["y"]
    dis = math.sqrt(x**2 + y**2)
    return dis


def eye_detection(landmark):
    dis1 = distance(31, 37, landmark)
    dis2 = distance(33, 35, landmark)
    dis3 = distance(30, 34, landmark)

    dis4 = distance(14, 20, landmark)
    dis5 = distance(16, 18, landmark)
    dis6 = distance(13, 17, landmark)

    # dis7 = distance(32, 36, landmark)
    # dis8 = distance(15, 19, landmark)

    average = ((dis1 + dis2)/(2*dis3) + (dis4 + dis5)/(2*dis6)) / 2
    # average2 = (dis7/dis3 + dis8/dis6)/2
    if average < 0.2:
        return "close"
    else:
        return "open"


def mouth_detect(landmark):
    length = landmark[58]["x"]-landmark[62]["x"]
    height = landmark[67]["y"]-landmark[70]["y"]
    judge = height/length
    if judge < 0.2:
        return "close"
    else:
        return "open"


def test_warning(count):
    if count > 10:
        return True
    else:
        return False


def draw_warning(im):
    pts = np.array([[100, 70], [60, 139], [140, 139]], np.int32)
    cv2.polylines(im, [pts], True, (0, 0, 0), thickness=5)
    pts = np.array([[100, 75], [65, 134], [135, 134]], np.int32)
    cv2.fillPoly(im, [pts], color=(0, 255, 255))
    cv2.circle(im, center=(100, 95), radius=3, color=(0, 0, 0), thickness=5)
    pts = np.array([[94, 95], [100, 119], [106, 95]], np.int32)
    cv2.fillPoly(im, [pts], color=(0, 0, 0))
    cv2.circle(im, center=(100, 127), radius=2, color=(0, 0, 0), thickness=5)


# VIDEO_NAME = 'FUNNY_CATS .mp4'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH, 'model_v1.0_summary', 'output_inference_graph', 'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH, 'annotations', 'labelmap.pbtxt')

# PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# video = cv2.VideoCapture(PATH_TO_VIDEO)
video = cv2.VideoCapture(0)

request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
access_token = '24.67eab68b342df8ddddd780a72c87d383.2592000.1561540502.282335-16084761'
request_url = request_url + "?access_token=" + access_token

params = {}
params['image_type'] = 'BASE64'
params['face_field'] = 'age,beauty,landmark,landmark72'

eye_outputStr = 'state of eye:'
mouth_outputStr = 'state of mouth:'
eye_state = ''
mouth_state = ''

eye_count = 0
mouth_count = 0

while(video.isOpened()):
    ret, frame = video.read()

    frame_expanded = np.expand_dims(frame, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # ------------------------------------------------------------------------------------------------------------------
    # 检测开始
    img = frame
    img = cv2.imencode('.jpg', img)[1].tostring()
    img = base64.b64encode(img)
    img = img.decode('utf-8')

    params['image'] = img

    request = urllib2.Request(url=request_url, data=urllib.parse.urlencode(params).encode(encoding='UTF8'))
    request.add_header('Content-Type', 'application/json')

    response = urllib2.urlopen(request)
    content = response.read()
    content = content.decode('utf-8')
    flag = content.find('SUCCESS')

    if flag != -1:
        content = ast.literal_eval(content)
        result = content['result']
        face_list = result['face_list']
        face_list = face_list[0]
        landmark72 = face_list['landmark72']

        eye_state = eye_detection(landmark72)
        mouth_state = mouth_detect(landmark72)

        eyes_flag = eye_state.find('open')
        mouth_flag = mouth_state.find('open')

        if eyes_flag == -1:
            eye_count = eye_count + 1
        else:
            eye_count = 0

        if mouth_flag != -1:
            mouth_count = mouth_count + 1
        else:
            mouth_count = 0

        if test_warning(eye_count) | test_warning(mouth_count):
            draw_warning(frame)

        for item in landmark72:
            point = (int(item['x']), int(item['y']))
            cv2.circle(frame, point, radius=1, color=(255, 0, 0), thickness=4)
    # 检测结束
    # ------------------------------------------------------------------------------------------------------------------

    cv2.putText(frame, eye_outputStr + eye_state, (250, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 25), 2)
    cv2.putText(frame, mouth_outputStr + mouth_state, (250, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 25), 2)

    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

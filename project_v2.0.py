import urllib
import urllib.request as urllib2
import base64
import ast
import cv2
import math
import numpy as np


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


cap = cv2.VideoCapture(0)
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

while True:
    ret, image = cap.read()

    # ------------------------------------------------------------------------------------------------------------------
    # 检测开始
    img = image
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
            draw_warning(image)

        for item in landmark72:
            point = (int(item['x']), int(item['y']))
            cv2.circle(image, point, radius=1, color=(255, 0, 0), thickness=4)
    # 检测结束
    # ------------------------------------------------------------------------------------------------------------------

    cv2.putText(image, eye_outputStr + eye_state, (250, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 25), 2)
    cv2.putText(image, mouth_outputStr + mouth_state, (250, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 25), 2)
    cv2.imshow('image', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('quit!')
        break

cap.release()
cv2.destroyAllWindows()

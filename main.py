from time import time
from threading import Thread
import math
import cv2
import mediapipe as mp
import numpy as np
import requests


def draw_neck_info(img: np.array, ang, color: tuple, x: int, y: int):
    cv2.putText(img, f"Neck angle: {ang}", (20, 140), cv2.FONT_HERSHEY_PLAIN,
                left_cart_font_size, color, 2)

    cv2.putText(img, f"{ang}", (x, y), cv2.FONT_HERSHEY_PLAIN,
                left_cart_font_size, color, 2)


def draw_torso_info(img: np.array, ang, color: tuple, x: int, y: int):
    cv2.putText(img, f"Torso angle: {ang}", (20, 180), cv2.FONT_HERSHEY_PLAIN,
                left_cart_font_size, color, 2)

    cv2.putText(img, f"{ang}", (x, y), cv2.FONT_HERSHEY_PLAIN,
                left_cart_font_size, color, 2)


def send_msg_telegram(msg, token, chat_id):
    send_text = 'https://api.telegram.org/bot' + token + '/sendMessage?chat_id=' + chat_id + \
                '&parse_mode=Markdown&text=' + msg

    response = requests.get(send_text)
    if response.status_code != 200:
        print("Request error")
        print(response.text)


def angle2pt(a, b):
    change_inx = b[0] - a[0]
    change_iny = b[1] - a[1]
    ang = math.degrees(math.atan2(change_iny, change_inx))
    return ang * -1 if ang < 0 else ang


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_lm_list(frame: np.array, draw=True) -> list:
    landmark_list = []

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False  # lepsza wydajnosc
    results = pose.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        if draw:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for lm in results.pose_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_list.append((x, y, lm.z, lm.visibility))

    return [frame, landmark_list]


cap = cv2.VideoCapture(r"Videos/vid2.mp4")
# cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

bot_token = "5338689548:AAGRkUlgH5fzaTztZGjjnSbytA95n5_1rFQ"
chat_id = "1103873621"

p_time = 0
left_cart_font_size = 1.5
orange = (0, 100, 200)
red = (0, 0, 230)
green = (0, 230, 0)
radius = 8
final_img = False
good_posture_frames = 0
bad_posture_frames = 0
alert_time = 960
alert_sent = False
neck_angle_thr = 120
torso_angle_thr = 93

#  1s = 16 klatek tak +-
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(f"Res2.mp4", fourcc, 20, (1280, 720))
while True:
    posture_info = set()

    success, img = cap.read()
    if success is False:
        break
    img = cv2.resize(img, (1280, 720))
    img = cv2.flip(img, 1)
    overlay = img.copy()
    h, w, c = img.shape

    img, lm_list = get_lm_list(img, draw=False)

    c_time = time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time

    if len(lm_list) != 0:
        l_shoulder = lm_list[11]
        r_shoulder = lm_list[12]

        l_hip = lm_list[23]
        r_hip = lm_list[24]

        l_ear = lm_list[7]
        r_ear = lm_list[8]

        cv2.circle(img, r_shoulder[:2], radius, orange, -1)
        cv2.circle(img, l_shoulder[:2], radius, orange, -1)

        shoulder_dist = int(get_distance(l_shoulder, r_shoulder))
        cv2.rectangle(overlay, (15, 15), (310, 200), (0, 0, 0), -1)
        cv2.rectangle(overlay, (900, 15), (1265, 180), (0, 0, 0), -1)
        alpha = 0.4  # Transparency factor.

        # Following line overlays transparent rectangle
        # over the image
        final_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.rectangle(final_img, (10, 10), (315, 205), (255, 255, 255), 2)
        cv2.rectangle(final_img, (895, 10), (1270, 185), (255, 255, 255), 2)

        if shoulder_dist < 100:
            cv2.line(final_img, l_shoulder[:2], r_shoulder[:2], green, 2)
            cv2.putText(final_img, f"Shoulders aligned ({shoulder_dist})", (20, 100), cv2.FONT_HERSHEY_PLAIN, left_cart_font_size, green, 2)
            if r_shoulder[2] > l_shoulder[2]:
                torso_angle = int(angle2pt(l_hip, l_shoulder))
                neck_angle = int(angle2pt(l_shoulder, l_ear))

                if neck_angle < neck_angle_thr:
                    draw_neck_info(final_img, neck_angle, green, l_shoulder[0] + 10, l_shoulder[1])
                    cv2.line(final_img, l_shoulder[:2], l_ear[:2], green, 2)
                    posture_info.add(1)
                else:
                    draw_neck_info(final_img, neck_angle, red, l_shoulder[0] + 10, l_shoulder[1])
                    cv2.line(final_img, l_shoulder[:2], l_ear[:2], red, 2)
                    posture_info.add(0)

                if torso_angle < torso_angle_thr:
                    draw_torso_info(final_img, torso_angle, green, l_hip[0] + 10, l_hip[1])
                    cv2.line(final_img, l_shoulder[:2], l_hip[:2], green, 2)
                    posture_info.add(1)
                else:
                    draw_torso_info(final_img, torso_angle, red, l_hip[0] + 10, l_hip[1])
                    cv2.line(final_img, l_shoulder[:2], l_hip[:2], red, 2)
                    posture_info.add(0)

                cv2.circle(final_img, l_hip[:2], radius, orange, -1)
                cv2.circle(final_img, l_ear[:2], radius, orange, -1)
            else:
                torso_angle = int(angle2pt(r_shoulder, r_hip))
                neck_angle = int(angle2pt(r_ear, r_shoulder))

                if neck_angle < neck_angle_thr:
                    draw_neck_info(final_img, neck_angle, green, r_shoulder[0] + 10, r_shoulder[1])
                    cv2.line(final_img, r_ear[:2], r_shoulder[:2], green, 2)
                    posture_info.add(1)
                else:
                    draw_neck_info(final_img, neck_angle, red, r_shoulder[0] + 10, r_shoulder[1])
                    cv2.line(final_img, r_ear[:2], r_shoulder[:2], red, 2)
                    posture_info.add(0)

                if torso_angle < torso_angle_thr:
                    draw_torso_info(final_img, torso_angle, green, r_hip[0] + 10, r_hip[1])
                    cv2.line(final_img, r_hip[:2], r_shoulder[:2], green, 2)
                    posture_info.add(1)
                else:
                    draw_torso_info(final_img, torso_angle, red, r_hip[0] + 10, r_hip[1])
                    cv2.line(final_img, r_hip[:2], r_shoulder[:2], red, 2)
                    posture_info.add(0)

                cv2.circle(final_img, r_ear[:2], radius, orange, -1)
                cv2.circle(final_img, r_hip[:2], radius, orange, -1)

            if all(posture_info):
                good_posture_frames += 1
                bad_posture_frames = 0
                alert_sent = False

                if good_posture_frames == 1:
                    good_pos_start_t = time()

                good_posture_time = round(time() - good_pos_start_t, 1)

                cv2.putText(final_img, f"Good posture", (910, 60), cv2.FONT_HERSHEY_PLAIN, 3, green, 3)
                cv2.putText(final_img, f"For: {good_posture_time}s", (990, 110), cv2.FONT_HERSHEY_PLAIN, 2.4, green, 3)
            else:
                good_posture_frames = 0
                bad_posture_frames += 1
                if bad_posture_frames == 1:
                    bad_pos_start_t = time()

                bad_posture_time = round(time() - bad_pos_start_t, 1)

                if bad_posture_frames == alert_time:
                    Thread(target=send_msg_telegram, args=("Nie garb siem", bot_token, chat_id)).start()
                    alert_sent = True

                cv2.putText(final_img, f"Bad posture", (930, 60), cv2.FONT_HERSHEY_PLAIN, 3, red, 3)
                cv2.putText(final_img, f"For: {bad_posture_time}s", (990, 110), cv2.FONT_HERSHEY_PLAIN, 2.4, red, 3)
                if alert_sent:
                    cv2.putText(final_img, f"Alert already sent", (915, 160), cv2.FONT_HERSHEY_PLAIN, 2.2, red, 3)

        else:
            cv2.putText(final_img, f"Shoulders not aligned ({shoulder_dist})", (20, 100), cv2.FONT_HERSHEY_PLAIN,
                        left_cart_font_size, red, 2)
            cv2.line(final_img, l_shoulder[:2], r_shoulder[:2], red, 2)

        cv2.putText(final_img, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 200, 200), 2)

        out.write(final_img)
        cv2.imshow("Res", final_img)

    else:
        cv2.putText(img, f"No human detected", (400, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, red, 2)

        cv2.putText(img, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 200, 200), 2)
        cv2.imshow("Res", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()

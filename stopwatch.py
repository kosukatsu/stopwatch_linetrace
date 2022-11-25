import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import sys
from time import sleep, perf_counter
import yaml
import requests
import json

import cv2
import numpy as np

STATE_CAPTURE_BACKGROUND = 1
STATE_SET_STARTLINE = 2
STATE_MEASURE_TIME = 3


def main(conf):
    if conf["debug"] == True:
        camera = cv2.VideoCapture(conf["debug_params"]["video_path"])
        background_img = cv2.imread(conf["debug_params"]["back_path"])
        state = STATE_SET_STARTLINE
        print("Set start line and Press {} key".format(conf["key_params"]["set_start"]))
    else:
        camera = cv2.VideoCapture(conf["camera_id"])  # カメラCh.(ここでは0)を指定

        if camera.isOpened() == True:
            print("camera connected!")

        else:
            print("This webcamera id is not available")
            return
        background_img = None
        state = STATE_CAPTURE_BACKGROUND
        print(
            "Press {} key to capture background image.".format(
                conf["key_params"]["capture_background"]
            )
        )

    main_window_name = "camera"
    diff_img_windows_name = "diff"

    start_line_left_top = None
    start_line_right_bottom = None

    show_ratio = conf["show_rescale_ratio"]

    detect_flag = False

    def mouse_callbuck(event, x, y, flags, params):
        nonlocal start_line_left_top, start_line_right_bottom
        if conf["debug"] == True:
            print(event, x, y, flags, params)
        if state == STATE_SET_STARTLINE:
            if event == cv2.EVENT_LBUTTONDOWN:
                start_line_left_top = (int(x / show_ratio), int(y / show_ratio))
            elif event == cv2.EVENT_LBUTTONUP:
                start_line_right_bottom = (int(x / show_ratio), int(y / show_ratio))

    cv2.namedWindow(main_window_name)
    cv2.setMouseCallback(main_window_name, mouse_callbuck)

    prev_time = None
    x0, x1, y0, y1 = None, None, None, None
    is_setted_start_line = False

    try:
        while True:
            ret, frame = camera.read()  # フレームを取得
            draw_frame = None
            k = cv2.waitKey(1) & 0xFF
            # if k == 27:
            #     break
            if state == STATE_CAPTURE_BACKGROUND:
                draw_frame = frame
                if k == ord(conf["key_params"]["capture_background"]):
                    background_img = frame
                    # camera_b = camera.get(cv2.CAP_PROP_EXPOSURE)
                    # camera_f = camera.get(cv2.CAP_PROP_FOCUS)
                    # camera_wb = camera.get(cv2.CAP_PROP_WB_TEMPERATURE)
                    # camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                    # camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    # camera.set(cv2.CAP_PROP_AUTO_WB, 0)
                    # camera.set(cv2.CAP_PROP_EXPOSURE, camera_b)
                    # camera.set(cv2.CAP_PROP_FOCUS, camera_f)
                    # camera.set(cv2.CAP_PROP_WB_TEMPERATURE, camera_wb)
                    state = STATE_SET_STARTLINE
                    print(
                        "Set start line and Press {} key".format(
                            conf["key_params"]["set_start"]
                        )
                    )

            elif state == STATE_SET_STARTLINE:
                assert background_img is not None
                draw_frame = frame
                if (start_line_left_top is not None) and (
                    start_line_right_bottom is not None
                ):
                    y0 = start_line_left_top[1]
                    y1 = start_line_right_bottom[1]
                    x0 = start_line_left_top[0]
                    x1 = start_line_right_bottom[0]
                    if (y0 < y1) and (x0 < x1):
                        is_setted_start_line = True
                    else:
                        is_setted_start_line = False
                        print("error start line")
                if is_setted_start_line:
                    cv2.rectangle(
                        draw_frame,
                        start_line_left_top,
                        start_line_right_bottom,
                        color=(255, 0, 0),
                        thickness=5,
                    )
                    if k == ord(conf["key_params"]["set_start"]):
                        cv2.namedWindow(diff_img_windows_name)
                        state = STATE_MEASURE_TIME
                        print("start measuring time")

            elif state == STATE_MEASURE_TIME:
                assert background_img is not None
                assert start_line_left_top is not None
                assert start_line_right_bottom is not None
                assert y0 < y1
                assert x0 < x1
                diff = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(
                    int
                ) - cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY).astype(int)
                diff_mono = np.abs(diff).astype("uint8")
                _, binary = cv2.threshold(
                    diff_mono, conf["threthold"]["binary"], 255, cv2.THRESH_BINARY
                )
                start_line_binary = binary[y0:y1, x0:x1] / 255
                start_line_across_ratio = (
                    np.sum(start_line_binary) / ((y1 - y0) * (x1 - x0)) * 100
                )
                is_crossing = start_line_across_ratio > conf["threthold"]["start_line"]
                if (detect_flag == False) and (is_crossing == True):
                    detect_flag = True
                    if prev_time is None:
                        prev_time = perf_counter()
                        print("first detecting")
                    else:
                        now_time = perf_counter()
                        rap_time = now_time - prev_time
                        prev_time = now_time
                        is_valid_result = rap_time > conf["threthold"]["invalid_result"]
                        rap_message = "{:.3f}".format(rap_time)
                        # 短い記録を無視しない or 短い記録でないとき，結果を出力
                        if (conf["ignore_tiny_rap"] == False) or is_valid_result:
                            if conf["send_message"] == True:
                                try:
                                    requests.post(
                                        conf["web_hook_url"],
                                        data=json.dumps(
                                            {
                                                "text": rap_message,
                                                "icon_emoji": ":stopwatch:",
                                                "username": "stopwatch",
                                            }
                                        ),
                                    )
                                except:
                                    print("cannot send message!")
                        print(rap_message)
                elif (detect_flag == True) and (is_crossing == True):
                    prev_time = perf_counter()
                elif (detect_flag == True) and (is_crossing == False):
                    detect_flag = False
                draw_frame = frame
                # if True:
                if prev_time is not None:
                    cv2.putText(
                        draw_frame,
                        "time:{:.3f}s".format(perf_counter() - prev_time),
                        (0, 25),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    draw_frame,
                    "detect_flag:{}".format(detect_flag),
                    (0, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    draw_frame,
                    "is_crossing:{}".format(is_crossing),
                    (0, 75),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    draw_frame,
                    "across_ratio:{}".format(start_line_across_ratio),
                    (0, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    binary,
                    start_line_left_top,
                    start_line_right_bottom,
                    color=(255, 0, 0),
                    thickness=2,
                )
                cv2.rectangle(
                    draw_frame,
                    start_line_left_top,
                    start_line_right_bottom,
                    color=(255, 0, 0),
                    thickness=2,
                )
                cv2.imshow(
                    diff_img_windows_name,
                    cv2.resize(binary, dsize=None, fx=show_ratio, fy=show_ratio),
                )
            else:
                print("state error")
                break
            if conf["debug"] is True:
                print(start_line_left_top, start_line_right_bottom)
                print(state)
            cv2.imshow(
                main_window_name,
                cv2.resize(draw_frame, dsize=None, fx=show_ratio, fy=show_ratio),
            )  # フレームを画面に表示
    finally:
        # 撮影用オブジェクトとウィンドウの解放
        print("release")
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    with open("conf.yml", "rb") as f:
        conf = yaml.safe_load(f)
    main(conf)

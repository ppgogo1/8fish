#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import *
import math
import random

import cv2
from PIL import Image
import numpy as np

import sys
import argparse
import subprocess

### RXD TXD
import time
import serial
import Queue
import threading
import xml.dom.minidom
import uuid
from hashlib import md5


### args for camera
WINDOW_NAME = 'CameraDemo'


def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [960]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [540]',
                        default=1080, type=int)

    parser.add_argument('--demo', dest='demo',
                        help='demo',
                        default=True, type=bool)

    parser.add_argument('--id', dest='id',
                        help='id',
                        default=0, type=int)

    parser.add_argument('--weightbase', dest='weightbase',
                        help='baseweight',
                        default=0, type=float)
    args = parser.parse_args()
    return args


# display the pic after detecting.
def showPicResult(image):
    img = cv2.imread(image)
    cv2.imwrite(out_img, img)
    for i in range(len(r)):
        x1 = r[i][2][0] - r[i][2][2] / 2
        y1 = r[i][2][1] - r[i][2][3] / 2
        x2 = r[i][2][0] + r[i][2][2] / 2
        y2 = r[i][2][1] + r[i][2][3] / 2
        im = cv2.imread(out_img)
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        # This is a method that works well.
        cv2.imwrite(out_img, im)
    cv2.imshow('yolo_image_detector', cv2.imread(out_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

'''class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]'''

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("../lib/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    # im = load_image(image, 0, 0)
    # print(im)
    #im = load_image("/work/darknet/img/bianyu001.jpg", 0, 0)#???????????????????????????
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im) #darknet.so????????????
    # print(predict_image(net, im))
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    res = []
    for j in range(num):
        for i in range(meta.classes):
            #print j, i, num
            if dets[j].prob[i] :
                if dets[j].prob[i] > 0.0:
                    # print("probiblity")
                    # print(dets[j].prob[i])
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE


def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image


def open_cam_onboard(width, height):
    print('open_cam_onboard')
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)960, height=(int)540, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')


import Tkinter as tk  # ??????Tkinter??????????????????
from Tkinter import *
#import Tkinter.messagebox
import Tkinter
import time
from PIL import Image, ImageTk

import threading

luck_new = []
bianhao = "003-EWRC-JK-0011"


def WBcheck(label):
    wfile = open("whitelist1.txt", "r")
    line = wfile.readline()
    # line=wline[:-1]
    while line:
        line = wfile.readline()
        for i in range(len(label)):
            if line.split(',')[0] == label[i]:
                # print('ok')
                end = 'fou'
                wfile.close()
                return end
    bfile = open("blacklist1.txt", "r")
    bline = bfile.readline()
    # print(bline)
    # bline=bline[:-1]
    while bline:
        if bfile.readline() == '':
            for i in range(len(label)):

                if bline.split(',')[0] == label[i]:
                    end = 'shi'
                    bfile.close()
                    return end
        else:
            bline = bfile.readline()
            for i in range(len(label)):

                if bline.split(',')[0] == label[i]:
                    end = 'shi'
                    bfile.close()
                    return end
    return "weizhi"


#import RPi.GPIO as GPIO
import time
import thread
import cv2
import socket


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


stack1 = Stack()

serverip = '192.168.1.107'
serverport = 8000


def heart(serverip, serport):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((serverip, serverport))
    try:
        while True:
            time.sleep(10)
            s.send('on'.encode())
            print("2222")
    finally:
        s.close()


dataip = '192.168.1.107'
dataport = 8086

picip = '192.168.1.107'
picport = 8088
import numpy as np

picflag = False


def pic(picip, picport, picture):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((picip, picport))

    time.sleep(1)
    # frame = cv2.imread("2.jpg")
    frame = picture
    # print(frame)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(imgencode)
    stringData = data.tostring()
    s.send(str.encode(str(len(stringData)).ljust(16)))
    s.send(stringData)
    # receive = sock.recv(1024)
    # if len(receive):print(str(receive,encoding='utf-8'))
    s.close()
    picflag = True


import json

'''


def dataconfirm(idnum,label,weight,SFYX,picture,rec_flag):
    #print("12214564")
    while 1:
        #time.sleep(0.2)
        #print(stack1.pop)
        #stack1.push("pin1")
        #print(stack1.is_empty)
        #print( str(stack1.pop))
        global stack1
        if (not stack1.is_empty()) and str(stack1.pop()) == "pin1"and rec_flag==True:
            #print(stack1.is_empty)
            #stack1=Stack()
            while not stack1.is_empty():
                stack1.pop()
                print("1232144")

            print("wdqwdq")
            thread.start_new_thread(pic,(picip,picport,picture))
            s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            s.connect((dataip,dataport))


            try:

            #time.sleep(10)


                info={}
                info["yongzhong"]=str(label)
                info["zhongliang"]=str(weight)
                info["shifouweiyouxiaoyuhuo"]=str(SFYX)

                info["shifoudaiding"]="fou"


                data={}
               #data["diaotai"]=str(idnum)
               # data[str(idnum)]=info
                data["code"]=str(idnum)
                data["yuzhong"]=str(label)[0:-3]
                data["zhongliang"]=str(weight).split("k")[0]
                data["shifouyouxiaoyuhuo"]=str(SFYX)
                data["shifoudaiding"]="0"

                jsonStr = json.dumps(data)  
                s.send(jsonStr.encode())
                break
                #print("2222")
            finally:
                s.close()

        if (not stack1.is_empty()) and str(stack1.pop()) == "pin2"and rec_flag==True:
            #print(stack1.is_empty)
            #stack1=Stack()
            while not stack1.is_empty():
                stack1.pop()
                print("1232144")
            print("wdqwdq")
            thread.start_new_thread(pic,(picip,picport,picture))
            s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            s.connect((dataip,dataport))


            try:

            #time.sleep(10)


                info={}
                info["yuzhong"]=str(label)
                info["zhongliang"]=str(weight)
                info["shifouweiyouxiaoyuhuo"]=str(SFYX)

                info["shifoudaiding"]="shi"


                data={}
                #data["diaotai"]=str(idnum)
                data[str(idnum)]=info

                jsonStr = json.dumps(data)
                s.send(jsonStr.encode())
                break
                #print("2222")
            finally:
                s.close()



'''

fishnum = {"yongyu": 10001, "lianyu": 10002, "caoyu": 10003, "qingyu": 10004, "liyu": 10005, "bianyu": 10006,
           "nianyu": 10007, "luyu": 10008}

import requests
import base64
import cv2

from io import BufferedReader, BytesIO


def picandjsonhttp(img, request_body):
    # path = '/home/hp/Desktop/test.jpg'
    # img=cv2.imread(path)	#?????????????????????????????????????????????????????????????????????
    ret, img_encode = cv2.imencode('.jpg', img)
    str_encode = img_encode.tostring()  # ???array????????????????????????
    f4 = BytesIO(str_encode)  # ?????????_io.BytesIO??????
    f4.name = str(time.time()) + '.jpg'  # ????????????
    f5 = BufferedReader(f4)  # ?????????_io.BufferedReader??????
    res = {"file": f5}

    headers = {}
    headers['Content-Type'] = 'application/json'

    url = 'http://121.229.42.223:89/api/fish/add'
    # print(request_body)
    # request_body = {"code": "10003","zhongliang":2.9,"yuzhong":"caoyu","changdu":7.23}

    r0 = requests.post(url=url, files=res, data=request_body)
    print(r0.text)


def pichttp(i):
    # i=cv2.imread("/home/hp/Desktop/test.jpg")
    img = cv2.imencode(".jpg", i)[1].tobytes()
    img = base64.b64encode(img).decode()
    image = []
    image.append(img)
    res = {"image": image}

    # r = requests.post("http://192.168.1.107:8085/",data=res)
    r = requests.post("http://121.229.42.223:89/api/fish/listPlatform", data=res)

    print(r.text)


def jsonhttp(request_body):
    # url = 'http://192.168.1.107:8086/'
    url = 'http://121.229.42.223:89/api/fish/listPlatform'

    r0 = requests.post(url=url, json=request_body)
    print(r0.text)


def dataconfirm():
    global luck_new
    while 1:
        if len(luck_new) > 0:
            picture = luck_new.pop(0)
            # time.sleep(5)

            # thread.start_new_thread(pic,(picip,picport,picture[1]))
            # thread.start_new_thread(pichttp,(picture[1],))

            picflag = False
            ##s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            ##s.connect((dataip,dataport))
            global bianhao
            # thread.start_new_thread(dataconfirm,(did,str([r[i][0] for i in range(len(r))])[1:-1],luck.split(":")[-1].split("\n")[0],WBdata,detdata[-1][1],True))

            bianhao = "003-EWRC-JK-0011"

            try:

                data = {}
                # data["diaotai"]=str(idnum)
                # data[str(idnum)]=info
                # data["code"]=fishnum[str(picture[0][0][0])[0:-1]]
                data["code"] = str(bianhao)
                # data["yuzhong"]=str(picture[0][0][0])[0:-2]
                # print(str(picture[0][0][0])[0:-2])
                if str(picture[0][0][0])[0:-1] in fishnum.keys():
                    data["yuzhong"] = fishnum[str(picture[0][0][0])[0:-1]]
                    # data["yuzhong"]=str(picture[0][0][0])[0:-1]

                # data["zhongliang"]=float(picture[2].split(":")[-1].split("\n")[0].split("k")[0])youyong
                print(time.ctime(time.time()))

                #                w,h=picture[1].shape

                # print(picture[1].shape)
                a = picture[0][0][-1][2] - picture[0][0][-1][0]
                b = picture[0][0][-1][3] - picture[0][0][-1][1]
                # if a>b:
                #   data["changdu"]=a*3.375

                # else:
                #    data["changdu"]=b*2.53125 

                jsonStr = json.dumps(data)

                # jsonhttp(jsonStr)
                # picandjsonhttp(picture[1],data)*****youyong
                ##s.send(jsonStr.encode())
            finally:
                ##s.close()
                pass


# thread.start_new_thread(dataconfirm,())


# Pin Definitions
input_pin = 15  # 15,16,18 BCM pin 18, BOARD pin 12
pin_y = 16
pin_f = 18


def pin1(stack1):
    prev_value = None
    # print("1132135")
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin

    # GPIO.setup(pin_y, GPIO.IN)  # set pin as an input pin
    # GPIO.setup(pin_f, GPIO.IN)  # set pin as an input pin
    print("start")
    try:
        while True:
            value = GPIO.input(input_pin)
            if value != prev_value:
                # if True:
                time.sleep(0.08)
                if value == GPIO.HIGH:
                    value_str = "HIGH"
                else:
                    value_str = "LOW"
                    while not stack1.is_empty():
                        stack1.pop()
                    stack1.push("pin1")
                    break
                    # print(stack1.pop())

                # print("Value read from pin {} : {}".format(input_pin,value_str))

                # prev_value = value
            # time.sleep(1)
    finally:
        GPIO.cleanup()


def pin2(stack1):
    # print("ediwqedj")
    prev_value = None

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi

    GPIO.setup(pin_y, GPIO.IN)  # set pin as an input pin
    # GPIO.setup(pin_f, GPIO.IN)  # set pin as an input pin
    print("start")
    try:
        while True:
            value = GPIO.input(pin_y)
            if value != prev_value:
                # if True:
                time.sleep(0.08)
                if value == GPIO.HIGH:
                    value_str = "HIGH"
                else:
                    value_str = "LOW"
                    stack1.push("pin2")

                # print("Value read from pin {} : {}".format(pin_y,value_str))
                # prev_value = value

            # time.sleep(1)
    finally:
        GPIO.cleanup()


def pin3(stack1):
    # print("ediwqedj")
    prev_value = None

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi

    GPIO.setup(pin_f, GPIO.IN)  # set pin as an input pin
    # GPIO.setup(pin_f, GPIO.IN)  # set pin as an input pin
    # print("start")
    try:
        while True:
            value = GPIO.input(pin_f)
            if value != prev_value:
                # if True:
                time.sleep(0.08)
                if value == GPIO.HIGH:
                    value_str = "HIGH"
                else:
                    value_str = "LOW"
                    stack1.push("pin3")

                # print("Value read from pin {} : {}".format(pin_f,value_str))
                # prev_value = value
            # time.sleep(1)
    finally:
        GPIO.cleanup()


stackwei = Stack()

argsw = parse_args()


def wei(weightbase):
    while 1:
        data_list = []
        i = 0
        # s = hex(sum([int(i,16)for i in re.split('','01 03 00 01 00 05 D4 09') if i != '']))
        # print(type(s))
        # 0x01 0x03 0x00 0x01 0x00 0x05 0xD4 0x09
        # chr(0x01)+""+chr(0x03)+""+chr(0x00)+""+chr(0x01)+""+chr(0x00)+""+chr(0x05)+""+chr(0xD4)+""+chr(0x09)
        # print("********************")
        while i < 1:
            # send_data = chr(0x01)+""+chr(0x03)+""+chr(0x00)+""+chr(0x01)+""+chr(0x00)+""+chr(0x05)+""+chr(0xD4)+""+chr(0x09)
            send_data = "\x01\x03\x00\x05\x00\x02\xD4\x0A"
            # send_data = "\x01\x03\x00\x03\x00\x02\x34\x0B"
            # send_data = "\x01\x03\x00\x01\x00\x05\xD4\x09"
            #            a = ser.write(send_data)
            # print(a)
            #            data = ser.read(44)
            # print(data)
            #            data_list.append(data)
            data_list.append(1)
            i = i + 1

        # print(data_list)

    # print(struct.unpack('<L',str(data_list[0][0])))

    if data_list.size() > 0:
        bin_data = bin(ord(data_list[0][3]))
        print(bin_data)
        add = str(bin_data)[-3]
        biaozhi = str(bin_data)[0] + str(bin_data)[2] + str(bin_data)[3] + str(bin_data)[4]
        high_eight = ord(data_list[0][4])
        mid_eight = ord(data_list[0][5])
        low_eight = ord(data_list[0][6])
        res = (high_eight * 256 * 256 + mid_eight * 256 + low_eight) / 1000.
        if add == '0':
            res = res
        else:
            res = -res
        luck = str(res)
        # print(luck)
        stackwei.push((biaozhi, luck))
        if stackwei.size() == 10:

            while stackwei.size() > 1:
                stackwei.pop()
                # print("stackwei")


'''
    while 1:
    #stackwei.push()
        data=''
    #mid_data=[]
    #for i in range(3):
            #t=time.time()
            #ct=time.ctime(t)
            #print(ct)
        data = ser.readline(33)
        #mid_data.append(data)
            #print(data)

            #t = time.time()
            #ct = time.ctime(t)
            #print(ct, ':')
            #print(data)


        #data=mid_data[2]
        #if len([r[i][0] for i in range(len(r))]) != 0:
       # print(float(str(data).split(" ")[0].split(",")[-1]+str(data).split(" ")[-2].split("kg")[0]))
            #luck="total weight:"+str(data).split(" ")[0].split(",")[-1]+str(data).split(" ")[-2]+"\n"+str([r[i][0] for i in range(len(r))])[2:-4]
        luck="total weight:"+str(float(str(data).split(" ")[0].split(",")[-1]+str(data).split(" ")[-2].split("kg")[0])-weightbase)+"kg\n"
        #print(luck) 
        #else:

        #luck="total weight:"+str(data).split(" ")[0].split(",")[-1]+str(data).split(" ")[-2]+"\n"+str([r[i][0] for i in range(len(r))])[1:-1]
        #luck="total weight:"+str(float(str(data).split(" ")[0].split(",")[-1]+str(data).split(" ")[-2].split("kg")[0])-weightbase)+"kg\n"


        #T=threading.Thread(target=l4w,args=(l4,luck,window))
        #T.start()
        #print(len(stackwei))
        #print(threading.enumerate())
        stackwei.push(luck)
        if stackwei.size()==10:

            while stackwei.size()>1:
                stackwei.pop()
                #print("stackwei")
'''


# if __name__ == '__main__':
#    main()


class GUI():
    class Struct(object):
        #__slots__ = ['image', 'kind', 'weight', 'length', 'ID', 'confirm', 'question']
        def __init__(self, image, kind, weight, length, ID, confirm, question):
            self.image = image
            self.kind = kind
            self.weight = weight
            self.length = length
            self.ID = ID
            self.confirm = confirm
            self.question = question

    def make_struct(self, image, kind, weight, length, ID, confirm, question):
        return self.Struct(image, kind, weight, length, ID, confirm, question)

    def __init__(self, root):
        #self.net = load_net("/work/darknet/yolov3_f/cfg/yolov3-voc-test.cfg".encode("UTF-8"), "/work/darknet/yolov3_f/models/yolov3-voc.backup".encode("UTF-8"), 0)
        #self.meta = load_meta("/work/darknet/yolov3_f/cfg/voc.data".encode("UTF-8"))
        self.net = load_net("../models/yolov3-voc-test.cfg".encode("UTF-8"), "../models/yolov3-voc.backup".encode("UTF-8"), 0)
        self.meta = load_meta("../models/voc.data".encode("UTF-8"))
        
        #?????????  ???ttyUSB0  ttyTHS2?????????
        self.m_serial = serial.Serial( 
        port='/dev/ttyUSB0',
        baudrate=19200,
        parity=serial.PARITY_NONE, 
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1)  #timeout#??????????????? writeTimeout???0.5#?????????
        print self.m_serial 
        
        if serial == None:
            exit(0)
        if self.m_serial.isOpen(): 
            print("open ttyUSB0 success") 
        else: 
            print("open ttyUSB0 failed") 
        self.m_serial.flushInput() #????????????????????????????????????

        #?????????200kg  ??????crc???????????? ?????????
        #buffer = [0x01, 0x10, 0x00, 0x11, 0x00, 0x02,0x04,0x00,0x00,0x4e,0x20]
        #crc_transformation = self.CalCRC16(buffer, len(buffer)) 
        #self.CRCBuffer(buffer, crc_transformation)
        #print ("crc:",buffer)
        #print("???????????????:",result)

        #result = self.m_serial.write(buffer)

        #strWeight=self.getWeight()

        self.queueImage = Queue.Queue(10)  # ?????????????????????????????????????????????????????????????????????
        self.queueImageDetect = Queue.Queue(10)
        self.strRegisterCode = ""
        self.strUUID = ""
        self.fScale = 1

        self.config("../config/config.xml")
        self.bDetectd = False
        self.fish_kind = ["??????", "??????", "??????", "??????", "??????", "??????", "??????", "??????"]
        #?????????????????????
        #self.queueResultConfirm = Queue.Queue(5)  # ?????????????????????????????????????????????????????????????????????
        #??????????????????
        self.queueResultSend = Queue.Queue(10)  # ?????????????????????????????????????????????????????????????????????

        self.queueImageConfirm = Queue.Queue(1)  # ?????????????????????????????????????????????????????????????????????

        self.result_weight = 0

        self.fangshengButton = Button()
        self.daidingButton = Button()
        self.quedingButton = Button()

        time.sleep(1)

        self.initGUI(root)
    def config(self, strPath):
        # ??????xml??????
        dom = xml.dom.minidom.parse(strPath)
        # ????????????????????????
        root = dom.documentElement

        element = root.getElementsByTagName('rtsp')
        el = element[0]
        self.strRtsp = el.getAttribute("value")

        element = root.getElementsByTagName('ID')
        el = element[0]
        self.DeviceID = el.getAttribute("value")

        element = root.getElementsByTagName('register')
        el = element[0]
        self.strRegisterCode = el.getAttribute("value")

        element = root.getElementsByTagName('scale')
        el = element[0]
        strScale = el.getAttribute("value")
        self.fScale = float(100.0 / int(strScale))#???????????????????????????

        element = root.getElementsByTagName('UUID')
        el = element[0]
        self.strUUID = el.getAttribute("value")
        if (self.getUUID() != self.strUUID):
            self.strUUID = self.getUUID()
            #print "uuid: ", self.strUUID, self.getUUID()
            #??????xml
            el.setAttribute("value", self.strUUID)
            f = open(strPath, 'w')
            dom.writexml(f, encoding='utf-8')
            f.close()
        print "DeviceID:", self.DeviceID
        print "rtsp URL:", self.strRtsp
        print "UUID:", self.strUUID

    def CalCRC16(self, data, length):
        
        print(data, length)                          #?????????????????????
        crc=0xFFFF
        if length == 0:
           length = 1
        j = 0
        while length != 0:
            crc ^= list.__getitem__(data, j)
            #print('j=0x%02x, length=0x%02x, crc=0x%04x' %(j,length,crc))
            for i in range(0,8):
                if crc & 1:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
            length -= 1
            j += 1
        return crc
    # buffer?????????????????????  crc_transformation CalCRC16??????crc??????
    def CRCBuffer(self, buffer, crc_transformation): 
        rgCrc = [0x00,0x00]
        rgCrc[0] = crc_transformation & 0xFF  # ???8???
        rgCrc[1] = (crc_transformation >> 8) & 0xFF #???8???
        L =hex(rgCrc[0]) # ???16???????????????
        H =hex(rgCrc[1])
        L_value = int(L,16) # 16????????????????????????
        H_value = int(H,16)
        buffer.append(L_value) # ???8????????? ????????????
        buffer.append(H_value)
        print(L, H)
        return buffer

    def show_frame(self, root):
        window = root
        while(self.thdFrame.is_alive()):
            if self.queueImage.empty() == True:
                time.sleep(0.1)
                continue
        #while not self.queueImage.empty():
            image = self.queueImage.get()
            #print "show_frame get pic"
            camera_Rgb_Img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            camera_Rgb_Img = cv2.resize(camera_Rgb_Img, (960, 540), interpolation=cv2.INTER_CUBIC)
            tk_image = Image.fromarray(camera_Rgb_Img)
            tk_image = ImageTk.PhotoImage(tk_image)
            #self.camera_img_label = Label(window, image=tk_image)
            #self.camera_img_label.config(image=tk_image)
            self.camera_img_label.configure(image=tk_image)
            self.camera_img_label.image = tk_image
            #self.camera_img_label.pack()
        # after??????????????????show_msg
        #self.root.after(100, self.show_frame, root)

    def openRtsp(self, strRtsp):
        ret = 0
        self.cap = cv2.VideoCapture(strRtsp)  # rtsp://admin:hk123456@192.168.18.244:554/stream1 /work/darknet/liyu2.mp4 "/work/darknet/liyu2.mp4"
        # ?????????????????????
        '''fps = self.cap.get(cv2.CAP_PROP_FPS)
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fNUMS = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print "mp4 ", fps, size, fNUM'''

        if not self.cap.isOpened():
            print ('Failed to open camera!')
            ret = 0
        else:
            ret = 1
        return ret
    #???????????? ????????????????????????????????????
    def onRtsp(self, root):
        while(self.thdRtsp.is_alive()):
            ret, frame = self.cap.read()
            #time.sleep(0.1)
            if ret:
                # rtsp????????????
                if(self.queueImage.full()):
                    self.queueImage.get()
                    #print "rtsp???????????????????????????"
                self.queueImage.put(frame)
                # ??????????????????
                if (self.queueImageDetect.full()):
                    self.queueImageDetect.get()
                    #print "Detect???????????????????????????"
                self.queueImageDetect.put(frame)
                #print "add a frame"
            else:
                #print "cap read failed"
                time.sleep(0.2)
                cv2.destroyAllWindows()
                self.cap.release()
                print "??????rtsp"
                self.openRtsp(self.strRtsp)
    def getUUID(self):
        node = uuid.getnode()
        #print node
        strMac = uuid.UUID(int = node).hex[-12:]
        strTemp = "*&^jdsfa_+~" + strMac
        strUUID = strTemp + "@#$MJBtact"
        strUUIDMD5 = md5(strUUID.encode('utf-8')).hexdigest()
        return str(strUUIDMD5)
    def getRegisterCode(self):
        strUUID = self.strUUID
        strTemp = "JYXKLfdksewiimcfaqe*&^)IK^!#?>jyx<ldfdsaljoSN:"
        strTemp = strTemp + str(strUUID) + "dk_+=-|';LK d@$#~&^(*&)M_7"
        #print "strtemp: ", strTemp
        strRegisterCode = md5(strTemp.encode('utf-8')).hexdigest()
        #print strRegisterCode
        return strRegisterCode

    def getWeight(self):
        timestart = time.time()
        self.m_serial.flushInput() #????????????????????????????????????
        #time.sleep(0.1)
        while(1):
            strRevData = self.m_serial.readline()
            print "len:" + str(len(strRevData)) + " recv:" + strRevData
            bFindwn = strRevData.find("wn")
            bFindKg = strRevData.find("Kg", bFindwn)
            if bFindwn >= 0 and bFindKg > 0 and bFindKg > bFindwn:
                strCut = strRevData[(bFindwn+2):(bFindKg+2)]
                bFindfu = strCut.find("-")
                print "cut:" + strCut
                if bFindfu >= 0: #????????????
                    return "0.00Kg"
                return strCut
            if time.time() - timestart >= 0.5:
                return "0.00Kg"

    def onDetect(self, root):
        flag = True
        iCount = 0
        timestart = time.time()
        iFrameDetect = 0
        timeDetect = time.time()
        iFirst = 1
        #???????????????bianyu,caoyu,lianyu,liyu,luyu,nianyu,qingyu,yongyu
        fish_kind = [0, 0, 0, 0, 0, 0, 0, 0]
        #???????????????
        iFishVote = -1
        #?????????9?????????????????????????????????????????????
        iFishLength = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        while(self.thdDetect.is_alive()):
            if self.queueImageDetect.empty() == True:
                time.sleep(0.1)
                continue
            image = self.queueImageDetect.get()

            if self.bDetectd:
                #print "????????????"
                continue

            imageArray = np.array(image)
            imag = nparray_to_image(imageArray)  #  ???numpy????????????IMAGE

            resultDetect = detect(self.net, self.meta, imag) # ??????
            #print "detect result: ", resultDetect
            iCount = iCount+1
            if time.time() - timestart >= 10:
                fFPS = iCount / (time.time() - timestart)
                print "??????fps: ", fFPS
                iCount = 0
                timestart = time.time()
            #5s??????????????????????????????????????????
            if time.time() - timeDetect >= 5:
                self.bDetectd = False
                iFrameDetect = 0
                fish_kind = [0, 0, 0, 0, 0, 0, 0, 0]
                iFishLength = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            # ??????????????????????????????
            if str(resultDetect) != "[]":
                timeDetect = time.time()
                # ??????x??????
                iFishLength[iFrameDetect] = resultDetect[0][2][2]
                iFrameDetect = iFrameDetect + 1
                fish_kind_detect = str(resultDetect[0][0])
                #print "detect result: ", fish_kind_detect, iFrameDetect, resultDetect[0][2]
                if fish_kind_detect == "bianyu":
                    fish_kind[0] = fish_kind[0] + 1
                    #print "bianyu+1", fish_kind[0]
                elif fish_kind_detect == "caoyu":
                    fish_kind[1] = fish_kind[1] + 1
                    #print "caoyu+1", fish_kind[1]
                elif fish_kind_detect == "lianyu":
                    fish_kind[2] = fish_kind[2] + 1
                    #print "lianyu+1", fish_kind[2]
                elif fish_kind_detect == "liyu":
                    fish_kind[3] = fish_kind[3] + 1
                    #print "liyu+1", fish_kind[3]
                elif fish_kind_detect == "luyu":
                    fish_kind[4] = fish_kind[4] + 1
                    #print "luyu+1", fish_kind[4]
                elif fish_kind_detect == "nianyu":
                    fish_kind[5] = fish_kind[5] + 1
                    #print "nianyu+1", fish_kind[0]
                elif fish_kind_detect == "qingyu":
                    fish_kind[6] = fish_kind[6] + 1
                    #print "qingyu+1", fish_kind[6]
                elif fish_kind_detect == "yongyu":
                    fish_kind[7] = fish_kind[7] + 1
                    #print "yongyu+1", fish_kind[7]

            #??????  ??????9???????????? ??????????????????5??????????????????
            if iFrameDetect == 9 and time.time() - timeDetect < 5:
                #???????????????????????????
                #print fish_kind
                fish_kind_temp = fish_kind[:]#????????????????????????????????????????????????????????????????????????????????????
                #?????????????????????????????????????????????
                fish_kind_temp.sort(reverse=True)
                fishResult = fish_kind_temp[0]
                #print fish_kind, fish_kind_temp, fishResult
                #?????????????????????????????????????????????????????????????????????????????????
                iCountFish = fish_kind_temp.count(fishResult)
                #print "iCountFish:", iCountFish
                #?????????????????????????????????
                if iCountFish == 1:
                    self.bDetectd = True
                    iFishVote = fish_kind.index(fishResult)
                if iCountFish > 1:
                    self.bDetectd = False

                if self.bDetectd:
                    #????????????
                    iCountTemp = 0
                    iCountTotal = 0
                    for iLength in iFishLength:
                        if iLength > 5:
                            #print "length: ", iLength
                            iCountTemp = iCountTemp + 1
                            iCountTotal = iCountTotal + iLength
                    fFishPixels = float(iCountTotal / iCountTemp)
                    fFishLength = fFishPixels * self.fScale
                    fFishLength = round(fFishLength, 2)  # ?????????????????????1.01
                    #print "fish length: ", fFishLength
                    print "???????????????: ", self.fish_kind[iFishVote], fFishLength
                    if self.queueImageConfirm.full():
                        self.queueImageConfirm.get()
                        print "Confirm.???????????????????????????"
                    self.queueImageConfirm.put(image)

                    fish_rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    fish_rgb_img = cv2.resize(fish_rgb_img, (400, 300), interpolation=cv2.INTER_CUBIC)
                    fish_rgb_img = Image.fromarray(fish_rgb_img)
                    fish_rgb_img = ImageTk.PhotoImage(fish_rgb_img)
                    self.fish_img_label.config(image=fish_rgb_img)
                    self.fish_img_label.image = fish_rgb_img

                    self.var_fish_category.config(text=str(self.fish_kind[iFishVote]))  # ?????????
                    self.var_fish_length.config(text=str(fFishLength))  # ?????????
                    strWeight = self.getWeight()
                    self.var_fish_weight.config(text=strWeight)

                    self.fangshengButton.pack()
                    self.daidingButton.pack()
                    self.quedingButton.pack()
                else:
                    print "??????????????????????????????????????????"

                timeDetect = time.time()
                iFrameDetect = 0
                fish_kind = [0, 0, 0, 0, 0, 0, 0, 0]
                iFishLength = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            continue

    # ?????????????????? ????????????queueResultSend
    def onSendResult(self, root):
        while(self.thdSendResult.is_alive()):
            if self.queueResultSend.empty() == True:
                time.sleep(0.1)
                continue

            result = self.queueResultSend.get()
            print "send result!!!", result.kind, result.weight, result.length


    def onFangsheng(self):
        image = self.queueImageConfirm.get()

        self.fangshengButton.pack_forget()
        self.daidingButton.pack_forget()
        self.quedingButton.pack_forget()
        self.bDetectd = False

    def onQueren(self):
        image = self.queueImageConfirm.get()
        #??????????????????
        fishKindLabel = self.var_fish_category.cget("text")
        weight = self.var_fish_weight.cget("text")
        length = self.var_fish_length.cget("text")

        #??????????????????
        detectResulttoConfirm = self.make_struct(image, fishKindLabel, weight, length, self.DeviceID, 1, 0)
        if self.queueResultSend.full():
            self.queueResultSend.get()
            print "Confirm.???????????????????????????"
        self.queueResultSend.put(detectResulttoConfirm)

        self.fangshengButton.pack_forget()
        self.daidingButton.pack_forget()
        self.quedingButton.pack_forget()
        self.bDetectd = False

    def onDaiding(self):
        image = self.queueImageConfirm.get()
        #??????????????????
        fishKindLabel = self.var_fish_category.cget("text")
        weight = self.var_fish_weight.cget("text")
        length = self.var_fish_length.cget("text")

        #??????????????????
        detectResulttoConfirm = self.make_struct(image, fishKindLabel, weight, length, self.DeviceID, 0, 1)
        if self.queueResultSend.full():
            self.queueResultSend.get()
            print "Confirm.???????????????????????????"
        self.queueResultSend.put(detectResulttoConfirm)

        self.fangshengButton.pack_forget()
        self.daidingButton.pack_forget()
        self.quedingButton.pack_forget()
        self.bDetectd = False

    def initGUI(self, root):
        self.root = root
        show_help = True
        full_scrn = False
        help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
        font = cv2.FONT_HERSHEY_PLAIN
        window = root

        # ???2????????????????????????????????????
        window.title("Fish Platform")
        window.resizable(0, 0)
        window.configure(background="blue")
        screenwidth = window.winfo_screenwidth() - 400
        screenheight = window.winfo_screenheight() - 100
        width = screenwidth
        height = screenheight
        bg_img = cv2.imread("../img/bg.jpg")
        bg_img = cv2.resize(bg_img, (width, height))
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = Image.fromarray(bg_img) #array??????image
        bg_img = ImageTk.PhotoImage(bg_img) #????????????Tkinter????????????????????????photo image???
        bg_label = Label(window, image=bg_img) #??????????????????
        bg_label.place(anchor="center", relx=0.5, rely=0.5) #??????????????????
        alignstr = '%dx%d+%d+%d' % (width, height, -10, 0)
        window.geometry(alignstr)                  #width height???????????? -10 0 ?????????????????????

        #l = tk.Label(window, text='??????????????????', bg='green', font=('Arial', 12), width=30, height=2)
        #l2 = tk.Label(window, text='', bg='green', font=('Arial', 12), width=5, height=2)
        #pm = "??????    ????????????    ?????????\n1      01      3.2kg\n03      02      3.1kg"
        #l3 = tk.Label(window, text=pm, bg='green', font=('Arial', 12), width=30, height=8)
        #lwhi = tk.Label(window, text="", font=('Arial', 12), width=30, height=2)

        # ??????frame?????? ?????????????????? label?????????frame???
        self.frame = Frame(window)

        self.frmTop = Frame(self.frame) # ???frame???side?????????  ??????label.place?????????
        self.frmBottom = Frame(self.frame) # top????????? ?????????  ????????????
        self.frmTop.pack(side=TOP)
        self.frmBottom.pack(side=BOTTOM)

        self.frmRtsp = Frame(self.frmTop) #???????????????frametop
        #???????????????ui???
        camera_Source_Img = cv2.imread("../img/111.jpg")
        camera_Rgb_Img = cv2.cvtColor(camera_Source_Img, cv2.COLOR_BGR2RGB)
        tk_image = Image.fromarray(camera_Rgb_Img)
        tk_image = ImageTk.PhotoImage(tk_image)
        self.camera_img_label = Label(self.frmRtsp)  #??????label??????frmRtsp??????
        #self.camera_img_label.place(relx=0.3, rely=0.5, anchor="center")
        self.camera_img_label.configure(image=tk_image)
        self.camera_img_label.image = tk_image
        #self.camera_img_label.pack(side=LEFT, anchor="center")
        self.camera_img_label.pack()
        #self.camera_img_label.grid(row=0, column=0)
        self.frmRtsp.pack(side=LEFT)  # frmRtsp?????????frametop?????????

        # ??????frame??????
        self.frmResult = Frame(self.frmTop)
        self.frmResult_top = Frame(self.frmResult)
        self.frmResult_bottom = Frame(self.frmResult)  #??????????????????
        self.frmResult_bottom_left = Frame(self.frmResult_bottom)

        self.frmResult_bottom_left1 = Frame(self.frmResult_bottom_left)
        self.frmResult_bottom_left2 = Frame(self.frmResult_bottom_left)
        self.frmResult_bottom_left3 = Frame(self.frmResult_bottom_left)
        self.frmResult_bottom_left4 = Frame(self.frmResult_bottom_left)

        self.frmResult_bottom_right = Frame(self.frmResult_bottom)
        self.frmResult_bottom_right1 = Frame(self.frmResult_bottom_right)
        self.frmResult_bottom_right2 = Frame(self.frmResult_bottom_right)
        self.frmResult_bottom_right3 = Frame(self.frmResult_bottom_right)
        self.frmResult_bottom_right4 = Frame(self.frmResult_bottom_right)

        fish_Source_Img = cv2.imread("../img/111.jpg")
        fish_Source_Img = cv2.resize(fish_Source_Img, (400, 300))
        fish_Rgb_Img = cv2.cvtColor(fish_Source_Img, cv2.COLOR_BGR2RGB)
        fish_Rgb_Img = Image.fromarray(fish_Rgb_Img)
        fish_Rgb_Img = ImageTk.PhotoImage(fish_Rgb_Img)
        self.fish_img_label = Label(self.frmResult_top, image=fish_Rgb_Img)
        self.fish_img_label.pack()
        self.frmResult_top.pack(side=TOP)
        # fish_img_label.place(anchor="center", relx=0.7, rely=0.3)

        self.frmResult_bottom_left.pack(side=LEFT)
        self.frmResult.pack(side=RIGHT) #??????frame?????????frametop?????????

        deviceIDLabel = Label(self.frmResult_bottom_left1, text="??????:", font=('Arial', 32), fg="black", width=6, height=1)
        deviceIDLabel.pack()
        self.frmResult_bottom_left1.pack(side=TOP)

        #fish_category.place(rely=0.5, relx=0.6)
        fish_category = Label(self.frmResult_bottom_left2, text="??????:", font=('Arial', 32), fg="black", width=6, height=1)
        #fish_weight.place(rely=0.6, relx=0.6)
        fish_category.pack()
        self.frmResult_bottom_left2.pack(side=TOP)

        fish_weight = Label(self.frmResult_bottom_left3, text="??????:", font=('Arial', 32), fg="black", width=6, height=1)
        #fish_length.place(rely=0.7, relx=0.6)
        fish_weight.pack()
        self.frmResult_bottom_left3.pack(side=TOP)

        fish_length = Label(self.frmResult_bottom_left4, text="??????:", font=('Arial', 32), fg="black", width=6, height=1)
        # fish_length.place(rely=0.7, relx=0.6)
        fish_length.pack()
        self.frmResult_bottom_left4.pack(side=TOP)

        self.deviceIDLabel = Label(self.frmResult_bottom_right1, text=self.DeviceID, font=('Arial', 32), fg="black", width=9, height=1)
        #self.var_fish_category.place(rely=0.5, relx=0.7)
        self.deviceIDLabel.pack()
        self.frmResult_bottom_right1.pack(side=TOP)

        self.var_fish_category = Label(self.frmResult_bottom_right2, text="NULL", font=('Arial', 32), fg="black", width=9, height=1)
        #self.var_fish_weight.place(rely=0.6, relx=0.7)
        self.var_fish_category.pack()
        self.frmResult_bottom_right2.pack(side=TOP)

        self.var_fish_weight = Label(self.frmResult_bottom_right3, text="NULL", font=('Arial', 32), fg="black", width=9, height=1)
        #self.var_fish_length.place(rely=0.7, relx=0.7)
        self.var_fish_weight.pack()
        self.frmResult_bottom_right3.pack(side=TOP)

        self.var_fish_length = Label(self.frmResult_bottom_right4, text="NULL", font=('Arial', 32), fg="black", width=9, height=1)
        # self.var_fish_length.place(rely=0.7, relx=0.7)
        self.var_fish_length.pack()
        self.frmResult_bottom_right4.pack(side=TOP)

        self.frmResult_bottom_right.pack(side=RIGHT)
        self.frmResult_bottom.pack()

        self.frmResult_button = Frame(self.frmResult)

        self.frmResult_button1 = Frame(self.frmResult_button, borderwidth=10)
        self.frmResult_button2 = Frame(self.frmResult_button, borderwidth=10)
        self.frmResult_button3 = Frame(self.frmResult_button, borderwidth=10)

        self.fangshengButton = Button(self.frmResult_button1, text="??????", height=1, width=5, font=('Arial', 20), command=self.onFangsheng)
        self.fangshengButton.pack_forget()
        self.frmResult_button1.pack(side=RIGHT)

        self.daidingButton = Button(self.frmResult_button2, text="??????", height=1, width=5, font=('Arial', 20), command=self.onDaiding)
        self.daidingButton.pack_forget()
        self.frmResult_button2.pack(side=RIGHT)

        self.quedingButton = Button(self.frmResult_button3, text="??????", height=1, width=5, font=('Arial', 20), command=self.onQueren)
        self.quedingButton.pack_forget()
        self.frmResult_button3.pack(side=RIGHT)
        self.frmResult_button.pack(side=TOP)

        self.frmRank = Frame(self.frmBottom)
        self.frmRankTop = Frame(self.frmRank)
        self.frmRankBottom = Frame(self.frmRank)

        rankLabel = Label(self.frmRankTop, text="??????", font=('Arial', 32), fg="black", width=6, height=1)
        rankLabel.pack()
        self.frmRankTop.pack(side=TOP)

        '''cols = ('??????', '????????????', '?????????')
        table = ttk.Treeview(self.frmRankBottom, columns=cols, show='headings', height=5)
        for col in cols:
            table.heading(col, text=col)
        table.column('??????', width=200, anchor=S)  # ?????????
        table.column('????????????', width=200, anchor=S)  # ?????????
        table.column('?????????', width=200, anchor=S)  # ?????????
        table.pack()'''
        self.frmRankBottom.pack(side=TOP)

        self.frmRank.pack(side=TOP)
        self.frame.pack()
        #??????????????????
        if self.getRegisterCode() != self.strRegisterCode:
            #print"??????", self.getRegisterCode()
            #print "??????", self.strRegisterCode
            tkinter.messagebox.showerror('??????', '???????????????????????????????????????')
            exit(0)
        #????????????
        self.thdFrame = threading.Thread(target=self.show_frame, args=(root,)) # ????????????????????????
        self.thdFrame.setDaemon(True)  # ????????????
        self.thdFrame.start()

        self.openRtsp(self.strRtsp) # ??????rtsp
        self.thdRtsp = threading.Thread(target=self. onRtsp, args=(root,)) # ???????????? ????????????????????????????????????
        self.thdRtsp.setDaemon(True)  # ????????????
        self.thdRtsp.start()

        self.thdDetect = threading.Thread(target=self.onDetect, args=(root,))
        self.thdDetect.setDaemon(True)  # ????????????
        self.thdDetect.start()

        self.thdSendResult = threading.Thread(target=self.onSendResult, args=(root,)) # ?????????????????? ????????????queueResultSend
        self.thdSendResult.setDaemon(True)  # ????????????
        self.thdSendResult.start()

        root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    myGUI = GUI(root)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2022/1/27 10:08
# @Author: yangh
# @Desc  : 获取图片相关信息


import json
import os
import urllib.request

import exifread


def getListFiles(path):
    # 获取文件目录下的所有文件（包含子文件夹内的文件）
    assert os.path.isdir(path), '%s not exist,' % path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret


def get_imgfile(path, file_list):
    dir_list = os.listdir(path)
    for x in dir_list:
        new_x = os.path.join(path, x)
        if os.path.isdir(new_x):
            get_imgfile(new_x, file_list)
        else:
            file_tuple = os.path.splitext(new_x)
            if file_tuple[1] == '.JPG':
                file_list.append(new_x)
    return file_list


# 打印所有照片信息
'''
for tag in tags.keys():
    print("Key: {}, value {}".format(tag, tags[tag]))
'''


# 获取经度或纬度
def getLatOrLng(refKey, tudeKey):
    if refKey not in picture_info:
        return None
    ref = picture_info[refKey].printable
    LatOrLng = picture_info[tudeKey].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
    LatOrLng = float(LatOrLng[0]) + float(LatOrLng[1]) / 60 + float(LatOrLng[2]) / float(LatOrLng[3]) / 3600
    if refKey == 'GPS GPSLatitudeRef' and picture_info[refKey].printable != "N":
        LatOrLng = LatOrLng * (-1)
    if refKey == 'GPS GPSLongitudeRef' and picture_info[refKey].printable != "E":
        LatOrLng = LatOrLng * (-1)
    return LatOrLng


# 调用百度地图API通过经纬度获取位置
def getlocation(lat, lng):
    # url = 'http://api.map.baidu.com/geocoder/v2/?location=' + lat + ',' + lng + '&output=json&pois=1&ak=6ssm83TiCaLylycdGqDNOLhv1xfFVoKT'
    url = 'http://api.map.baidu.com/reverse_geocoding/v3/?ak=MFjEIOUPH1nk220I85IWhmUFOIi0VrNo&output=json&coordtype=wgs84ll&location=' + lat + ',' + lng
    req = urllib.request.urlopen(url)
    res = req.read().decode("utf-8")
    str = json.loads(res)
    # print(str)
    jsonResult = str.get('result')
    formatted_address = jsonResult.get('formatted_address')
    return formatted_address


if __name__ == '__main__':
    file_list = []
    # path = 'C:/Users/yangh/Desktop/testyh'
    path = 'F:/python-project/sift-util/data'
    get_imgfile(path, file_list)
    for img_file in file_list:
        # print(img_file)
        f = open(img_file, 'rb')
        picture_info = exifread.process_file(f)
        if picture_info:
            '''
            for tag, value in picture_info.items():
                print(f'{tag}:{value}')
            '''
            # print('拍摄时间：', picture_info['EXIF DateTimeOriginal'])
            # print('照相机制造商：', picture_info['Image Make'])
            # print('照相机型号：', picture_info['Image Model'])
            # print('照片尺寸：', picture_info['EXIF ExifImageWidth'], picture_info['EXIF ExifImageLength'])
            lng = getLatOrLng('GPS GPSLongitudeRef', 'GPS GPSLongitude')  # 经度
            lat = getLatOrLng('GPS GPSLatitudeRef', 'GPS GPSLatitude')  # 纬度
            print('图片：{}, 经度：{}, 纬度:{}'.format(img_file, lng, lat))
            # location = getlocation(str(lat), str(lng))
            # print('位置：{}'.format(location))

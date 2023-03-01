#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2022/1/27 10:08
# @Author: zhouhao
# @Desc  : ndvi: 植被指数

import os

import numpy as np
from osgeo import gdal

# list_tif = glob.glob(r'C:/Users/周浩/Desktop/测试/*.tif')
# C://Users//周浩//Desktop//测试//test4.tif test图像小.tif
# out_path = r'C:/Users/周浩/Desktop/测试/结果/'

str = input("请输入图片路径：")
in_ds = gdal.Open(str)
# 获取文件所在路径以及不带后缀的文件名
(filepath, fullname) = os.path.split(str)
(prename, suffix) = os.path.splitext(fullname)
if in_ds is None:
    print('Could not open the file ' + str)
else:
    # 将MODIS原始数据类型转化为反射率
    red = in_ds.GetRasterBand(1).ReadAsArray() * 0.0001
    nir = in_ds.GetRasterBand(2).ReadAsArray() * 0.0001
    ndvi = (nir - red) / (nir + red)
    # 将NAN转化为0值
    nan_index = np.isnan(ndvi)
    ndvi[nan_index] = 0
    ndvi = ndvi.astype(np.float32)
    # 将计算好的NDVI保存为GeoTiff文件
    gtiff_driver = gdal.GetDriverByName('GTiff')
    # 批量处理需要注意文件名是变量，这里截取对应原始文件的不带后缀的文件名
    out_ds = gtiff_driver.Create(filepath + '/' + prename + 'NDVI.tif',
                                 ndvi.shape[1], ndvi.shape[0], 1, gdal.GDT_Float32)
    # 将NDVI数据坐标投影设置为原始坐标投影
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(ndvi)
    out_band.FlushCache()

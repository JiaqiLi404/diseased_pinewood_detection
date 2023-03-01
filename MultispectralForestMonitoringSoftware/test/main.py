import os
import numpy as np
from osgeo import gdal
import cv2

tif = '..\\data\\test4.tif'

def ndvi(path):
    in_ds = gdal.Open(path)
    # 获取文件所在路径以及不带后缀的文件名
    (filepath, fullname) = os.path.split(path)
    (prename, suffix) = os.path.splitext(fullname)
    if in_ds is None:
        print('Could not open the file ' + path)
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
        out_ds = gtiff_driver.Create(filepath + '/' + prename + '_ndvi.tif',
                                     ndvi.shape[1], ndvi.shape[0], 1, gdal.GDT_Float32)
        # 将NDVI数据坐标投影设置为原始坐标投影
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(ndvi)
        out_band.FlushCache()


# ndvi(tif)

# 测试图片，为反斜杠
pic = './tif/test4_ndvi.tif'
# a.图像的二值化 ，完整通道展示
src = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
# b.设置卷积核5*5
kernel = np.ones((2, 2), np.uint8)
# c.图像的腐蚀，默认迭代次数
erosion = cv2.erode(src, kernel)
# 图像的膨胀
dst = cv2.dilate(erosion, kernel)

cv2.imwrite('./tif/show.tif', dst)

# 图像比例
# cv2.namedWindow('origin', cv2.WINDOW_KEEPRATIO)
# cv2.namedWindow('after erosion', cv2.WINDOW_KEEPRATIO)
# cv2.namedWindow('after dilate', cv2.WINDOW_KEEPRATIO)

# 效果展示
# cv2.imshow('origin', src)
# # 腐蚀后
# cv2.imshow('after erosion', erosion)
# # 膨胀后
# cv2.imshow('after dilate', dst)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

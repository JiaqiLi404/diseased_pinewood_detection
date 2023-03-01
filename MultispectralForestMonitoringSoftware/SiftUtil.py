import fractions
import math
import time

import cv2.cv2 as cv2
import exifread
import numpy as np
from PIL import Image


class SiftImageOperator:
    def __init__(self, imgLeft, imgRight, leftGPS, rightGPS, showImg=False, nfeatures=None,
                 nOctaveLayers=None,
                 contrastThreshold=None,
                 edgeThreshold=None,
                 sigma=None):
        """
        :param imgLeft: 主图
        :param imgRight: 辅助图
        :param showImg:是否展示过程
        :param nfeatures: 特征点数目（算法对检测出的特征点排名，返回最好的nfeatures个特征点）
        :param nOctaveLayers: nOctaveLayers：金字塔中每组的层数（算法中会自己计算这个值）
        :param contrastThreshold: contrastThreshold：过滤掉较差的特征点的对阈值. contrastThreshold越大，返回的特征点越少.
        :param edgeThreshold: 过滤掉边缘效应的阈值. edgeThreshold越大，特征点越多（被过滤掉的越少）.
        :param sigma: 金字塔第0层图像高斯滤波系数.

        opencv默认参数：
        nOctaveLayers =3
        contrastThreshold = 0.04
        edgeThreshold = 10
        sigma =1.6
        """
        self.piex_distance_h = None
        self.piex_distance_w = None
        self.zoom = None
        self.piex_k = None
        self.piex_sina = None
        self.piex_cosa = None
        self.piex_h = None
        self.piex_w = None
        self.__H = None
        self.__dst_result_pts = None
        self.__src_result_pts = None
        self.imgFinal = None
        self.showImg = showImg
        self.__imgLeft = cv2.cvtColor(np.asarray(imgLeft), cv2.COLOR_RGB2BGR)
        self.__imgRight = cv2.cvtColor(np.asarray(imgRight), cv2.COLOR_RGB2BGR)
        self.leftGPS = leftGPS
        self.rightGPS = rightGPS
        self.__grayLeft = cv2.cvtColor(self.__imgLeft, cv2.COLOR_BGR2GRAY)
        self.__grayRight = cv2.cvtColor(self.__imgRight, cv2.COLOR_BGR2GRAY)
        self.__sift = cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
        self.leftMedia = [self.__imgLeft.shape[1] // 2, self.__imgLeft.shape[0] // 2]  # w h

    def _siftCompute(self, grayImg):
        keyPoints, describes = self.__sift.detectAndCompute(grayImg, None)
        return keyPoints, describes

    def _KDTreeMatch(self, describesLeft, describesRight):
        # K-D tree建立索引方式的常量参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # checks指定索引树要被遍历的次数
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_1 = flann.knnMatch(describesLeft, describesRight, k=2)  # 进行匹配搜索，参数k为返回的匹配点对数量
        # 把保留的匹配点放入good列表
        good1 = []
        T = 0.5  # 阈值
        # 筛选特征点
        for i, (m, n) in enumerate(matches_1):
            if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
                good1.append(m)
            #  双向交叉检查方法
        matches_2 = flann.knnMatch(describesRight, describesLeft, k=2)  # 进行匹配搜索
        # 把保留的匹配点放入good2列表
        good2 = []
        for (m, n) in matches_2:
            if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
                good2.append(m)
        match_features = []  # 存放最终的匹配点
        for i in good1:
            for j in good2:
                if (i.trainIdx == j.queryIdx) & (i.queryIdx == j.trainIdx):
                    match_features.append(i)
        return match_features

    def _getHomography(self, dst_pts, src_pts):
        # 获取视角变换矩阵
        """
         findHomography: 计算多个二维点对之间的最优单映射变换矩阵 H（3行x3列） ，使用最小均方误差或者RANSAC方法
         参考网址： https://blog.csdn.net/fengyeer20120/article/details/87798638
        """
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 10)  # mask中返回的匹配点是否满足最优单映射变换矩阵
        return H, mask

    def _drawMatches(self, imageA, imageB, src_result_pts, dst_result_pts):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for (p1, p2) in zip(src_result_pts, dst_result_pts):
            # 当点对匹配成功时，画到可视化图上
            p2[0][0] = p2[0][0] + wA
            cv2.line(vis, (int(p1[0][0]), int(p1[0][1])), (int(p2[0][0]), int(p2[0][1])), (0, 255, 0), 1)

        # 返回可视化结果
        return vis

    def _getFinalImg(self, H):
        if self.imgFinal is not None:
            return self.imgFinal
        # 得到右图的坐标点
        h1, w1, p1 = self.__imgRight.shape
        h2, w2, p2 = self.__imgLeft.shape

        # 计算四个坐标点
        corner = np.zeros((4, 2))  # 存放四个角坐标，依次为左上角，左下角， 右上角，右下角
        row, col, c = h1, w1, p1

        # 左上角(0, 0, 1)
        v2 = np.array([0, 0, 1])
        v1 = np.dot(H, v2)
        corner[0, 0] = v1[0] / v1[2]
        corner[0, 1] = v1[1] / v1[2]

        # 左下角
        v2[0] = 0
        v2[1] = row
        v1 = np.dot(H, v2)
        corner[1, 0] = v1[0] / v1[2]
        corner[1, 1] = v1[1] / v1[2]

        # 右上角
        v2[0] = col
        v2[1] = 0
        v1 = np.dot(H, v2)
        corner[2, 0] = v1[0] / v1[2]
        corner[2, 1] = v1[1] / v1[2]

        # 右下角
        v2[0] = col
        v2[1] = row
        v1 = np.dot(H, v2)
        corner[3, 0] = v1[0] / v1[2]
        corner[3, 1] = v1[1] / v1[2]

        right_top_x = np.int32(corner[2, 0])
        right_bottom_x = np.int32(corner[3, 0])
        left_top_x = np.int32(corner[0, 0])
        left_bottom_x = np.int32(corner[1, 0])

        right_top_y = np.int32(corner[2, 1])
        right_bottom_y = np.int32(corner[3, 1])
        left_top_y = np.int32(corner[0, 1])
        left_bottom_y = np.int32(corner[1, 1])

        w_max = np.maximum(right_top_x, right_bottom_x)
        w_min = np.minimum(left_bottom_x, left_top_x)
        h_max = np.maximum(left_bottom_y, right_bottom_y)
        h_min = np.minimum(right_top_y, left_top_y)

        print("原图坐标:(0,0),(", h2, ",", w2, "),矫正图坐标:(", left_top_y, ",", left_top_x, "),(", right_bottom_y, ",",
              right_bottom_x, ")")

        # 坐标转换
        if h_min < 0:
            # 补上图像
            imgRightTemp = np.zeros((h2 + np.abs(h_min), w2, p1), dtype=np.uint8)
            imgRightTemp[np.abs(h_min):, :] = self.__imgRight
            if (h1 - h_min) > (h_max - h_min):
                h = h1 - h_min
            else:
                h = h_max - h_min
            self.imgFinal = cv2.warpPerspective(imgRightTemp, H, (w_max, h))  # 坐标转换

            if self.showImg:
                showimg = cv2.resize(self.imgFinal, (1800, 900))
                cv2.imshow("imgright_h<0", showimg)

            self.imgFinal[np.abs(h_min):(np.abs(h_min) + h2), :w2] = self.__imgLeft  # 参考图像高度补齐

        else:
            if self.__imgLeft.shape[0] > h_max:
                h = self.__imgLeft.shape[0]
            else:
                h = h_max
            self.imgFinal = cv2.warpPerspective(self.__imgRight, H, (w_max, h))  # 坐标转换

            if self.showImg:
                showimg = cv2.resize(self.imgFinal, (1800, 900))
                cv2.imshow("imgright_h>0", showimg)

            self.imgFinal[:h2, :w2] = self.__imgLeft

        return self.imgFinal

    def computeImages(self):
        """
        计算相关点，为gps像素转换做准备，得到两图对应特征点像素值集
        """
        print("开始特征点计算")
        keyPointsLeft, describesLeft = self._siftCompute(self.__grayLeft)
        keyPointsRight, describesRight = self._siftCompute(self.__grayRight)
        print("左图", len(keyPointsLeft), "个特征点，右图", len(keyPointsRight), "个特征点")
        # 特征匹配
        print("开始特征点匹配")
        match_features = self._KDTreeMatch(describesLeft, describesRight)
        # 获取视角变换矩阵
        print("开始视角变换矩阵计算")
        src_pts = np.float32([keyPointsLeft[m.queryIdx].pt for m in match_features]).reshape(-1, 1, 2)  # 转换成列表
        dst_pts = np.float32([keyPointsRight[m.trainIdx].pt for m in match_features]).reshape(-1, 1, 2)

        self.__H, mask = self._getHomography(dst_pts, src_pts)

        # 存放精匹配后的特征点
        self.__src_result_pts = []
        self.__dst_result_pts = []
        self.piex_distance_w = []
        self.piex_distance_h = []
        for i, value in enumerate(mask):
            if value == 1:
                self.__src_result_pts.append(src_pts[i])
                self.__dst_result_pts.append(dst_pts[i])
                # bef=np.array([dst_pts[i][0][0], dst_pts[i][0][1], 1])
                # aft = np.dot(self.__H, bef)
                # w = aft[0] / bef[2]
                # h = aft[1] / bef[2]
                # self.__piex_distance.append(
                #     [round(src_pts[i][0][0] - w), round(src_pts[i][0][1] - h)])
                self.piex_distance_w.append(round(dst_pts[i][0][0] - src_pts[i][0][0]))
                self.piex_distance_h.append(round(dst_pts[i][0][1] - src_pts[i][0][1]))

        # 精密计算系数:
        self.zoom = 1e7

        # 计算比例系数和角度
        # 计算两个向量的坐标
        self.piex_w = np.median(self.piex_distance_w)
        self.piex_h = np.median(self.piex_distance_h)
        gps_w = self.rightGPS[0] * self.zoom - self.leftGPS[0] * self.zoom
        gps_h = self.rightGPS[1] * self.zoom - self.leftGPS[1] * self.zoom

        # 计算向量比例系数
        piex_distance = (self.piex_w ** 2 + self.piex_h ** 2) ** 0.5
        gps_distance = (gps_w ** 2 + gps_h ** 2) ** 0.5
        self.piex_k = gps_distance / piex_distance

        # 计算向量角度cosa=gps*piex/|gps|*|piex|  ->  gps=cosa*|gps|*|piex|/piex
        self.piex_cosa = (gps_w * self.piex_w + gps_h * self.piex_h) / (piex_distance * gps_distance)
        print(gps_w, gps_h)
        self.piex_sina = (1 - self.piex_cosa ** 2) ** 0.5

        print("特征点寻找完毕，共", len(self.__src_result_pts), "对")
        print("宽度方向特征点像素差:", self.piex_w, " ", self.piex_distance_w)
        print("高度方向特征点像素差:", self.piex_h, " ", self.piex_distance_h)

    # 进行坐标转换
    def getGPS(self, w, h):
        piex_coord = [w - self.leftMedia[0], h - self.leftMedia[1]]
        gps_coord_temp = [self.piex_k * piex_coord[0], self.piex_k * piex_coord[1]]
        # 向量旋转公式：https://blog.csdn.net/zhinanpolang/article/details/82912325
        # x1 = x0 * cosB - y0 * sinB y1 = x0 * sinB + y0 * cosB
        gps_coord = [gps_coord_temp[0] * self.piex_cosa - gps_coord_temp[1] * self.piex_sina,
                     gps_coord_temp[0] * self.piex_sina + gps_coord_temp[1] * self.piex_cosa]
        gps_coord = [gps_coord[0] / self.zoom + self.leftGPS[0], gps_coord[1] / self.zoom + self.leftGPS[1]]
        return gps_coord

    def mergeImages(self, filename="result"):
        """
        进行图像拼接
        :param filename:保存文件名
        :return:
        """
        if self.__src_result_pts is None:
            self.computeImages()
        # 拼接图像
        self.imgFinal = self._getFinalImg(self.__H)
        cv2.imwrite(filename + ".png", self.imgFinal)

        if self.showImg:
            test = self._drawMatches(self.__imgLeft, self.__imgRight, self.__src_result_pts, self.__dst_result_pts)
            test = cv2.resize(test, (1800, 900))
            cv2.imshow("test", test)
            showimg = cv2.resize(self.imgFinal, (1800, 900))
            cv2.imshow("final", showimg)
            cv2.waitKey(0)


# 读取文件exif信息
def getGPS(file):
    """
    这里要注意，opencv左上角为原点，w，h为相对原点的宽、长度距离
    """
    gpsListLong = None
    gpsListLa = None
    with open(file, 'rb') as f:
        info = exifread.process_file(f)
        gpsListLong = info.get('GPS GPSLongitude', '0').values
        gpsListLa = info.get('GPS GPSLatitude', '0').values
    gps = GPSList2Float(gpsListLong, gpsListLa)
    # print("gps:", gps)
    return gps


# gps度分秒转小数
def GPSList2Float(gpsListLong, gpsListLa):
    if gpsListLong is None or gpsListLa is None: return None
    return [gpsListLong[0] + gpsListLong[1] / 60 + float(gpsListLong[2] / 3600),
            gpsListLa[0] + gpsListLa[1] / 60 + float(gpsListLa[2] / 3600)]


# gps小数转度分秒
def GPSFloat2List(gpsLong, gpsLa):
    def get_DuFenMiao(num):
        du = math.floor(num)
        num = (num - du) * 60
        fen = math.floor(num)
        num = (num - fen) * 60
        miao = fractions.Fraction(num)
        return [du, fen, miao]

    return [get_DuFenMiao(gpsLong), get_DuFenMiao(gpsLa)]


def test_images(txt_name, img1_url, img2_url):
    def write_list(file, data):
        for i in data:
            file.write(str(i) + '\n')

    imgL = Image.open(img1_url)
    imgR = Image.open(img2_url)
    siftImageOperator = SiftImageOperator(imgL, imgR, getGPS(img1_url), getGPS(img2_url), True)
    siftImageOperator.computeImages()
    txt = open(txt_name.replace(':', '-') + '.txt', 'a')
    delta_piex = [[i - siftImageOperator.piex_w, j - siftImageOperator.piex_h] for i, j in
                  zip(siftImageOperator.piex_distance_w, siftImageOperator.piex_distance_h)]
    delta_gps_temp = [[siftImageOperator.piex_k * i[0], siftImageOperator.piex_k * i[1]] for i in delta_piex]
    delta_gps = [[(i[0] * siftImageOperator.piex_cosa - i[1] * siftImageOperator.piex_sina) / siftImageOperator.zoom,
                  (i[0] * siftImageOperator.piex_sina + i[1] * siftImageOperator.piex_cosa) / siftImageOperator.zoom]
                 for i in delta_gps_temp]
    txt.write('图像内部误差：' + '\n')
    write_list(txt, delta_gps)


def test():
    data_pairs = [[223, 224], [224, 225], [233, 234], [234, 235], [235, 236], [211, 212], [212, 213], [213, 214],
                  [214, 215], [215, 216], [216, 217], [217, 218], [218, 219], [219, 220], [209, 210]]
    txt_name = str(time.asctime(time.localtime(time.time())))

    def get_full_img_name(num):
        img_url_bef = "./img/DJI_0"
        img_url_aft = ".JPG"
        return img_url_bef + str(num) + img_url_aft

    for i, data in enumerate(data_pairs):
        print("process:", i + 1, '/', len(data_pairs))
        main_file = get_full_img_name(data[0])
        assist_file = get_full_img_name(data[1])
        test_images(txt_name, main_file, assist_file)


test()

imgMainUrl = "./img/DJI_0218.JPG"
imgAssistUrl = "./img/DJI_0217.JPG"
imgLeft = Image.open(imgMainUrl)
imgRight = Image.open(imgAssistUrl)
# imgLeft = Image.open("./test/1.png")
# imgRight = Image.open("./test/2.png")
"""
这里要注意，opencv左上角为原点，w，h为相对原点的宽、长度距离
"""
siftImageOperator = SiftImageOperator(imgLeft, imgRight, getGPS(imgMainUrl), getGPS(imgAssistUrl), True)
siftImageOperator.computeImages()
print(siftImageOperator.getGPS(3085, 1477))

# imgMainUrl = "./img/DJI_0217.JPG"
# imgAssistUrl = "./img/DJI_0218.JPG"
# imgLeft = Image.open(imgMainUrl)
# imgRight = Image.open(imgAssistUrl)
# # imgLeft = Image.open("./test/1.png")
# # imgRight = Image.open("./test/2.png")
# """
# 这里要注意，opencv左上角为原点，w，h为相对原点的宽、长度距离
# """
# siftImageOperator = SiftImageOperator(imgLeft, imgRight, getGPS(imgMainUrl), getGPS(imgAssistUrl), True)
# siftImageOperator.computeImages()
# print(siftImageOperator.getGPS(1065, 2559))

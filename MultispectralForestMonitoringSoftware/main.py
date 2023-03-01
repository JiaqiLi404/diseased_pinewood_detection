import cv2
import numpy as np
from osgeo import gdal
from osgeo import osr


def getNVDI(img):
    height, width, colors = img.shape
    img_cv = img.copy()
    # BGR
    img_np = np.array(img_cv, np.float32)
    img_np_temp = img_np.copy()
    # RGB
    img_np_temp[:, :, 0] = img_np[:, :, 1]
    img_np_temp[:, :, 1] = img_np[:, :, 0]
    img_np_temp[:, :, 2] = img_np[:, :, 0]
    img_float = img_np.astype(np.float32, copy=True)

    nvdi_mask = (img_np_temp[:, :, 0] - img_np_temp[:, :, 1]) / (img_np_temp[:, :, 0] + img_np_temp[:, :, 1])
    ndvi_mask = np.bitwise_and(np.where(nvdi_mask > 0, 1, 0), np.where(nvdi_mask < 0.2, 1, 0))
    ndvi_mask = np.bitwise_and(ndvi_mask, np.where(img_np_temp[:, :, 0] > 0.25, 1, 0))
    ndvi_mask = np.bitwise_and(ndvi_mask, np.where(img_np_temp[:, :, 1] > 0.25, 1, 0))
    ndvi_mask = np.where(ndvi_mask > 0, 255, 0)
    ndvi_mask = ndvi_mask.astype(np.uint8)
    print(ndvi_mask)
    return ndvi_mask


def image_process(ndvi):
    cv2.imshow("bef_process", ndvi)
    core = np.ones((5, 5))
    img = ndvi.copy()
    aft_img = cv2.dilate(img, core)
    core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 10))
    core = core[1:, :]
    aft_img = cv2.erode(aft_img, core)
    core = np.ones((15, 15))
    aft_img = cv2.dilate(aft_img, core)
    core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    aft_img = cv2.erode(aft_img, core)
    core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    aft_img = cv2.erode(aft_img, core)
    return aft_img


def get_center(origin, aft_nvdi):
    img = aft_nvdi.copy()
    origin = origin.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    center = []
    for index in range(num_labels):
        # TODO：这里用的是连通数的最小外接矩形的面积，matlab代码用的是真实点数代表面积
        if stats[index][4] > 50:
            center.append(centroids[index])
            # 画矩形
            cv2.rectangle(origin, (stats[index][0], stats[index][1]),
                          (stats[index][0] + stats[index][2], stats[index][1] + stats[index][3]), color=(255, 0, 0))
            cv2.rectangle(aft_nvdi, (stats[index][0], stats[index][1]),
                          (stats[index][0] + stats[index][2], stats[index][1] + stats[index][3]), color=(255, 0, 0))
    cv2.imshow("aft_process", aft_nvdi)
    cv2.imshow("or", origin)
    cv2.waitKey(0)

    return center


class TifOperator:
    def __init__(self, tif_file):
        self.file = gdal.Open(tif_file)
        self.XSize = self.file.RasterXSize
        self.YSize = self.file.RasterYSize
        self.gt_matrix = self.file.GetGeoTransform()
        self.proj = self.file.GetProjection()

    def get_coord(self, width, height):
        # 图上坐标转投影坐标
        x = self.gt_matrix[0] + width * self.gt_matrix[1] + height * self.gt_matrix[2]
        y = self.gt_matrix[3] + width * self.gt_matrix[4] + height * self.gt_matrix[5]
        # 投影坐标转经纬度
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(self.proj)
        geosrs = prosrs.CloneGeogCS()
        ct = osr.CoordinateTransformation(prosrs, geosrs)
        coords = ct.TransformPoint(x, y)
        return coords[1], coords[0]

    def get_ndvi(self):
        # 将MODIS原始数据类型转化为反射率
        red = self.file.GetRasterBand(1).ReadAsArray() * 0.0001
        nir = self.file.GetRasterBand(2).ReadAsArray() * 0.0001
        ndvi = (nir - red) / (nir + red)
        # 将NAN转化为0值
        nan_index = np.isnan(ndvi)
        ndvi[nan_index] = 0
        ndvi = ndvi.astype(np.float32)
        # 将计算好的NDVI保存为GeoTiff文件
        gtiff_driver = gdal.GetDriverByName('GTiff')
        # 批量处理需要注意文件名是变量，这里截取对应原始文件的不带后缀的文件名
        out_ds = gtiff_driver.Create('./img/_ndvi.tif',
                                     ndvi.shape[1], ndvi.shape[0], 1, gdal.GDT_Float32)
        # 将NDVI数据坐标投影设置为原始坐标投影
        out_ds.SetProjection(self.file.GetProjection())
        out_ds.SetGeoTransform(self.file.GetGeoTransform())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(ndvi)
        out_band.FlushCache()


def with_gdal_ndvi():
    img = cv2.imread("./img/test4.tif", cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (1500, 800))
    tif = TifOperator("./img/test4.tif")
    tif.get_ndvi()
    ndvi = cv2.imread("./img/_ndvi.tif", cv2.IMREAD_UNCHANGED)
    ndvi = cv2.resize(ndvi, (1500, 800))
    cv2.imshow("nvdi", ndvi)
    cv2.waitKey(0)
    ndvi = np.bitwise_and(np.where(ndvi > 0, 255, 0), np.where(ndvi < 0.2, 255, 0))
    ndvi = np.array(ndvi).astype(np.uint8)
    ndvi = image_process(ndvi)
    get_center(img, ndvi)
    tif = TifOperator("./img/he2.tif")
    print(tif.get_coord(10, 10))


def with_matlab_ndvi():
    img = cv2.imread("./img/test.png", cv2.IMREAD_UNCHANGED)
    ndvi = getNVDI(img)
    ndvi = image_process(ndvi)
    get_center(img, ndvi)
    tif = TifOperator("./img/he2.tif")
    print(tif.get_coord(10, 10))


if __name__ == '__main__':
    with_matlab_ndvi()

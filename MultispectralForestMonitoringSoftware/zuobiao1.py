import cv2
from osgeo import gdal

gdal.AllRegister()

filePath = r"C:/Users/yangh/Desktop/matlab/datatest/datatest/he2.tif"
# filePath = r"C:/Users/yangh/Desktop/matlab/DSC06549.JPG"
dataset = gdal.Open(filePath)

adfGeoTransform = dataset.GetGeoTransform()
print(adfGeoTransform)
print(adfGeoTransform[0])
print(adfGeoTransform[3])

nXSize = dataset.RasterXSize  # 列数
nYSize = dataset.RasterYSize  # 行数

arrSlope = []
for i in range(nYSize):
    row = []
    for j in range(nXSize):
        px = adfGeoTransform[0] + i * adfGeoTransform[1] + j * adfGeoTransform[2]
        py = adfGeoTransform[3] + i * adfGeoTransform[4] + j * adfGeoTransform[5]
        col = [px, py]
        row.append(col)
    arrSlope.append(row)
# print(arrSlope)
# print(len(arrSlope))

img = cv2.imread(filePath, 2)
# print(img)
print(img.shape)
# print(img.dtype)
# print(img.min())
# print(img.max())
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.namedWindow("image")
cv2.imshow("image", img)

key = cv2.waitKey(1000 // 12)
if key == ord('q'):
    cv2.destroyAllWindows()

if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
    cv2.destroyAllWindows()

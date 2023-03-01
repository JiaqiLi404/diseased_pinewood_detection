import csv
import glob
import tkinter as tk
import tkinter.ttk
from itertools import groupby
from operator import itemgetter

import cv2
import exifread
import numpy as np
import openpyxl
from PIL import Image
from icontmp import ico
import base64, os


# 加载窗体
def set_winfo(win, title, size, fixed=False):
    win.title(title)
    win.geometry(size)
    set_icon(win)
    if fixed:
        # 固定窗口大小
        win.resizable(0, 0)
    win.mainloop()


# 窗体居中大小
def center_window(win, width, height):
    screenwidth = win.winfo_screenwidth()
    screenheight = win.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    return size


# 设置窗口图标
def set_icon(win):
    icon_name = 'tmp.ico'
    tmp = open(icon_name, 'wb+')
    tmp.write(base64.b64decode(ico))
    tmp.close()
    win.iconbitmap(icon_name)
    os.remove(icon_name)


# 获取指定目录下的指定后缀的文件数量
def file_num(path, suffix='*'):
    return len(glob.glob(pathname=path + '/*.' + suffix))


# 计算共多少组照片
def group_img_num(path, img_arr=['.jpeg', '.jpg', '.png', '.gif', '.tif']):
    img_suffix_arr = []
    path_arr, name_arr = file_list(path)
    for index, file_name in enumerate(name_arr):
        if os.path.splitext(file_name)[-1].lower() in img_arr:
            suffix = file_name[0:file_name.index('_')]
            img_suffix_arr.append(suffix)
    return list(set(img_suffix_arr))

# 获取指定目录下的文件
def file_list(path):
    path_arr = []
    name_arr = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            path_arr.append(file_path)
            name_arr.append(file_name)
    return path_arr, name_arr


# 根据飞行时间和拍摄频率 获取照片数量
def shot_img_num(fly_time, rate):
    return int(fly_time / rate)


# 获取指定目录下文件名称最大的
def file_max_name(path):
    return max([file_name for file_name in file_list(path)[1]])


# def file_num(path, suffix='*'):
#     return len(glob.glob(path + '/%s' % suffix))


# 打开excel
def excel_open(path, sheet_name='Sheet1'):
    # 打开文件
    wb = openpyxl.load_workbook(path)
    # 选择表单
    sh = wb[sheet_name]
    # 获取数据
    return list(sh.rows)


# 获取excel数据集
def excel_data(path, sheet_name='Sheet1'):
    rows_data = excel_open(path, sheet_name)
    # 收集信息
    data_arr = []
    for case in rows_data:
        data = []
        for cell in case:  # 获取一条测试用例数据
            data.append(cell.value)
        data_arr.append(data)
    return data_arr


# 获取指定excel的数据行数
def excel_row_num(path, sheet_name='Sheet1', start_row=1):
    data_list = excel_open(path, sheet_name)
    return len(data_list) - start_row;


# 拼接csv路径
def csv_path(path):
    # return path[0: path.rindex('/') + 1] + 'data.csv'
    return path + '/' + 'data.csv'


# 读取csv文件(含表头)
def csv_data(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        column = [row for row in reader]
    return column


# 获取csv行数
def csv_line(path):
    path = csv_path(path)
    f1 = open(path, "r")
    line_num = len(f1.readlines())
    f1.close()
    return line_num


# 删除csv指定行
def csv_remove(path, csv_list):
    csv_arr = []
    for row_arr in csv_list:
        csv_arr.append(','.join(row_arr))
    f = open(path, "w+")
    f.write('\n'.join(csv_arr))
    f.close()


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
    return gps


# gps度分秒转小数
def GPSList2Float(gpsListLong, gpsListLa):
    if gpsListLong is None or gpsListLa is None: return None
    return [gpsListLong[0] + gpsListLong[1] / 60 + float(gpsListLong[2] / 3600),
            gpsListLa[0] + gpsListLa[1] / 60 + float(gpsListLa[2] / 3600)]


# 获取照片各像素点灰度值
# def px_grayscale(path):
#     arr = []
#     img = Image.open(path)
#     image = img.load()
#     width = img.size[0]
#     height = img.size[1]
#     for x in range(width):
#         for y in range(height):
#             r, g, b = image[x, y]
#             # 获得灰度值
#             arr.append((r + g + b) / 3)
#     return arr

# 获取照片各像素点灰度值
def px_grayscale(path):
    img = Image.open(path)
    gray_img = img.convert('L')
    return np.array(gray_img)


# 检测目录中 图像灰度值超过255 占到照片的10%及以上比例
def grayscale_bright(path, window, pro, point, img_arr=['.jpeg', '.jpg', '.png', '.gif', '.tif']):
    file_path_arr = []
    pro['maximum'] = file_num(path)
    pro['value'] = 1
    for file_name in os.listdir(path):
        # 检测是否是照片
        if os.path.splitext(file_name)[-1].lower() in img_arr:
            file_path = os.path.join(path, file_name)
            try:
                # 获取照片灰度值
                gray_arr = px_grayscale(file_path)
                # 占比超过10%
                if np.sum(gray_arr >= 255) / np.sum(gray_arr >= 0) >= point:
                    file_path_arr.append(file_path)
            except:
                print('文件 %s 255灰度值操作打开时出现异常' % file_path)
        pro['value'] += 1
        window.update()
    return file_path_arr


# 检测目录中 图像灰度值为0 占到照片的10%及以上比例
def grayscale_dark(path, window, pro, point, img_arr=['.jpeg', '.jpg', '.png', '.gif', '.tif']):
    file_path_arr = []
    pro['maximum'] = file_num(path)
    pro['value'] = 1
    for file_name in os.listdir(path):
        # 检测是否是照片
        if os.path.splitext(file_name)[-1].lower() in img_arr:
            file_path = os.path.join(path, file_name)
            try:
                # 获取照片灰度值
                gray_arr = px_grayscale(file_path)
                # 占比超过10%
                if np.sum(gray_arr == 0) / np.sum(gray_arr >= 0) >= point:
                    file_path_arr.append(file_path)
            except:
                print('文件 %s 0灰度值操作打开时出现异常' % file_path)
        pro['value'] += 1
        window.update()
    return file_path_arr


# 获取目录下模糊的图像
def img_vague(path, window, pro, img_arr=['.jpeg', '.jpg', '.png', '.gif', '.tif'], threshold=100):
    """
    :param path: 目录路径
    :param threshold: 模糊参考值
    :return: 模糊的图像路径
    """
    arr = []
    pro['maximum'] = file_num(path)
    pro['value'] = 1
    for img_path in file_list(path)[0]:
        if os.path.splitext(img_path)[-1].lower() in img_arr:
            img = cv2.imread(img_path)
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                if fm < threshold:
                    arr.append(img_path)
            except:
                print('文件 %s 模糊操作打开时出现异常' % img_path)
        pro['value'] += 1
        window.update()
    return arr


# 获取指定路径的文件夹大小（单位：GB）
def directory_size(path, window, pro):
    # 获取文件数量
    pro['maximum'] = file_num(path)
    size = 0.0
    for root, dirs, files in os.walk(path):
        # size += sum([os.path.getsize(os.path.join(root, file)) for file in files])
        for file in files:
            size += os.path.getsize(os.path.join(root, file))
            pro['value'] += 1
            window.update()
    size = round(size / 1024 / 1024 / 1024, 2)
    return size


# 扫描检测csv中的GPS数据和照度数据，查看表格中是否缺失数据，若缺失，记录编号，剔除对应数据和图像
def inspect_eliminate(path, window, pro):
    img_path = path
    path = csv_path(path)
    record_no = []
    pro['maximum'] = csv_line(img_path)
    csv_list = []
    f = open(path, 'r')
    reader = csv.reader(f)
    for index, rows in enumerate(reader):
        not_null = True
        # 检测每一组数据
        for i, item in enumerate(rows):
            # 检测GPS
            if (i == 4 or i == 6 or i == 1) and len(item.strip()) == 0:
                not_null = False
                # 记录编号 第一列为编号
                record_no.append(rows[0])
        if not_null:
            csv_list.append(rows)
        pro['value'] += (index + 1)
        window.update()
    f.close()
    # 删除对应图像
    img_remove(img_path, record_no)
    # 删除行
    csv_remove(path, csv_list)


# 剔除高度低于5米的数据
def inspect_height(path, window, pro, height):
    img_path = path
    path = csv_path(path)
    record_no = []
    pro['maximum'] = csv_line(img_path)
    csv_list = []
    f = open(path, 'r')
    reader = csv.reader(f)
    for index, rows in enumerate(reader):
        not_null = True
        # 检测每一组数据
        for i, item in enumerate(rows):
            # 检测高度
            if (i == 11) and float(item) < height:
                not_null = False
                # 记录编号 第一列为编号
                record_no.append(rows[0])
        if not_null:
            csv_list.append(rows)
        pro['value'] += (index + 1)
        window.update()
    f.close()
    # 删除对应图像
    img_remove(img_path, record_no)
    # 删除行
    csv_remove(path, csv_list)


# 计算每组图片的平均值 相差超过10%的照片
def group_gray(path, pro, img_arr=['.jpeg', '.jpg', '.png', '.gif', '.tif']):
    # 获得所有照片
    img_list = []
    pro['maximum'] = file_num(path)
    pro['value'] = 1
    for file_name in os.listdir(path):
        # 检测是否是照片
        if os.path.splitext(file_name)[-1].lower() in img_arr:
            file_path = os.path.join(path, file_name)
            # 前缀
            suffix = file_name[0:file_name.index('_')]
            img_list.append({'suffix': suffix, 'name': file_name, 'path': file_path})
        pro['value'] += 1
    # 排序
    img_list.sort(key=itemgetter('suffix'))
    # 分组
    img_group = groupby(img_list, itemgetter('suffix'))
    # 计算每组平均灰度值
    group_gray_arr = []
    for key, group in img_group:
        # avg_sum = 0
        for item in group:
            # 求灰度值
            gray_arr = px_grayscale(item['path'])
            # 平均灰度值
            avg_gray = np.average(gray_arr)
            # avg_sum += avg_gray
        # 灰度值相差不超过10%

    return group_gray_arr


# 删除图像
def img_remove(path, no, img_arr=['.jpeg', '.jpg', '.png', '.gif', '.tif']):
    no = list(map(int, no))
    for file_name in os.listdir(path):
        if os.path.splitext(file_name)[-1].lower() in img_arr:
            suffix = int(file_name[0:file_name.index('_')])
            if suffix in no:
                del_file = path + '/' + file_name
                os.remove(del_file)


# 任务进度
def show_progress(window, label_text, label_x, label_y, label_width, progress_width=465, progress_height=22):
    # 进度条标签
    tk.Label(window, text=label_text, ).place(x=label_x, y=label_y)
    pro = tkinter.ttk.Progressbar(window, length=465)
    pro.place(x=label_width, y=label_y)
    # 进度值最大值
    pro['maximum'] = 100
    # 进度值初始值
    pro['value'] = 0
    return pro


# 进度条清空
def progress_clear(*pros):
    for pro in pros:
        pro['value'] = 0

import random
from xml.etree import ElementTree as ET
import os
import cv2

image_folder = "./Tree4TrainNew/JPEGImages"
annotation_folder = "./Tree4TrainNew/Annotations"
train_data_folder = "./Tree4TrainNew/trainImages"


# 读取所有图片
def load_image_files(folder):
    image_files = os.listdir(folder)
    images = []
    for file in image_files:
        indexDot = file.rfind('.')
        images.append(file[:indexDot])
    return images


# opencv打开某一图片
def load_image_by_opencv(file):
    return cv2.imread(image_folder + "/" + file + ".png")


# 读取疫木的位置信息,返回[xmin,ymin,xmax,ymax]横向是x轴，竖向是y轴
def load_annotation(file):
    # ET去打开xml文件
    tree = ET.parse(annotation_folder + "/" + file + ".xml")
    # 获取根标签
    root = tree.getroot()
    rois = []
    for obj in root.findall("object"):
        rois.append([int(obj.find("bndbox").find("xmin").text), int(obj.find("bndbox").find("ymin").text),
                     int(obj.find("bndbox").find("xmax").text), int(obj.find("bndbox").find("ymax").text)])
    return rois


# 裁剪图片
def cut_image(image, xmin, xmax, ymin, ymax):
    return image[ymin:ymax, xmin:xmax]


# 剪裁所有数据集
def cut_train_sets(images):
    def __indent(elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                __indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def build_text_root(name, text):
        root_temp = ET.Element(name)
        root_temp.text = text
        return root_temp

    def build_bndbox_root(xmin, ymin, xmax, ymax):
        root_temp = ET.Element('bndbox')
        root_temp.append(build_text_root('xmin', str(xmin)))
        root_temp.append(build_text_root('ymin', str(ymin)))
        root_temp.append(build_text_root('xmax', str(xmax)))
        root_temp.append(build_text_root('ymax', str(ymax)))
        return root_temp

    for im in images:
        bundles = load_annotation(im)
        image = load_image_by_opencv(im)
        # 截取反例
        for index_train in range(2):
            xmin = random.randint(0, 519)
            ymin = random.randint(0, 519)
            train_image = cut_image(image, xmin, xmin + 80, ymin, ymin + 80)
            cv2.imwrite(train_data_folder + "/healthy/" + "H" + im + "." + str(index_train) + ".png", train_image)
            # 将反例加入xml文件中
            # ET去打开xml文件
            tree = ET.parse(annotation_folder + "/" + im + ".xml")
            # 获取根标签
            root = tree.getroot()
            # 加入正常树信息
            obj_root = ET.Element('object')
            obj_root.append(build_text_root("name", "HealthyTree"))
            obj_root.append(build_text_root("pose", "Unspecified"))
            obj_root.append(build_text_root("truncated", "0"))
            obj_root.append(build_text_root("difficult", "0"))
            obj_root.append(build_bndbox_root(xmin, ymin, xmin + 80, ymin + 80))
            root.append(obj_root)
            __indent(root)
            tree.write(annotation_folder + '/new/' + im + '.xml', encoding='utf-8', xml_declaration=True)
        for index_train, b in enumerate(bundles):
            train_image = cut_image(image, b[0], b[2], b[1], b[3])
            cv2.imwrite(train_data_folder + "/ill/" + "I" + im + "." + str(index_train) + ".png", train_image)
    print("裁剪完毕")


images = load_image_files(image_folder)
cut_train_sets(images)
# healthy_images = load_image_files(train_data_folder + "/healthy")
# ill_images = load_image_files(train_data_folder + "/ill")
# print(healthy_images)
# cv2.waitKey(0)

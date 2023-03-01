# -*- coding: utf-8 -*-
from PIL import Image


# 将图片填充为正方形
def fill_image(image):
    width, height = image.size
    # 选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    # 生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    # 将之前的图粘贴在新图上，居中
    if width > height:  # 原图宽大于高，则填充图片的竖直维度
        # (x,y)二元组表示粘贴上图相对下图的起始位置
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image


# 切图
def cut_image(image, num):
    width, height = image.size
    item_width = int(width / num)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, num):  # 两重循环，生成9张图片基于原图的位置
        for j in range(0, num):
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)
    print('拆分尺寸计算完成...')
    return [image.crop(box) for box in box_list]


# 保存
def save_images(new_dir, image_list):
    for index, image in enumerate(image_list):
        img_num = str(index + 1)
        print('正在保存第%s张...' % img_num)
        image.save(new_dir + '/' + img_num + '.tif', 'PNG')


def start(path, new_dir, num):
    num = int(num ** 0.5)
    image = Image.open(path)
    image = fill_image(image)
    image_list = cut_image(image, num)
    save_images(new_dir, image_list)


if __name__ == '__main__':
    dir = 'D:/Desktop/cut/odm_orthophoto.tif'
    new_dir = 'D:/Desktop/cuted'
    # 拆分张数 需为平方根 例: 1、4、9、16 ... -> 拆分为 1张、4张、9张、16张....
    num = 4
    start(dir, new_dir, num)

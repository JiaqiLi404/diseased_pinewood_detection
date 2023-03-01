#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2022/1/25 10:08
# @Author: yangh
# @Desc  : 递归获取文件夹大小

import os
import time

from os.path import join, getsize

t1 = time.time()


def getdirsize(dir):
    '''
    获取文件的字节数

    :param dir: 目录
    :return:
    '''
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([getsize(join(root, name)) for name in files])
    return size


def size_format(size):
    '''
    格式化
    :param size: 字节
    :return:
    '''
    if size < 1000:
        return '%i' % size + ' byte'
    elif 1024 <= size < 1024 * 1024:
        return '%.3f' % float(size / 1024) + ' KB'
    elif 1024 * 1024 <= size < 1024 * 1024 * 1024:
        return '%.3f' % float(size / 1024 / 1024) + ' MB'
    elif 1024 * 1024 * 1024 <= size < 1024 * 1024 * 1024 * 1024:
        return '%.3f' % float(size / 1024 / 1024 / 1024) + ' GB'
    elif 1024 * 1024 * 1024 * 1024 <= size:
        return '%.3f' % float(size / 1024 / 1024 / 1024 / 1024) + ' TB'


if __name__ == '__main__':
    # fileStr = input("请输入文件路径:")
    # size = getdirsize(r'G:\BaiduNetdiskDownload\20211026-4(10)')
    size = getdirsize(r'G:\BaiduNetdiskDownload\20211026-4(10)')
    t2 = time.time()
    print('文件大小：', size_format(size), '，共耗时：%s' % str(t2 - t1))

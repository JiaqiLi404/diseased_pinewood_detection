import os
import tkinter as tk
import tkinter.messagebox as msg
import tkinter.filedialog as fd
import util.taskUtil as util

'''
打包命令：
pyinstaller --clean --win-private-assemblies -i ./handleimg.ico ./handleimg.py -F -w -p F:/python-project/multispectral_forestlnglat/util
'''

# 字体
global_font = (None, 14)
window = tk.Tk()


# 进度条
pro_win_save = util.show_progress(window, '检查图像是否保存：', 145, 80, 260)

pro_win_effective = util.show_progress(window, '检查GPS和照度数据的有效性：', 85, 120, 260)

pro_win_agreement = util.show_progress(window, '检查照片与GPS照度数据的数量一致性：', 37, 160, 260)

pro_win_fly = util.show_progress(window, '检查起飞降落的高度：', 132, 200, 260)

pro_win_light = util.show_progress(window, '检查照片是否存在过曝：', 120, 240, 260)

pro_win_dark = util.show_progress(window, '检查图像是否过暗：', 144, 280, 260)

# pro_win_saturation = util.show_progress(window, '检查各组图像之间的亮暗差距是否过大：', 36, 320, 260)

pro_win_vague = util.show_progress(window, '检查图像是否存在运动模糊：', 94, 320, 260)

# 配置信息
# 文件夹大小
tk.Label(window, text='文件夹大小：', ).place(x=142, y=400)
dir_size_input = tk.StringVar()
dir_size_input.set(70)
tk.Entry(window, show=None, textvariable=dir_size_input).place(x=222, y=400)
tk.Label(window, text='GB', ).place(x=343, y=400)

# 剔除高度
tk.Label(window, text='剔除高度：', ).place(x=428, y=400)
height_input = tk.StringVar()
height_input.set(5)
tk.Entry(window, show=None, textvariable=height_input).place(x=497, y=400)
tk.Label(window, text='米  ', ).place(x=626, y=400)

# 灰度占比
tk.Label(window, text='灰度值占比：', ).place(x=142, y=440)
gray_input = tk.StringVar()
gray_input.set(10)
tk.Entry(window, show=None, textvariable=gray_input).place(x=222, y=440)
tk.Label(window, text='%  ', ).place(x=343, y=440)

# 超过占比
tk.Label(window, text='超过占比未达标：', ).place(x=392, y=440)
over_flower_input = tk.StringVar()
over_flower_input.set(33.33)
tk.Entry(window, show=None, textvariable=over_flower_input).place(x=497, y=440)
tk.Label(window, text='%  ', ).place(x=626, y=440)


# 选择文件
def open_file():
    directory = fd.askdirectory()
    if directory:
        # 清空进度条
        util.progress_clear(pro_win_save, pro_win_effective, pro_win_agreement, pro_win_fly, pro_win_light,
                            pro_win_dark, pro_win_vague)
        # 检查图像是否保存
        directory_size = util.directory_size(directory, window, pro_win_save)
        if directory_size < float(dir_size_input.get()):
            msg.showerror(title='提示', message='检测文件夹大小：%sGB，结果未达标！' % directory_size)
            return

        # 检查GPS和照度数据的有效性
        util.inspect_eliminate(directory, window, pro_win_effective)

        # 检查照片与GPS照度数据的数量一致性
        pro_win_agreement['value'] = 20
        window.update()
        csv_no = util.csv_line(directory)
        pro_win_agreement['value'] = 70
        window.update()
        group_img_num = len(util.group_img_num(directory))
        pro_win_agreement['value'] = 100
        window.update()
        if csv_no != group_img_num:
            msg.showerror(title='提示', message='照片序号与csv数据序号不等！')
            return

        # 剔除起飞降落的高度小于5米的
        util.inspect_height(directory, window, pro_win_fly, float(height_input.get()))

        # 图像总数
        img_num = util.file_num(directory, 'tif')

        # 照片过曝
        img_light_num = len(util.grayscale_bright(directory, window, pro_win_light, float(gray_input.get()) / 100))
        if img_light_num / img_num > float(over_flower_input.get()) / 100:
            msg.showerror(title='提示', message='%s%%以上照片存在过曝情况！' % over_flower_input.get())
            return

        # 照片过暗
        img_dark_num = len(util.grayscale_dark(directory, window, pro_win_dark, float(gray_input.get()) / 100))
        if img_dark_num / img_num > float(over_flower_input.get()) / 100:
            msg.showerror(title='提示', message='%s%%以上照片存在过暗情况！' % over_flower_input.get())
            return

        # 每组图像之间的亮暗差距是否过大
        # util.group_gray(directory, pro_win_saturation)

        # 检查图像是否存在运动模糊
        img_vague_num = len(util.img_vague(directory, window, pro_win_vague))
        if img_vague_num / img_num > float(over_flower_input.get()) / 100:
            msg.showerror(title='提示', message='%s%%以上照片存在模糊情况！' % over_flower_input.get())
            return

        # 检测完毕
        msg.showinfo(title='提示', message='检测完毕，均无未达标情况！')


start_btn = tk.Button(window, text='开始处理', font=global_font, command=open_file)
start_btn.place(x=355, y=20)

win_size = util.center_window(window, 800, 500)
util.set_winfo(window, '图像处理分析检测', win_size, True)

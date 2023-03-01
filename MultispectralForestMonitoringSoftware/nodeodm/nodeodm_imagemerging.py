import math
import os
from pyodm import Node
import time
from tqdm import tqdm

image_dir = './data/'
result_path = './results/'
ip = 'localhost'  # 改为自己的ip
port = 3000
file_per_iter = 25

start = time.time()
images_name = os.listdir(image_dir)
images_name = [image_dir + image_name for image_name in images_name]
print(images_name)

n = Node(ip, port)
print("Node连接成功，{}张图开始处理".format(len(images_name)))

task_iter = math.ceil(len(images_name) / file_per_iter)
task_iter = 5
while task_iter != 0:
    images_iter = images_name[(task_iter - 1) * file_per_iter:task_iter * file_per_iter]
    print("\n创建新任务,iter:{}/{}\n".format(math.ceil(len(images_name) / file_per_iter) - task_iter + 1,
                                        math.ceil(len(images_name) / file_per_iter)))
    task = n.create_task(images_iter, {'orthophoto-resolution': 0.0274, "min-num-features": 10000, "dsm": False,
                                       "ignore-gsd": False})
    pbar = tqdm(total=100)
    processing = 0
    while True:
        try:
            info = task.info()
        except Exception as e:
            print('连接失败 %s' % (e))
            time.sleep(1)
            continue
        if info.progress == 100:
            break
        pbar.update(info.progress - processing)
        processing = info.progress
        if info.last_error != '':
            print("error ", info.last_error)

        time.sleep(1)
    pbar.close()
    try:
        task.download_assets(result_path + str(task_iter) + '/')
    except Exception as e:
        print('航拍图像错误 %s' % (e))

    task_iter -= 1

print("处理完成")
# task.wait_for_completion()
# task.download_assets(result_path)

print("{}张图消耗{}秒".format(len(images_name), time.time() - start))

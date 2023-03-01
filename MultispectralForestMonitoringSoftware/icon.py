import base64

open_icon = open("handleimg.ico", "rb")  # 选择图标文件
b64str = base64.b64encode(open_icon.read())
open_icon.close()
write_data = "ico=%s" % b64str
f = open("icontmp.py", "w+")
f.write(write_data)  # 生成ASCII码
f.close()

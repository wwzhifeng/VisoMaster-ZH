print()
print()
import pyfiglet

# 生成大字 "VisoMaster-CN"
big_text = pyfiglet.figlet_format("VisoMaster-CN", font="slant")
print("========================================================================")
# 打印大字
print(big_text)
print("========================================================================")
print()
print()

# 打印其他信息
print("---中文版🍒---")
print()
print()
print("---油管：王知风---")
print()
print("---B站：AI王知风---")
print()
#print("---AI工具QQ群：957732664---")
print()

from app.ui import main_ui
from PySide6 import QtWidgets 
import sys

import qdarktheme
from app.ui.core.proxy_style import ProxyStyle

if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle())
    with open("app/ui/styles/dark_styles.qss", "r") as f:
        _style = f.read()
        _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"})+'\n'+_style
        app.setStyleSheet(_style)
    window = main_ui.MainWindow()
    window.show()
    app.exec()
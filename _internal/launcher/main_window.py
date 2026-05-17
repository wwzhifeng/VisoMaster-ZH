"""
启动器主窗口
============
布局:
  ┌─────────────────────────────────────────┐
  │  VisoMaster TRT 整合包                   │
  │  ─────────────────────────────────────  │
  │  状态卡片 x3 (环境 / 模型 / 引擎)        │
  │  ─────────────────────────────────────  │
  │  快速操作 (网格按钮)                     │
  │  ─────────────────────────────────────  │
  │      [  启 动 主 程 序  ]                │
  │  ─────────────────────────────────────  │
  │  日志输出区 (折叠/展开)                  │
  └─────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

_ACCEPTED = QDialog.DialogCode.Accepted

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from _internal.launcher.workers import (
    BuildEnginesWorker, DownloadWorker, EnvCheckWorker, MainAppLauncher,
    WarmupEnginesWorker,
)
from _internal.launcher.dialogs import (
    BuildEnginesDialog, DownloadPickerDialog, SettingsDialog,
)


# ---------------------------------------------------------------------------
# 状态卡片
# ---------------------------------------------------------------------------
class StatusCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            StatusCard {
                background-color: #2c2f35;
                border: 1px solid #3a3d44;
                border-radius: 8px;
            }
        """)
        self.setMinimumHeight(90)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(4)

        self.title = QLabel(title)
        self.title.setStyleSheet("color:#999; font-size:12px;")
        lay.addWidget(self.title)

        self.value = QLabel("检测中...")
        f = QFont(); f.setPointSize(13); f.setBold(True)
        self.value.setFont(f)
        self.value.setStyleSheet("color:#eee;")
        lay.addWidget(self.value)

        self.detail = QLabel("")
        self.detail.setStyleSheet("color:#888; font-size:11px;")
        self.detail.setWordWrap(True)
        lay.addWidget(self.detail)

    def set_status(self, value: str, detail: str = "", color: str = "#eee"):
        self.value.setText(value)
        self.value.setStyleSheet(f"color:{color};")
        self.detail.setText(detail)


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisoMaster TRT 整合包 启动器")
        self.resize(880, 620)

        self._apply_dark_theme()
        self._build_ui()

        self._workers = []  # 防止 QThread 被 GC
        QTimer.singleShot(100, self._refresh_status)

    # -----------------------------------------------------------------------
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e2024; }
            QLabel { color: #ddd; }
            QGroupBox {
                color: #ddd; border: 1px solid #3a3d44;
                border-radius: 6px; margin-top: 12px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
            }
            QPushButton {
                background-color: #3a3d44; color: #eee;
                border: 1px solid #4a4d54; border-radius: 6px;
                padding: 8px 16px; min-height: 24px;
            }
            QPushButton:hover { background-color: #45484f; }
            QPushButton:pressed { background-color: #2c2f35; }
            QPushButton#primary {
                background-color: #4facc9; color: white;
                font-weight: bold; font-size: 14px;
                border: none; border-radius: 8px;
                min-height: 44px;
            }
            QPushButton#primary:hover { background-color: #5fbcd9; }
            QPushButton#primary:pressed { background-color: #3f9cb9; }
            QPushButton#primary:disabled { background-color: #555; color: #999; }
            QTextEdit {
                background-color: #15171a; color: #c0c0c0;
                border: 1px solid #3a3d44; border-radius: 4px;
                font-family: 'Consolas', 'Microsoft YaHei Mono', monospace;
                font-size: 11px;
            }
            QProgressBar {
                background-color: #2c2f35; border: 1px solid #3a3d44;
                border-radius: 4px; text-align: center; color: white;
            }
            QProgressBar::chunk { background-color: #4facc9; border-radius: 3px; }
            QDialog { background-color: #1e2024; }
            QCheckBox, QRadioButton { color: #ddd; spacing: 8px; }
            QComboBox, QSpinBox {
                background-color: #2c2f35; color: #eee;
                border: 1px solid #3a3d44; border-radius: 4px;
                padding: 4px 8px; min-height: 22px;
            }
            QComboBox::drop-down { border: none; }
        """)

    # -----------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # 标题
        title_row = QHBoxLayout()
        title = QLabel("VisoMaster TRT 整合包")
        f = QFont(); f.setPointSize(18); f.setBold(True)
        title.setFont(f); title.setStyleSheet("color:#4facc9;")
        title_row.addWidget(title)
        title_row.addStretch()
        ver = QLabel("cu128 / TRT 10.16 / v2.0")
        ver.setStyleSheet("color:#777; font-size:11px;")
        title_row.addWidget(ver)
        root.addLayout(title_row)

        # 状态卡片
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self.card_env = StatusCard("运行环境")
        self.card_models = StatusCard("模型状态")
        self.card_engines = StatusCard("TRT 引擎")
        cards_row.addWidget(self.card_env)
        cards_row.addWidget(self.card_models)
        cards_row.addWidget(self.card_engines)
        root.addLayout(cards_row)

        # 快速操作
        ops = QGroupBox("快速操作")
        grid = QGridLayout(ops)
        grid.setSpacing(10)
        self.btn_download = QPushButton("📥  下载模型")
        self.btn_build = QPushButton("⚙️  编译 LP 引擎")
        self.btn_build.setToolTip(
            "默认只编译 LivePortrait (表情驱动) 用的 6 个引擎。\n"
            "其它模型由主程序的 ORT TensorRT EP 自动编译,\n"
            "首次选 TensorRT provider 时会卡 1~3 分钟编译, 这是正常的。"
        )
        self.btn_warmup = QPushButton("🔥  预热 LP 引擎")
        self.btn_warmup.setToolTip(
            "预热 LivePortrait 引擎到显存。\n"
            "如果你只用换脸 (不用表情驱动), 这步可跳过。"
        )
        self.btn_recheck = QPushButton("🔍  重新检测环境")
        self.btn_settings = QPushButton("🛠  高级设置")
        self.btn_open_dir = QPushButton("📂  打开项目目录")
        self.btn_clean = QPushButton("🧹  清理临时缓存")
        grid.addWidget(self.btn_download, 0, 0)
        grid.addWidget(self.btn_build, 0, 1)
        grid.addWidget(self.btn_warmup, 0, 2)
        grid.addWidget(self.btn_recheck, 1, 0)
        grid.addWidget(self.btn_settings, 1, 1)
        grid.addWidget(self.btn_open_dir, 1, 2)
        grid.addWidget(self.btn_clean, 2, 0)
        root.addWidget(ops)

        self.btn_download.clicked.connect(self._on_download)
        self.btn_build.clicked.connect(self._on_build)
        self.btn_warmup.clicked.connect(self._on_warmup)
        self.btn_recheck.clicked.connect(self._refresh_status)
        self.btn_settings.clicked.connect(self._on_settings)
        self.btn_open_dir.clicked.connect(self._on_open_dir)
        self.btn_clean.clicked.connect(self._on_clean)

        # 进度条 (隐藏直到任务开始)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumHeight(8)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        # 主启动按钮
        self.btn_launch = QPushButton("启 动 主 程 序")
        self.btn_launch.setObjectName("primary")
        self.btn_launch.clicked.connect(self._on_launch)
        root.addWidget(self.btn_launch)

        # 日志输出
        log_box = QGroupBox("运行日志")
        log_lay = QVBoxLayout(log_box)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(200)
        log_lay.addWidget(self.log)
        root.addWidget(log_box, stretch=1)

    # -----------------------------------------------------------------------
    def _log(self, msg: str):
        self.log.append(msg)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # -----------------------------------------------------------------------
    # 状态刷新
    # -----------------------------------------------------------------------
    def _refresh_status(self):
        self.card_env.set_status("检测中...", "", "#999")
        self.card_models.set_status("检测中...", "", "#999")
        self.card_engines.set_status("检测中...", "", "#999")
        self._log("=== 检测环境 ===")

        w = EnvCheckWorker()
        w.log.connect(self._log)
        w.done.connect(self._on_env_done)
        self._workers.append(w)
        w.start()

    def _on_env_done(self, r: dict):
        # 环境卡片
        if r["gpu_name"] != "未知" and r["torch_ver"] != "?" and r["trt_ver"] != "?":
            self.card_env.set_status(
                r["gpu_name"],
                f"驱动 {r['driver']} | TRT {r['trt_ver']} | PyTorch {r['torch_ver']}",
                "#4caf50",
            )
            launch_ok = True
        else:
            self.card_env.set_status(
                "环境异常", "查看日志了解详情", "#f44336",
            )
            launch_ok = False

        # 模型卡片
        m_ok, m_tot = r["models_ok"], r["models_total"]
        if m_tot == 0:
            self.card_models.set_status("无清单", "manifest.json 缺失", "#f44336")
            launch_ok = False
        elif m_ok == m_tot:
            self.card_models.set_status(f"{m_ok}/{m_tot}", "全部就位", "#4caf50")
        elif m_ok >= 8:  # 大致核心包够了
            self.card_models.set_status(f"{m_ok}/{m_tot}", "核心已备, 可启动", "#ffa726")
        else:
            self.card_models.set_status(f"{m_ok}/{m_tot}", "模型不足, 请下载", "#f44336")
            launch_ok = False

        # 引擎卡片
        e_ok, e_tot = r["engines_ok"], r["engines_total"]
        if e_ok == 0:
            self.card_engines.set_status("未编译", "首次启动需要编译", "#ffa726")
        elif e_ok >= 7:
            self.card_engines.set_status(f"{e_ok}/{e_tot}", "核心引擎已编译", "#4caf50")
        else:
            self.card_engines.set_status(f"{e_ok}/{e_tot}", "部分编译", "#ffa726")

        self.btn_launch.setEnabled(launch_ok)
        if launch_ok and e_ok == 0:
            self.btn_launch.setText("启 动 主 程 序 (首次启动将自动编译引擎)")
        else:
            self.btn_launch.setText("启 动 主 程 序")

    # -----------------------------------------------------------------------
    # 下载
    # -----------------------------------------------------------------------
    def _on_download(self):
        try:
            mf = json.loads((ROOT / "_internal" / "manifest.json")
                            .read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取清单失败: {e}")
            return

        # 计算各 pack 状态
        status_by_pack = {}
        for name, items in mf["packs"].items():
            ok = sum(1 for it in items
                     if (ROOT / it["path"]).exists()
                     and (ROOT / it["path"]).stat().st_size > 0)
            status_by_pack[name] = (ok, len(items))

        dlg = DownloadPickerDialog(mf, status_by_pack, self)
        if dlg.exec() != _ACCEPTED or not dlg.selected:
            return

        self._set_busy(True)
        self._log(f"\n=== 准备下载: {', '.join(dlg.selected)} ===")
        w = DownloadWorker(dlg.selected)
        w.log.connect(self._log)
        w.done.connect(lambda ok: self._on_download_done(ok))
        self._workers.append(w)
        w.start()

    def _on_download_done(self, ok: bool):
        self._set_busy(False)
        self._log(f"\n=== 下载{'完成' if ok else '失败/部分失败'} ===")
        if not ok:
            QMessageBox.warning(self, "下载",
                "部分文件下载失败, 请查看日志。可重新点击下载继续 (断点续传)。")
        self._refresh_status()

    # -----------------------------------------------------------------------
    # 编译引擎
    # -----------------------------------------------------------------------
    def _on_build(self):
        dlg = BuildEnginesDialog(self)
        if dlg.exec() != _ACCEPTED:
            return
        self._set_busy(True)
        self._log(f"\n=== 开始编译 TRT 引擎 (模式: {dlg.mode}) ===")
        self._log("这一步会持续较长时间, 请耐心等待...")
        w = BuildEnginesWorker(dlg.mode)
        w.log.connect(self._log)
        w.done.connect(lambda ok: self._on_build_done(ok))
        self._workers.append(w)
        w.start()

    def _on_build_done(self, ok: bool):
        self._set_busy(False)
        self._log(f"\n=== 编译{'完成' if ok else '部分失败'} ===")
        if not ok:
            QMessageBox.warning(self, "编译",
                "部分引擎编译失败。请查看 engines\\*\\_build_log.txt 了解详情。")
        self._refresh_status()

    # -----------------------------------------------------------------------
    # 设置
    # -----------------------------------------------------------------------
    def _on_warmup(self):
        if QMessageBox.question(self, "预热",
            "预热将依次加载所有 TRT 引擎到显存 + 初始化 CUDA context。\n"
            "完成后主程序首次使用 TensorRT-Engine 模式不会卡顿。\n\n"
            "预计耗时 30 秒 ~ 2 分钟, 期间显存占用会很高。\n"
            "开始吗?",
            QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self._set_busy(True)
        self._log("\n=== 开始预热 TRT 引擎 ===")
        w = WarmupEnginesWorker()
        w.log.connect(self._log)
        w.done.connect(lambda ok: self._on_warmup_done(ok))
        self._workers.append(w)
        w.start()

    def _on_warmup_done(self, ok: bool):
        self._set_busy(False)
        self._log(f"=== 预热{'完成' if ok else '失败'} ===")
        if ok:
            QMessageBox.information(self, "预热",
                "引擎预热完成! 现在启动主程序, 切到 TensorRT-Engine 模式\n"
                "第一次推理也不会卡顿了。")
        self._refresh_status()

    def _on_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == _ACCEPTED:
            self._log("[设置] 配置已保存, 重启主程序后生效。")

    # -----------------------------------------------------------------------
    # 其它按钮
    # -----------------------------------------------------------------------
    def _on_open_dir(self):
        os.startfile(str(ROOT))

    def _on_clean(self):
        if QMessageBox.question(self, "清理",
            "将清空 tmp\\ 临时缓存和所有 __pycache__ 目录。\n"
            "(不会删除模型和引擎) 继续吗?",
            QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        n = 0
        import shutil
        tmp = ROOT / "tmp"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(exist_ok=True)
        for pcache in ROOT.rglob("__pycache__"):
            try:
                shutil.rmtree(pcache, ignore_errors=True)
                n += 1
            except Exception:
                pass
        self._log(f"[清理] 完成, 删除 {n} 个 __pycache__ 目录")

    # -----------------------------------------------------------------------
    # 启动主程序
    # -----------------------------------------------------------------------
    def _on_launch(self):
        self._set_busy(True)
        self.btn_launch.setText("主程序运行中...")
        self._log("\n=== 启动主程序 ===")
        w = MainAppLauncher()
        w.log.connect(self._log)
        w.done.connect(self._on_main_done)
        self._workers.append(w)
        w.start()

    def _on_main_done(self, rc: int):
        self._set_busy(False)
        self.btn_launch.setText("启 动 主 程 序")
        self._log(f"=== 主程序已退出 (rc={rc}) ===")

    # -----------------------------------------------------------------------
    def _set_busy(self, busy: bool):
        for b in [self.btn_download, self.btn_build, self.btn_recheck,
                  self.btn_settings, self.btn_clean, self.btn_launch]:
            b.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.progress.setRange(0, 0)  # 无限滚动模式


# ---------------------------------------------------------------------------
def run():
    # bootstrap DLL paths 第一时间注入 (防止 PySide6/torch 加载 DLL 失败)
    import _internal.bootstrap as bs
    bs._register_dll_dirs()

    app = QApplication(sys.argv)
    app.setApplicationName("VisoMaster TRT Launcher")
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()

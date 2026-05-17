"""启动器对话框: 下载选择 / 引擎编译选项 / 高级设置"""
from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QSpinBox, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = ROOT / "config.ini"


# ---------------------------------------------------------------------------
# 下载选择对话框
# ---------------------------------------------------------------------------
class DownloadPickerDialog(QDialog):
    def __init__(self, manifest: dict, status_by_pack: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择要下载的模型包")
        self.setMinimumWidth(520)
        self.selected: list[str] = []

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("勾选要下载的模型包 (已完整的会自动跳过):"))

        self.checks: dict[str, QCheckBox] = {}
        grp = QGroupBox()
        gl = QVBoxLayout(grp)
        for name, items in manifest["packs"].items():
            meta = manifest["packs_meta"].get(name, {})
            ok, total = status_by_pack.get(name, (0, len(items)))
            size_mb = meta.get("size_mb", 0)
            desc = meta.get("desc", "")
            if ok == total:
                tag = f"[已完整]"
                color = "#4caf50"
            elif ok == 0:
                tag = f"[缺失]"
                color = "#f44336"
            else:
                tag = f"[{ok}/{total}]"
                color = "#ffa726"
            cb = QCheckBox(f"{name}  ({size_mb} MB)  {desc}")
            cb.setChecked(ok < total)
            cb.setStyleSheet(f"color:{color}")
            self.checks[name] = cb
            gl.addWidget(cb)
        layout.addWidget(grp)

        # 镜像选择
        mirror_row = QHBoxLayout()
        mirror_row.addWidget(QLabel("下载源:"))
        self.mirror = QComboBox()
        self.mirror.addItems(["hf-mirror", "huggingface", "github"])
        cfg = ConfigParser(); cfg.read(CONFIG, encoding="utf-8")
        cur = cfg.get("download", "mirror", fallback="hf-mirror")
        if cur in ["hf-mirror", "huggingface", "github"]:
            self.mirror.setCurrentText(cur)
        mirror_row.addWidget(self.mirror)
        mirror_row.addStretch()
        layout.addLayout(mirror_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self):
        self.selected = [n for n, cb in self.checks.items() if cb.isChecked()]
        # 写回镜像选择
        cfg = ConfigParser(); cfg.read(CONFIG, encoding="utf-8")
        if "download" not in cfg:
            cfg["download"] = {}
        cfg["download"]["mirror"] = self.mirror.currentText()
        with open(CONFIG, "w", encoding="utf-8") as f:
            cfg.write(f)
        self.accept()


# ---------------------------------------------------------------------------
# 引擎编译选项对话框
# ---------------------------------------------------------------------------
class BuildEnginesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编译 TensorRT 引擎")
        self.setMinimumWidth(520)
        self.mode = "core"

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("选择编译模式:"))

        self.r_core = QRadioButton("编译 LivePortrait 引擎 (推荐, 约 3 分钟, 共 6 个)")
        self.r_all = QRadioButton("编译全部 ONNX 引擎 (高级用户, 约 25~40 分钟)")
        self.r_force = QRadioButton("强制重新编译全部 (换显卡 / 升级 TRT 后)")
        self.r_core.setChecked(True)
        for r in (self.r_core, self.r_all, self.r_force):
            layout.addWidget(r)

        layout.addSpacing(8)
        info = QLabel(
            "说明:\n"
            "• 主程序里 LivePortrait (表情驱动) 功能直接使用编译好的引擎\n"
            "• 主程序里其它模型 (换脸/检测/修复/超分) 走 ORT TensorRT EP,\n"
            "  会在用户选 'TensorRT' provider 后由 ORT 自己重新编译到\n"
            "  tensorrt-engines\\ 目录, 启动器编译的不会被复用\n\n"
            "• 所以默认只编 LivePortrait 引擎就够了\n"
            "• 引擎与 GPU 型号绑定, 换显卡要重编"
        )
        info.setStyleSheet("color:#999; font-size:11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self):
        if self.r_core.isChecked():
            self.mode = "core"
        elif self.r_all.isChecked():
            self.mode = "all"
        else:
            self.mode = "force"
        self.accept()


# ---------------------------------------------------------------------------
# 高级设置对话框
# ---------------------------------------------------------------------------
class SettingsDialog(QDialog):
    """编辑 config.ini 的关键参数"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("高级设置")
        self.setMinimumWidth(540)

        self.cfg = ConfigParser()
        self.cfg.read(CONFIG, encoding="utf-8")

        layout = QVBoxLayout(self)

        # === 运行时 ===
        grp1 = QGroupBox("运行时")
        f1 = QFormLayout(grp1)
        self.provider = QComboBox()
        self.provider.addItems(["trt", "cuda", "cpu"])
        self.provider.setCurrentText(self.cfg.get("runtime", "provider", fallback="trt"))
        f1.addRow("推理后端:", self.provider)

        self.gpu_threads = QSpinBox()
        self.gpu_threads.setRange(1, 8)
        self.gpu_threads.setValue(self.cfg.getint("runtime", "gpu_threads", fallback=1))
        f1.addRow("GPU 推理线程:", self.gpu_threads)

        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(1, 32)
        self.cpu_threads.setValue(self.cfg.getint("runtime", "cpu_threads", fallback=4))
        f1.addRow("CPU 处理线程:", self.cpu_threads)
        layout.addWidget(grp1)

        # === 性能优化开关 ===
        grp2 = QGroupBox("性能优化 (修改后需重启主程序)")
        f2 = QVBoxLayout(grp2)
        self.opt_sync = QCheckBox("LivePortrait 单次同步 (推荐, +10~20%)")
        self.opt_fp16 = QCheckBox("FP16 IO 绑定 (推荐, +5~10%)")
        self.opt_trt = QCheckBox("换脸/修复走 TRT 引擎 (推荐, +30~60%)")
        self.opt_batch = QCheckBox("多人脸合批 (多脸场景明显, 单脸无影响)")
        self.opt_thread = QCheckBox("GPU 单线程化 (稳定性优先)")
        self.opt_graph = QCheckBox("CUDA Graph (实验, 可能不稳)")
        for w in (self.opt_sync, self.opt_fp16, self.opt_trt,
                  self.opt_batch, self.opt_thread, self.opt_graph):
            f2.addWidget(w)
        self.opt_sync.setChecked(self.cfg.getboolean("advanced", "batch_synchronize", fallback=True))
        self.opt_fp16.setChecked(self.cfg.getboolean("advanced", "fp16_io_binding", fallback=True))
        self.opt_trt.setChecked(self.cfg.getboolean("advanced", "trt_swapper", fallback=True))
        self.opt_batch.setChecked(self.cfg.getboolean("advanced", "batch_faces", fallback=False))
        self.opt_thread.setChecked(self.cfg.getboolean("advanced", "decouple_threading", fallback=False))
        self.opt_graph.setChecked(self.cfg.getboolean("advanced", "enable_cuda_graph", fallback=False))
        layout.addWidget(grp2)

        # === TRT 编译 ===
        grp3 = QGroupBox("TRT 编译")
        f3 = QFormLayout(grp3)
        self.workspace = QSpinBox()
        self.workspace.setRange(512, 16384); self.workspace.setSingleStep(512)
        self.workspace.setSuffix(" MB")
        self.workspace.setValue(self.cfg.getint("advanced", "trt_workspace_mb", fallback=4096))
        f3.addRow("编译显存上限:", self.workspace)
        layout.addWidget(grp3)

        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _set(self, section, key, val):
        if section not in self.cfg:
            self.cfg[section] = {}
        self.cfg[section][key] = str(val)

    def _on_save(self):
        self._set("runtime", "provider", self.provider.currentText())
        self._set("runtime", "gpu_threads", self.gpu_threads.value())
        self._set("runtime", "cpu_threads", self.cpu_threads.value())
        self._set("advanced", "batch_synchronize", self.opt_sync.isChecked())
        self._set("advanced", "fp16_io_binding", self.opt_fp16.isChecked())
        self._set("advanced", "trt_swapper", self.opt_trt.isChecked())
        self._set("advanced", "batch_faces", self.opt_batch.isChecked())
        self._set("advanced", "decouple_threading", self.opt_thread.isChecked())
        self._set("advanced", "enable_cuda_graph", self.opt_graph.isChecked())
        self._set("advanced", "trt_workspace_mb", self.workspace.value())
        with open(CONFIG, "w", encoding="utf-8") as f:
            self.cfg.write(f)
        self.accept()

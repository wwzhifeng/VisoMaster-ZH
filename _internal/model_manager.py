"""
VisoMaster TRT Portable - 模型管理器
================================
功能:
  --check {pack}         检查指定包模型是否齐全 (退出码 0=齐, 1=缺)
  --menu                 交互式菜单 (Download_Models.bat 入口)
  --download {pack}      命令行批量下载指定包
  --verify [pack]        SHA256 校验
  --list                 列出所有包和状态
特性:
  - 多镜像 fallback: hf-mirror -> huggingface -> github
  - HTTP Range 断点续传
  - SHA256 校验
  - 并行下载 (单文件分块 + 多文件并发)
  - 失败重试 + 镜像切换
依赖: 仅标准库 + tqdm (tqdm 在 site-packages 中已包含)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from configparser import ConfigParser
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "_internal" / "manifest.json"
CONFIG_PATH = ROOT / "config.ini"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_config() -> ConfigParser:
    cfg = ConfigParser()
    if CONFIG_PATH.exists():
        cfg.read(CONFIG_PATH, encoding="utf-8")
    return cfg


def get_preferred_mirror() -> str:
    cfg = load_config()
    return cfg.get("download", "mirror", fallback="hf-mirror")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"manifest.json 不存在: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def list_packs(manifest: dict) -> list[tuple[str, dict]]:
    """返回 [(pack_name, pack_meta_with_items), ...] 保持 manifest 顺序"""
    return list(manifest.get("packs", {}).items())


def pack_status(items: list[dict]) -> tuple[int, int, int]:
    """返回 (已下载数, 总数, 总字节)"""
    n_ok = 0
    n_total = len(items)
    size_total = 0
    for item in items:
        p = ROOT / item["path"]
        size_total += item.get("size_mb", 0)
        if p.exists() and p.stat().st_size > 0:
            n_ok += 1
    return n_ok, n_total, size_total


# ---------------------------------------------------------------------------
# 下载 (支持 Range 续传 + 多镜像 fallback)
# ---------------------------------------------------------------------------
USER_AGENT = "VisoMaster-Portable/1.0"
CHUNK_SIZE = 1 << 20  # 1 MiB


def resolve_url(template: str, mirror: str) -> str:
    """
    manifest 里的 mirrors 是模板 URL, 包含 {mirror} 占位 (可选)。
    这里允许两种写法:
      "https://hf-mirror.com/visomaster/assets/resolve/main/foo.onnx"
      "{mirror}/visomaster/assets/resolve/main/foo.onnx"
    """
    bases = {
        "hf-mirror":    "https://hf-mirror.com",
        "huggingface":  "https://huggingface.co",
        "github":       "https://github.com",
    }
    if "{mirror}" in template:
        return template.replace("{mirror}", bases.get(mirror, bases["hf-mirror"]))
    return template


def download_with_resume(url: str, dest: Path, expected_size: Optional[int] = None) -> bool:
    """
    单文件下载 + 断点续传。返回 True/False。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if resume > 0:
        req.add_header("Range", f"bytes={resume}-")

    # Python 3.10 urllib 默认不跟 308 (Permanent Redirect), 自己加一个 handler
    class _Redirect308(urllib.request.HTTPRedirectHandler):
        def http_error_308(self, req, fp, code, msg, headers):
            return self.http_error_302(req, fp, code, msg, headers)
    opener = urllib.request.build_opener(_Redirect308())

    try:
        with opener.open(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0)) + resume
            mode = "ab" if resume > 0 else "wb"
            downloaded = resume
            t0 = time.time()
            last_print = 0.0
            with open(tmp, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print > 0.5:
                        _print_progress(dest.name, downloaded, total, now - t0)
                        last_print = now
            _print_progress(dest.name, downloaded, total, time.time() - t0, end=True)
        tmp.rename(dest)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 416 and tmp.exists() and expected_size and tmp.stat().st_size >= expected_size:
            # 范围错误但本地已经完整, 直接 rename
            tmp.rename(dest)
            return True
        print(f"  {RED}HTTP {e.code}{RESET} {url}")
        return False
    except Exception as e:
        print(f"  {RED}下载失败{RESET}: {type(e).__name__}: {e}")
        return False


def _print_progress(name: str, cur: int, total: int, elapsed: float, end: bool = False):
    if total <= 0:
        return
    pct = cur * 100 / total
    speed = cur / max(elapsed, 0.01) / (1 << 20)  # MB/s
    eta = (total - cur) / max(speed * (1 << 20), 1)
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    msg = (f"  [{bar}] {pct:5.1f}% "
           f"{cur/(1<<20):7.1f}/{total/(1<<20):.1f} MB "
           f"{speed:5.1f} MB/s ETA {int(eta):3d}s  {name[:40]}")
    sys.stdout.write("\r" + msg + " " * 5)
    if end:
        sys.stdout.write("\n")
    sys.stdout.flush()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_item(item: dict, mirror_pref: str) -> bool:
    dest = ROOT / item["path"]
    if dest.exists() and dest.stat().st_size > 0:
        # 已存在: 可选校验
        expected = item.get("sha256")
        if expected:
            if sha256_file(dest) == expected:
                print(f"{GREEN}[已存在]{RESET} {item['path']}")
                return True
            print(f"{YELLOW}[校验失败, 重下]{RESET} {item['path']}")
            dest.unlink()
        else:
            print(f"{GREEN}[已存在]{RESET} {item['path']}")
            return True

    mirrors = item.get("mirrors", [])
    if not mirrors:
        print(f"{RED}[无镜像]{RESET} {item['path']}")
        return False

    # 按用户偏好排序: 主选镜像在前
    pref_order = [mirror_pref, "hf-mirror", "huggingface", "github"]
    seen = set()
    ordered = []
    for p in pref_order:
        for m in mirrors:
            if m in seen:
                continue
            if p in m or "{mirror}" in m:
                ordered.append((p, m))
                seen.add(m)
    for m in mirrors:
        if m not in seen:
            ordered.append((mirror_pref, m))
            seen.add(m)

    expected_bytes = int(item.get("size_mb", 0) * 1024 * 1024) if item.get("size_mb") else None
    for label, template in ordered:
        url = resolve_url(template, label)
        print(f"  尝试 {DIM}{label}{RESET}: {url}")
        if download_with_resume(url, dest, expected_bytes):
            # SHA256 校验
            expected = item.get("sha256")
            if expected:
                actual = sha256_file(dest)
                if actual != expected:
                    print(f"{RED}  SHA256 不匹配, 删除重试{RESET}")
                    dest.unlink()
                    continue
            return True
    return False


def download_pack(pack_name: str, manifest: dict, max_workers: int = 2) -> bool:
    items = manifest["packs"].get(pack_name)
    if not items:
        print(f"{RED}未知包名: {pack_name}{RESET}")
        return False
    mirror = get_preferred_mirror()
    print(f"\n{CYAN}=== 下载包: {pack_name} ({len(items)} 个文件) "
          f"主源: {mirror} ==={RESET}")
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(download_item, it, mirror): it for it in items}
        for fut in concurrent.futures.as_completed(futs):
            it = futs[fut]
            try:
                ok = fut.result()
            except Exception as e:
                ok = False
                print(f"{RED}异常{RESET}: {it['path']}: {e}")
            if not ok:
                failed.append(it["path"])
    if failed:
        print(f"\n{RED}失败 {len(failed)} 个:{RESET}")
        for f in failed:
            print(f"  - {f}")
        return False
    print(f"\n{GREEN}✓ 包 [{pack_name}] 全部下载完成{RESET}")
    return True


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def verify_pack(pack_name: Optional[str], manifest: dict) -> bool:
    packs = (
        [(pack_name, manifest["packs"][pack_name])]
        if pack_name else list_packs(manifest)
    )
    all_ok = True
    for name, items in packs:
        print(f"\n{CYAN}=== 校验 {name} ==={RESET}")
        for it in items:
            p = ROOT / it["path"]
            if not p.exists():
                print(f"  {YELLOW}[缺失]{RESET} {it['path']}")
                all_ok = False
                continue
            expected = it.get("sha256")
            if not expected:
                print(f"  {DIM}[无哈希]{RESET} {it['path']}")
                continue
            actual = sha256_file(p)
            if actual == expected:
                print(f"  {GREEN}[OK]{RESET} {it['path']}")
            else:
                print(f"  {RED}[损坏]{RESET} {it['path']}")
                all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# 菜单
# ---------------------------------------------------------------------------
def menu():
    manifest = load_manifest()
    packs = list_packs(manifest)
    mirror = get_preferred_mirror()

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(CYAN + "=" * 60 + RESET)
        print(CYAN + "  VisoMaster 模型下载工具" + RESET)
        print(CYAN + "=" * 60 + RESET)
        print(f"\n  当前主镜像: {mirror}\n")
        for i, (name, items) in enumerate(packs, 1):
            ok, total, mb = pack_status(items)
            if ok == total:
                tag = f"{GREEN}[完整]{RESET}"
            elif ok == 0:
                tag = f"{RED}[缺失]{RESET}"
            else:
                tag = f"{YELLOW}[{ok}/{total}]{RESET}"
            desc = manifest["packs_meta"].get(name, {}).get("desc", "")
            print(f"  [{i}] {name:<14} {mb:>5} MB  {tag}  {desc}")
        print(f"\n  [a] 下载全部缺失")
        print(f"  [v] 校验已下载文件 (SHA256)")
        print(f"  [m] 切换镜像 (hf-mirror / huggingface / github)")
        print(f"  [0] 退出\n")
        choice = input("  请选择: ").strip().lower()
        if choice == "0":
            return
        if choice == "v":
            verify_pack(None, manifest)
            input("\n按回车继续...")
            continue
        if choice == "m":
            new = input(f"  输入镜像名 (当前 {mirror}): ").strip()
            if new in ("hf-mirror", "huggingface", "github"):
                cfg = load_config()
                if "download" not in cfg:
                    cfg["download"] = {}
                cfg["download"]["mirror"] = new
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    cfg.write(f)
                mirror = new
            continue
        if choice == "a":
            for name, _ in packs:
                download_pack(name, manifest)
            input("\n按回车继续...")
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(packs):
                download_pack(packs[idx][0], manifest)
                input("\n按回车继续...")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VisoMaster 模型管理器")
    ap.add_argument("--check", metavar="PACK", help="检查指定包是否齐全")
    ap.add_argument("--download", metavar="PACK", help="下载指定包")
    ap.add_argument("--verify", metavar="PACK", nargs="?", const="*", help="SHA256 校验")
    ap.add_argument("--list", action="store_true", help="列出所有包")
    ap.add_argument("--menu", action="store_true", help="交互菜单")
    ap.add_argument("--quiet", action="store_true", help="静默 (仅 exit code)")
    args = ap.parse_args()

    manifest = load_manifest()

    if args.check:
        items = manifest["packs"].get(args.check)
        if not items:
            if not args.quiet:
                print(f"未知包: {args.check}")
            sys.exit(2)
        ok, total, _ = pack_status(items)
        if not args.quiet:
            print(f"{args.check}: {ok}/{total}")
        sys.exit(0 if ok == total else 1)

    if args.download:
        ok = download_pack(args.download, manifest)
        sys.exit(0 if ok else 1)

    if args.verify:
        pack = None if args.verify == "*" else args.verify
        ok = verify_pack(pack, manifest)
        sys.exit(0 if ok else 1)

    if args.list:
        for name, items in list_packs(manifest):
            ok, total, mb = pack_status(items)
            print(f"  {name:<14} {ok}/{total} ({mb} MB)")
        sys.exit(0)

    if args.menu or len(sys.argv) == 1:
        menu()
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)

# xvitop — Intel XPU (Arc) GPU 监控工具

类似 [nvitop](https://github.com/XuehaiPan/nvitop) 的 Intel GPU 资源监控 TUI，
专为 `xe` 驱动下的 Intel Arc 系列（如 Arc Pro B65 / B580）设计，无需 root。

## 功能

| 面板 | 数据源 |
|------|--------|
| GPU 主表：显存占用条、内存忙碌率、功耗条、核心/显存温度、频率、风扇、节流原因 | `xpu-smi`（Intel 官方 XPU System Management Interface）+ sysfs + hwmon |
| **ACT\*** 估算利用率条 | 功耗相对滚动空闲基线的自校准估算（见下方说明） |
| 每 GT 详情：gt0(rc)/gt1(mc) 频率、节流、空闲状态 | `/sys/class/drm/card*/device/tile*/gt*` |
| 进程表：每个进程的 GPU 显存、CPU%、RSS | `xpu-smi ps -j` + `/proc` |
| LLM 服务面板（vLLM）：KV cache 使用率、running/waiting 请求、gen/prompt tok/s、时延、prefix cache 命中率 | vLLM Prometheus `/metrics`（自动从 `vllm serve --port` 探测） |
| 系统面板：CPU、内存、load、uptime | `/proc` |

## 关于 ACT\*（估算利用率）

部分 Arc SKU / 固件 + `xe` 驱动组合（如 Arc Pro B65）下，Level Zero 的
metrics group 不可用，内核/驱动**无法上报真实的引擎利用率百分比**
（`xpu-smi` 会显示 N/A 并报错 `ZE_RESULT_ERROR_UNKNOWN`）。
xvitop 用**功耗法**估算：追踪最近 ~3 分钟内的最低功耗作为空闲基线，
`ACT = (当前功耗 − 空闲基线) / (功耗上限 − 空闲基线)`。
空闲时归零，推理负载下随功耗上升。它是估算值（标 `\*`），
若你的机器能拿到真实利用率，以驱动为准。

## 安装（无需 root）

```bash
# 1) 依赖: rich
sudo apt install python3-rich        # 或: pip install rich

# 2) xpu-smi（Intel 官方，单二进制）
cd /tmp
curl -sL -O https://github.com/intel/xpumanager/releases/download/v2.1.0/\
xpu-smi_2.1.0+26.33.6468cec-1.24.04_amd64.deb
sudo apt install ./xpu-smi_*.deb                 # 有 root 时
# 无 root 时：
dpkg-deb -x xpu-smi_*.deb ~/bin
apt-get download libhwloc15 && dpkg-deb -x libhwloc15_*.deb ~/lib

# 3) 本工具
mkdir -p ~/xvitop && cp xvitop.py xvitop ~/xvitop/ && chmod +x ~/xvitop/xvitop
export PATH="$HOME/xvitop:$PATH"                 # 写入 ~/.bashrc
```

xvitop 自动探测 xpu-smi 路径（`$XPU_SMI`、`/usr/bin/xpu-smi`、
`~/bin/usr/bin/xpu-smi` 等）与 `libhwloc.so.15` 所在目录，无需额外配置。

## 使用

```bash
xvitop            # 交互 TUI
xvitop --once     # 单帧快照（纯文本，适合脚本 / 远程 ssh）
xvitop --json     # 单帧 JSON
xvitop --watch 2  # 纯文本循环刷新（无 TUI）
xvitop --vllm off # 关闭 vLLM 面板
xvitop --vllm http://127.0.0.1:8000
```

TUI 快捷键：`q` 退出 · `i`/`s` 刷新间隔 ±0.5s · `p` 进程表 ·
`l` LLM 面板 · `y` 系统面板 · `1`/`2` 按显存/CPU 排序 · `h` 快捷键提示

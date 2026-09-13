#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xvitop — Intel XPU (Arc) GPU monitor, an nvitop-style tool for Linux.

Data sources (all optional, degrades gracefully when absent):
  * xpu-smi  (Intel XPU System Management Interface, Level Zero based)
      - VRAM used/total, power draw/limit, clocks, fan
      - per-process GPU memory (xpu-smi ps)
  * sysfs    (/sys/class/drm/card*/device/tile*/gt*/  or legacy /gt/gt*/)
      - per-GT frequency, throttle reasons, GT idle state
  * hwmon    (/sys/class/hwmon/*, name = xe | i915)
      - core / VRAM temperature, fan speed, energy (power cross-check)
  * /proc    - CPU usage, RAM, load, per-process CPU% / RSS
  * vLLM     - Prometheus /metrics of a local vLLM OpenAI-compatible server
      (auto-detected from running vllm processes, or --vllm URL)

Notes:
  * On some Arc SKUs / firmware + xe driver combinations (e.g. Arc Pro B65),
    the Level Zero metrics group is unavailable, so the driver cannot report
    real per-engine utilization %. xvitop then shows an ESTIMATED activity
    bar derived from power draw relative to a rolling idle baseline
    (marked with '*').
  * Requires Python 3.9+ and the 'rich' package (apt: python3-rich, or pip install rich).

Usage:
  xvitop              # interactive TUI
  xvitop --once       # single snapshot, plain text (good for scripts / ssh)
  xvitop --json       # single snapshot as JSON
  xvitop --watch 2    # plain-text refresh loop, no TUI
  xvitop --vllm off   # disable the vLLM panel
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pwd
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "xvitop requires the 'rich' package.\n"
        "  Ubuntu/Debian: apt install python3-rich\n"
        "  conda/pip:     pip install rich\n"
    )
    sys.exit(1)

VERSION = "0.1.0"
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE_KIB = 4  # x86_64 page size in KiB (statm uses pages)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def fnum(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default


def read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def read_str(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def pct_color(p: Optional[float]) -> str:
    if p is None:
        return "white"
    if p < 60:
        return "green"
    if p < 85:
        return "yellow"
    return "red"


def temp_color(t: Optional[float]) -> str:
    if t is None:
        return "white"
    if t < 60:
        return "green"
    if t < 80:
        return "yellow"
    return "red"


def fmt_mib(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v / 1024:.1f}G" if v >= 1024 else f"{v:.0f}M"


def http_get(url: str, timeout: float = 2.0) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "xvitop/" + VERSION})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


# --------------------------------------------------------------------------
# xpu-smi
# --------------------------------------------------------------------------

QUERY_FIELDS = (
    "name,uuid,driver_version,vbios_version,pci.bus_id,"
    "memory.total,memory.used,memory.free,utilization.memory,"
    "temperature.gpu,temperature.memory,power.draw,power.limit,"
    "clocks.current.graphics,clocks.max.graphics,"
    "clocks.current.media,clocks.max.media,fan.speed"
)


def _xpu_smi_candidates() -> List[str]:
    cands: List[str] = []
    env = os.environ.get("XPU_SMI")
    if env:
        cands.append(env)
    cands += [
        "/usr/bin/xpu-smi",
        "/usr/local/bin/xpu-smi",
        os.path.expanduser("~/.local/bin/xpu-smi"),
        os.path.expanduser("~/bin/usr/bin/xpu-smi"),
        os.path.expanduser("~/xpu-smi/bin/xpu-smi"),
        "/opt/intel/xpu-smi/bin/xpu-smi",
    ]
    w = shutil.which("xpu-smi")
    if w:
        cands.append(w)
    return cands


def find_xpu_smi() -> Optional[str]:
    for c in _xpu_smi_candidates():
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def xpu_smi_libpath(binary: str) -> List[str]:
    """Candidate LD_LIBRARY_PATH dirs that provide libhwloc.so.15 etc."""
    home = os.path.expanduser("~")
    cands = [
        f"{home}/lib/usr/lib/x86_64-linux-gnu",
        f"{home}/lib",
        f"{home}/.local/lib",
        "/usr/local/lib",
        os.path.join(os.path.dirname(os.path.dirname(binary)), "lib"),
    ]
    out = []
    for c in cands:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "libhwloc.so.15")):
            out.append(c)
    return out


class XpuSmi:
    def __init__(self, path: str):
        self.path = path
        self.libs = xpu_smi_libpath(path)
        self.error: Optional[str] = None

    def _run(self, args: List[str], timeout: float = 8.0):
        env = dict(os.environ)
        if self.libs:
            env["LD_LIBRARY_PATH"] = ":".join(self.libs) + (
                ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
            )
        try:
            p = subprocess.run(
                [self.path] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return 1, "", "timeout"
        return p.returncode, p.stdout, p.stderr

    def query_gpu(self) -> List[Dict[str, str]]:
        rc, out, err = self._run(
            ["--query-gpu", QUERY_FIELDS, "--format=csv"]
        )
        self.error = None
        if rc != 0 and not out.strip():
            self.error = (err or "query failed").strip()[:200]
            return []
        text = out
        header_idx = None
        for i, line in enumerate(text.splitlines()):
            if line.startswith("name,"):
                header_idx = i
                break
        if header_idx is None:
            self.error = "no header in xpu-smi output"
            return []
        rows = list(csv.reader(io.StringIO("\n".join(text.splitlines()[header_idx:]))))
        if len(rows) < 2:
            return []
        # header names may carry units, e.g. "memory.total (MiB)"
        hdr = [h.split(" (")[0].strip() for h in rows[0]]
        return [dict(zip(hdr, r)) for r in rows[1:] if r]

    def ps(self) -> List[Dict[str, Any]]:
        rc, out, err = self._run(["ps", "-j"])
        if rc != 0 or not out.strip():
            return []
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return []
        lst = data.get("device_util_by_proc_list", [])
        out_list = []
        for x in lst:
            out_list.append(
                {
                    "pid": x.get("process_id"),
                    "name": x.get("process_name", "?"),
                    "device": x.get("device_id", 0),
                    "gpu_mem_kib": x.get("mem_size", 0) or 0,
                    "shared_mem_kib": x.get("shared_mem_size", 0) or 0,
                    "engines": x.get("engines", 0),
                }
            )
        return out_list


# --------------------------------------------------------------------------
# sysfs: per-GT frequency / throttle / idle
# --------------------------------------------------------------------------

GT_REASON_FILES = [
    "pl1", "pl2", "pl4", "vr_tdc", "ratl", "prochot", "thermal", "vr_thermalert",
]


@dataclass
class GtInfo:
    tile: int
    gt: int
    label: str
    cur: Optional[int] = None
    max: Optional[int] = None
    throttle: List[str] = field(default_factory=list)
    idle: str = ""

    def as_dict(self):
        return {
            "tile": self.tile, "gt": self.gt, "label": self.label,
            "cur_mhz": self.cur, "max_mhz": self.max,
            "throttle": self.throttle, "idle": self.idle,
        }


def find_drm_cards() -> List[str]:
    d = "/sys/class/drm"
    out = []
    if not os.path.isdir(d):
        return out
    for e in sorted(os.listdir(d)):
        if not re.fullmatch(r"card\d+", e):
            continue
        drv = os.path.realpath(os.path.join(d, e, "device", "driver"))
        if os.path.basename(drv) in ("xe", "i915"):
            out.append(e)
    return out


def collect_gts(card: str) -> List[GtInfo]:
    gts: List[GtInfo] = []
    dev = os.path.join("/sys/class/drm", card, "device")
    if os.path.isdir(dev):
        for tile in sorted(os.listdir(dev)):
            if not re.fullmatch(r"tile\d+", tile):
                continue
            tdir = os.path.join(dev, tile)
            for gt in sorted(os.listdir(tdir)):
                if not re.fullmatch(r"gt\d+", gt):
                    continue
                gdir = os.path.join(tdir, gt)
                gts.append(_read_gt(gdir, int(tile[4:]), int(gt[2:])))
    legacy = os.path.join("/sys/class/drm", card, "gt")
    if os.path.isdir(legacy):  # legacy i915 layout
        for gt in sorted(os.listdir(legacy)):
            if not re.fullmatch(r"gt\d+", gt):
                continue
            gdir = os.path.join(legacy, gt)
            g = _read_gt(gdir, -1, int(gt[2:]))
            if g.cur is not None or g.max is not None:
                gts.append(g)
    return gts


def _read_gt(gdir: str, tile: int, gt: int) -> GtInfo:
    label = read_str(os.path.join(gdir, "gtidle", "name")) or f"gt{gt}"
    thr: List[str] = []
    for r in GT_REASON_FILES:
        if read_int(os.path.join(gdir, "freq0", "throttle", f"reason_{r}")):
            thr.append(r)
    idle = read_str(os.path.join(gdir, "gtidle", "idle_status")) or ""
    cur = read_int(os.path.join(gdir, "freq0", "cur_freq"))
    mx = read_int(os.path.join(gdir, "freq0", "max_freq"))
    if cur is None:  # legacy i915
        cur = read_int(os.path.join(gdir, "rps_cur_freq_mhz"))
        if cur is None:
            c = read_int(os.path.join(gdir, "rps_cur_freq"))
            cur = c // 100 if c is not None else None
        mx = read_int(os.path.join(gdir, "rps_max_freq_mhz"))
        if mx is None:
            m = read_int(os.path.join(gdir, "rps_max_freq"))
            mx = m // 100 if m is not None else None
    return GtInfo(tile=tile, gt=gt, label=label, cur=cur, max=mx,
                  throttle=thr, idle=idle)


# --------------------------------------------------------------------------
# hwmon: temps / fan / energy
# --------------------------------------------------------------------------

@dataclass
class HwmonData:
    hwmon: Optional[str] = None
    card: Optional[str] = None
    core_temp: Optional[float] = None
    vram_temp: Optional[float] = None
    fan: Optional[int] = None
    energy_card_uj: Optional[int] = None
    power_cap_w: Optional[float] = None


def collect_hwmon() -> HwmonData:
    d = "/sys/class/hwmon"
    info = HwmonData()
    if not os.path.isdir(d):
        return info
    for e in sorted(os.listdir(d)):
        name = read_str(os.path.join(d, e, "name"))
        if name not in ("xe", "i915", "drm-intel"):
            continue
        info.hwmon = e
        hdir = os.path.join(d, e)
        # map to drm card via device symlink
        try:
            tgt = os.readlink(os.path.join(hdir, "device"))
            m = re.search(r"(card\d+)", tgt)
            if m:
                info.card = m.group(1)
        except OSError:
            pass
        unlabeled_temp = False
        for f in sorted(os.listdir(hdir)):
            base = f.rsplit("_", 1)[0]
            val = read_int(os.path.join(hdir, f))
            if val is None:
                continue
            if f.startswith("temp") and f.endswith("_input"):
                t = val / 1000.0
                label = (read_str(os.path.join(hdir, f[:-6] + "_label")) or "").lower()
                if "vram" in label or "mem" in label:
                    info.vram_temp = t
                elif "core" in label or "pkg" in label or "gt" in label:
                    info.core_temp = t
                elif not unlabeled_temp:
                    info.core_temp = t
                    unlabeled_temp = True
            elif f.startswith("fan") and f.endswith("_input"):
                info.fan = val
            elif f.startswith("energy") and f.endswith("_input"):
                label = (read_str(os.path.join(hdir, f[:-6] + "_label")) or "").lower()
                if label == "card" or label == "":
                    info.energy_card_uj = val
            elif f == "power1_cap":
                info.power_cap_w = val / 1e6
        return info
    return info


class EnergyPower:
    """Instant power (W) from a cumulative energy counter (micro-Joules)."""

    def __init__(self):
        self._last: Optional[tuple] = None

    def power(self, energy_uj: Optional[int]) -> Optional[float]:
        if energy_uj is None:
            return None
        now = time.time()
        if self._last is None:
            self._last = (now, energy_uj)
            return None
        t0, e0 = self._last
        self._last = (now, energy_uj)
        dt = now - t0
        if dt <= 0:
            return None
        p = (energy_uj - e0) / 1e6 / dt
        return p if p >= 0 else None


# --------------------------------------------------------------------------
# activity estimate (power based)
# --------------------------------------------------------------------------

class ActivityEst:
    """Estimated GPU activity from power draw vs a rolling idle baseline.

    baseline = minimum power seen in the window; activity = (P - base)/(cap - base).
    """

    def __init__(self, maxlen: int = 90):
        self.buf: List[float] = []
        self.maxlen = maxlen

    def update(self, draw: Optional[float], cap: Optional[float]) -> Optional[float]:
        if draw is None or cap is None or cap <= 0:
            return None
        self.buf.append(draw)
        if len(self.buf) > self.maxlen:
            self.buf = self.buf[-self.maxlen:]
        lo = min(self.buf)
        hi = cap
        if hi <= lo:
            return 100.0
        return clamp(100.0 * (draw - lo) / (hi - lo), 0.0, 100.0)


# --------------------------------------------------------------------------
# /proc: processes + system
# --------------------------------------------------------------------------

GENERIC_NAMES = {"python", "python3", "node", "npm", "java", "bash", "sh"}


def _cmdline_name(pid: int, comm: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            toks = f.read().decode("utf-8", "replace").split("\0")
    except OSError:
        return comm
    toks = [t for t in toks if t]
    if len(toks) >= 2 and comm in GENERIC_NAMES:
        return os.path.basename(toks[1])
    return comm


class ProcTracker:
    def __init__(self):
        self.prev: Dict[tuple, tuple] = {}

    def snapshot(self) -> Dict[int, Dict[str, Any]]:
        now = time.time()
        out: Dict[int, Dict[str, Any]] = {}
        for e in os.listdir("/proc"):
            if not e.isdigit():
                continue
            pid = int(e)
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat = f.read()
            except OSError:
                continue
            m = re.match(r"^\s*(\d+)\s+\((.+?)\)\s+(\S+)\s+", stat)
            if not m:
                continue
            comm = m.group(2)
            rem = stat[m.end():].split()
            if len(rem) < 20:
                continue
            try:
                utime = int(rem[10])
                stime = int(rem[11])
                start = rem[18]
                rss_pages = int(rem[20])
            except (IndexError, ValueError):
                continue
            key = (pid, start)
            cpu: Optional[float] = None
            if key in self.prev:
                t0, u0, s0 = self.prev[key]
                dt = now - t0
                if dt > 0:
                    cpu = clamp(100.0 * ((utime + stime) - (u0 + s0)) / CLK_TCK / dt,
                                0.0, 100.0 * max(1, os.cpu_count() or 1))
            self.prev[key] = (now, utime, stime)
            if len(self.prev) > 8192:
                # keep latest per pid
                best: Dict[int, tuple] = {}
                for k, v in self.prev.items():
                    best[k[0]] = v
                self.prev = {k: v for k, v in best.items()}
            user = "?"
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("Uid:"):
                            user = pwd.getpwuid(int(line.split()[1])).pw_name
                            break
            except Exception:
                pass
            out[pid] = {
                "pid": pid,
                "name": _cmdline_name(pid, comm),
                "user": user,
                "cpu": cpu,
                "rss_kib": rss_pages * PAGE_KIB,
            }
        return out


def read_system() -> Dict[str, Any]:
    sysinfo: Dict[str, Any] = {"cpu": None, "cpu_count": os.cpu_count(),
                               "ram_total_kib": None, "ram_used_kib": None,
                               "load1": None, "load5": None, "load15": None,
                               "uptime": None}
    # meminfo
    mi = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                mi[k.strip()] = int(v.split()[0])
    except OSError:
        pass
    if "MemTotal" in mi:
        sysinfo["ram_total_kib"] = mi["MemTotal"]
        avail = mi.get("MemAvailable")
        if avail is None:
            avail = mi.get("MemFree", 0) + mi.get("Buffers", 0) + mi.get("Cached", 0)
        sysinfo["ram_used_kib"] = mi["MemTotal"] - avail
    # load / uptime
    try:
        with open("/proc/loadavg") as f:
            l1, l5, l15, _ = f.read().split()[:4]
        sysinfo["load1"], sysinfo["load5"], sysinfo["load15"] = float(l1), float(l5), float(l15)
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/uptime") as f:
            sysinfo["uptime"] = float(f.read().split()[0])
    except (OSError, ValueError):
        pass
    # cpu jiffies
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        return sysinfo, (total, idle)
    except (OSError, ValueError):
        return sysinfo, None


class CpuTracker:
    def __init__(self):
        self.prev: Optional[tuple] = None

    def update(self, pair: Optional[tuple]) -> Optional[float]:
        if pair is None:
            return None
        total, idle = pair
        if self.prev is None:
            self.prev = pair
            return None
        dt = total - self.prev[0]
        di = idle - self.prev[1]
        self.prev = pair
        if dt <= 0:
            return None
        return clamp(100.0 * (1.0 - di / dt), 0.0, 100.0)


# --------------------------------------------------------------------------
# vLLM (Prometheus /metrics)
# --------------------------------------------------------------------------

def parse_prom(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        k, _, v = line.rpartition(" ")
        if not _:
            continue
        try:
            val = float(v)
        except ValueError:
            continue
        name = k.split("{", 1)[0]
        out[name] = out.get(name, 0.0) + val
    return out


def detect_vllm_url() -> Optional[str]:
    """Find a local vllm serve process and its --port."""
    try:
        pids = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return None
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                toks = f.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if not any("vllm" in t.lower() for t in toks):
            continue
        for i, t in enumerate(toks):
            if t == "--port" and i + 1 < len(toks):
                return f"http://127.0.0.1:{toks[i + 1]}"
            if t.startswith("--port="):
                return f"http://127.0.0.1:{t.split('=', 1)[1]}"
    return None


class VllmMonitor:
    def __init__(self, url: Optional[str], enabled: bool = True):
        self.enabled = enabled
        self.url: Optional[str] = url
        self.model: Optional[str] = None
        self._model_cache_at: float = 0.0
        self.online: Optional[bool] = None
        self.fail_streak = 0
        self.prev: Dict[str, tuple] = {}  # name -> (t, value)
        self.data: Dict[str, Any] = {}

    def _refresh_model(self, force: bool = False):
        if not self.url:
            return
        now = time.time()
        if not force and now - self._model_cache_at < 30:
            return
        self._model_cache_at = now
        txt = http_get(self.url + "/v1/models", timeout=2.0)
        if txt:
            try:
                data = json.loads(txt).get("data", [])
                if data:
                    self.model = str(data[0].get("id", "?"))
            except (json.JSONDecodeError, AttributeError):
                pass

    def tick(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        if not self.url:
            self.url = detect_vllm_url() or "http://127.0.0.1:8000"
        text = http_get(self.url + "/metrics", timeout=2.5)
        if text is None:
            self.fail_streak += 1
            if self.fail_streak >= 3:
                self.online = False
            self.data = {"online": bool(self.online), "url": self.url,
                         "model": self.model, "error": "unreachable"}
            return self.data
        self.fail_streak = 0
        self.online = True
        self._refresh_model()
        m = parse_prom(text)
        now = time.time()

        def rate(name: str) -> Optional[float]:
            cur = m.get(name)
            if cur is None:
                return None
            prev = self.prev.get(name)
            r = None
            if prev is not None:
                dt = now - prev[0]
                if dt > 0 and cur >= prev[1]:
                    r = (cur - prev[1]) / dt
            self.prev[name] = (now, cur)
            return r

        def avg(name_sum: str, name_count: str) -> Optional[float]:
            s, c = m.get(name_sum), m.get(name_count)
            if s is None or not c:
                return None
            return s / c

        prefix_hits = m.get("vllm:prefix_cache_hits_total", 0.0)
        prefix_q = m.get("vllm:prefix_cache_queries_total", 0.0)
        self.data = {
            "online": True,
            "url": self.url,
            "model": self.model,
            "kv_pct": m.get("vllm:kv_cache_usage_perc"),
            "running": m.get("vllm:num_requests_running"),
            "waiting": m.get("vllm:num_requests_waiting"),
            "gen_tps": rate("vllm:generation_tokens_total"),
            "prompt_tps": rate("vllm:prompt_tokens_total"),
            "e2e_avg_s": avg("vllm:e2e_request_latency_seconds_sum",
                             "vllm:e2e_request_latency_seconds_count"),
            "itl_avg_s": avg("vllm:inter_token_latency_seconds_sum",
                             "vllm:inter_token_latency_seconds_count"),
            "ttft_avg_s": avg("vllm:time_to_first_token_seconds_sum",
                              "vllm:time_to_first_token_seconds_count"),
            "prefix_hits": prefix_hits,
            "prefix_queries": prefix_q,
            "prefix_hit_pct": (100.0 * prefix_hits / prefix_q) if prefix_q else None,
        }
        return self.data


# --------------------------------------------------------------------------
# sampler: merge everything into a Snapshot
# --------------------------------------------------------------------------

@dataclass
class Snapshot:
    ts: float
    gpus: List[Dict[str, Any]] = field(default_factory=list)
    procs: List[Dict[str, Any]] = field(default_factory=list)
    system: Dict[str, Any] = field(default_factory=dict)
    vllm: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpus": self.gpus,
            "procs": self.procs,
            "system": self.system,
            "vllm": self.vllm,
            "sources": self.sources,
        }


class Sampler:
    def __init__(self, xsmi: Optional[XpuSmi], vllm: VllmMonitor):
        self.xsmi = xsmi
        self.vllm = vllm
        self.hw = EnergyPower()
        self.procs = ProcTracker()
        self.cpu = CpuTracker()
        self.act: Dict[int, ActivityEst] = {}
        self.cards = find_drm_cards()
        self.last_gts: Dict[str, List[GtInfo]] = {}
        self.last_hw: Optional[HwmonData] = None

    def sample(self) -> Snapshot:
        t0 = time.time()
        snap = Snapshot(ts=t0)
        sources: Dict[str, str] = {}

        # --- xpu-smi (query + ps in parallel) ---
        raw_gpus: List[Dict[str, str]] = []
        gpu_procs: List[Dict[str, Any]] = []
        if self.xsmi:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(self.xsmi.query_gpu)
                f2 = ex.submit(self.xsmi.ps)
                raw_gpus = f1.result()
                gpu_procs = f2.result()
            if raw_gpus:
                sources["xpu-smi"] = "ok"
            else:
                sources["xpu-smi"] = f"error: {self.xsmi.error or 'no data'}"
        else:
            sources["xpu-smi"] = "not found"

        # --- sysfs / hwmon ---
        gts: Dict[str, List[GtInfo]] = {}
        for card in self.cards:
            gts[card] = collect_gts(card)
        self.last_gts = gts
        sources["sysfs"] = "ok" if self.cards else "n/a"
        hw = collect_hwmon()
        self.last_hw = hw
        hw_power = self.hw.power(hw.energy_card_uj)
        if hw.hwmon:
            sources["hwmon"] = "ok"
        else:
            sources["hwmon"] = "n/a"

        # --- system ---
        sysinfo, cpu_pair = read_system()
        sysinfo["cpu"] = self.cpu.update(cpu_pair)
        snap.system = sysinfo

        # --- vLLM ---
        snap.vllm = self.vllm.tick()

        # --- merge per GPU ---
        n = max(len(raw_gpus), len(self.cards))
        for i in range(n):
            g: Dict[str, Any] = {"index": i}
            if i < len(raw_gpus):
                r = raw_gpus[i]
                g["name"] = r.get("name", f"GPU {i}")
                g["uuid"] = r.get("uuid")
                g["bus_id"] = r.get("pci.bus_id")
                g["driver_version"] = r.get("driver_version")
                g["mem_total_mib"] = fnum(r.get("memory.total"))
                g["mem_used_mib"] = fnum(r.get("memory.used"))
                g["mem_free_mib"] = fnum(r.get("memory.free"))
                g["mem_busy_pct"] = fnum(r.get("utilization.memory"))
                g["temp_gpu_c"] = fnum(r.get("temperature.gpu"))
                g["temp_mem_c"] = fnum(r.get("temperature.memory"))
                g["power_w"] = fnum(r.get("power.draw"))
                g["power_cap_w"] = fnum(r.get("power.limit"))
                g["clk_g_cur"] = fnum(r.get("clocks.current.graphics"))
                g["clk_g_max"] = fnum(r.get("clocks.max.graphics"))
                g["clk_m_cur"] = fnum(r.get("clocks.current.media"))
                g["clk_m_max"] = fnum(r.get("clocks.max.media"))
                g["fan_pct"] = fnum(r.get("fan.speed"))
            else:
                g["name"] = f"GPU {i}"
            # hwmon fallback / supplement (single GPU: only card0)
            if hw.hwmon:
                if g.get("temp_gpu_c") in (None, 0):
                    g["temp_gpu_c"] = hw.core_temp
                if g.get("temp_mem_c") in (None, 0):
                    g["temp_mem_c"] = hw.vram_temp
                if g.get("fan_pct") is None and hw.fan is not None:
                    g["fan_rpm"] = hw.fan
                if g.get("power_w") is None:
                    g["power_w"] = hw_power
                if g.get("power_cap_w") is None:
                    g["power_cap_w"] = hw.power_cap_w
            if g.get("temp_gpu_c") == 0:
                g["temp_gpu_c"] = None
            if g.get("temp_mem_c") == 0:
                g["temp_mem_c"] = None
            # GTs
            if i < len(self.cards):
                card = self.cards[i]
                g["gts"] = [x.as_dict() for x in gts.get(card, [])]
            # estimated activity
            ae = self.act.setdefault(i, ActivityEst())
            g["act_est"] = ae.update(g.get("power_w"), g.get("power_cap_w"))
            g["act_is_estimate"] = True
            snap.gpus.append(g)

        # --- processes: /proc + gpu mem ---
        pmap = self.procs.snapshot()
        merged: Dict[int, Dict[str, Any]] = {}
        for p in pmap.values():
            merged[p["pid"]] = dict(p)
            merged[p["pid"]]["gpu_mem_kib"] = 0
        for gp in gpu_procs:
            pid = gp["pid"]
            if pid in merged:
                merged[pid]["gpu_mem_kib"] = max(
                    merged[pid]["gpu_mem_kib"], gp["gpu_mem_kib"])
                if gp["name"]:
                    merged[pid]["name"] = gp["name"]
            else:
                merged[pid] = {"pid": pid, "name": gp["name"], "user": "?",
                               "cpu": None, "rss_kib": 0,
                               "gpu_mem_kib": gp["gpu_mem_kib"]}
        procs = sorted(merged.values(),
                       key=lambda p: (-p["gpu_mem_kib"], -(p["cpu"] or 0)))
        snap.procs = procs
        snap.sources = sources
        snap.elapsed = time.time() - t0  # type: ignore[attr-defined]
        return snap


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

BAR_W = 14


def bar(pct: Optional[float], width: int = BAR_W, color: Optional[str] = None) -> Text:
    if pct is None:
        return Text(" " * width, style="bright_black")
    p = clamp(pct, 0.0, 100.0)
    filled = int(round(width * p / 100.0))
    c = color or pct_color(p)
    return Text(f"{'█' * filled}{'░' * (width - filled)}", style=c)


class UI:
    def __init__(self, args: argparse.Namespace):
        self.console = Console()
        self.refresh: float = args.refresh
        self.show_procs = not args.no_processes
        self.show_llm = not args.no_llm
        self.show_system = not args.no_system
        self.show_help = True
        self.sort_key = "gpu"  # gpu | cpu
        self.xsmi_path = args.xpu_smi

    def render(self, snap: Snapshot) -> Group:
        parts: List[Any] = []
        parts.append(self._header(snap))
        parts.append(self._gpu_table(snap))
        v = snap.vllm
        if self.show_llm and v and v.get("enabled", True) and (
                v.get("online") or v.get("url")):
            parts.append(self._llm_panel(v))
        if self.show_procs and snap.procs:
            parts.append(self._proc_table(snap))
        if self.show_system:
            parts.append(self._system_panel(snap.system))
        parts.append(self._footer(snap))
        return Group(*parts)

    def _header(self, snap: Snapshot) -> Panel:
        names = ", ".join(g.get("name", f"GPU {i}") for i, g in enumerate(snap.gpus)) or "no Intel GPU found"
        src = " ".join(f"{k}={v.split(':')[0]}" for k, v in snap.sources.items())
        body = Text.assemble(
            (f"{VERSION}  ", "bold cyan"),
            (f"{time.strftime('%H:%M:%S')}   ", "bright_black"),
            (names, "bold"),
            (f"\n{src}", "dim"),
        )
        return Panel(body, title="xvitop — Intel XPU monitor",
                     title_align="left", border_style="cyan")

    def _gpu_table(self, snap: Snapshot) -> Table:
        t = Table(expand=False, pad_edge=False, header_style="bold magenta")
        for col in ("GPU", "ACT*", "MEM", "MBUS", "PWR", "TEMP", "CLOCK", "FAN", "THR"):
            t.add_column(col, justify="left" if col == "GPU" else "right")
        for g in snap.gpus:
            name = (g.get("name") or "?")
            for tok in ("Intel(R) ", "(TM) ", "(TM)", "Graphics", "Intel(R)"):
                name = name.replace(tok, "")
            name = re.sub(r"\s+", " ", name).strip()
            if len(name) > 20:
                name = name[:19] + "…"
            cells: List[Any] = [Text(f"{g['index']} {name}", style="bold")]
            act = g.get("act_est")
            if act is not None:
                cells.append(Text.assemble(
                    (f"{bar(act)} ", pct_color(act)), (f"{act:3.0f}%", pct_color(act))))
            else:
                cells.append(Text("n/a", style="bright_black"))
            mu, mt = g.get("mem_used_mib"), g.get("mem_total_mib")
            if mu is not None and mt:
                p = 100.0 * mu / mt
                cells.append(Text.assemble(
                    (f"{bar(p)} ", pct_color(p)),
                    (f"{mu / 1024:5.1f}/{mt / 1024:5.1f}G", "white")))
            else:
                cells.append(Text("n/a", style="bright_black"))
            mb = g.get("mem_busy_pct")
            cells.append(Text(f"{mb:.0f}%" if mb is not None else "n/a",
                              style=pct_color(mb) if mb is not None else "bright_black"))
            pw, cap = g.get("power_w"), g.get("power_cap_w")
            if pw is not None and cap:
                p = 100.0 * pw / cap
                cells.append(Text.assemble(
                    (f"{bar(p, 10)} ", pct_color(p)),
                    (f"{pw:4.0f}/{cap:3.0f}W", "white")))
            else:
                cells.append(Text(f"{pw:4.0f}W" if pw is not None else "n/a"))
            tg, tm = g.get("temp_gpu_c"), g.get("temp_mem_c")
            cells.append(Text.assemble(
                (f"{tg:3.0f}", temp_color(tg)) if tg is not None else ("--", "bright_black"),
                ("/", "bright_black"),
                (f"{tm:3.0f}", temp_color(tm)) if tm is not None else ("--", "bright_black")))
            cg, cgm = g.get("clk_g_cur"), g.get("clk_g_max")
            cm, cmm = g.get("clk_m_cur"), g.get("clk_m_max")
            clock = "n/a"
            if cg is not None:
                clock = f"G{cg:4.0f}/{cgm:4.0f} M{cm:4.0f}/{cmm:4.0f}" if cm is not None \
                    else f"G{cg:4.0f}/{cgm:4.0f}"
            cells.append(Text(clock, style="white" if cg is not None else "bright_black"))
            if g.get("fan_rpm") is not None:
                cells.append(Text(f"{g['fan_rpm']}rpm"))
            elif g.get("fan_pct") is not None:
                cells.append(Text(f"{g['fan_pct']:.0f}%"))
            else:
                cells.append(Text("n/a", style="bright_black"))
            thr = []
            for gt in g.get("gts", []):
                thr.extend(gt.get("throttle", []))
            cells.append(Text(", ".join(sorted(set(thr))) or "—",
                              style="red" if thr else "green"))
            t.add_row(*cells)
        out: List[Any] = [t]
        # per-GT detail lines
        for g in snap.gpus:
            gts = g.get("gts") or []
            if not gts:
                continue
            bits = []
            for gt in gts:
                f = (f"{gt['cur_mhz']:4.0f}/{gt['max_mhz']:4.0f}MHz"
                     if gt.get("cur_mhz") is not None and gt.get("max_mhz") is not None else "n/a")
                bits.append(f"{gt['label']}: {f}")
                if gt.get("throttle"):
                    bits.append(f"[thr: {','.join(gt['throttle'])}]")
                if gt.get("idle") and not gt.get("throttle"):
                    bits.append(f"[{gt['idle']}]")
            out.append(Text(f"  tile/gt  " + "  ".join(bits), style="dim"))
        return Group(*out) if len(out) > 1 else t

    def _llm_panel(self, v: Dict[str, Any]) -> Panel:
        model = v.get("model") or "?"
        title = f"LLM Service · vLLM · {v.get('url', '?')} · {model}"
        if not v.get("online"):
            return Panel(Text("unreachable", style="red"), title=title,
                         title_align="left", border_style="red")
        kv = v.get("kv_pct")
        kv_cell = Text.assemble(
            (f"{bar(kv)} ", pct_color(kv) if kv is not None else "white"),
            (f"{kv * 100:5.1f}%" if kv is not None else "  n/a ", "white")) if kv is not None else Text("n/a")
        running = v.get("running")
        waiting = v.get("waiting")
        gen = v.get("gen_tps")
        prompt = v.get("prompt_tps")
        e2e = v.get("e2e_avg_s")
        itl = v.get("itl_avg_s")
        ttft = v.get("ttft_avg_s")
        hit = v.get("prefix_hit_pct")
        hits, queries = v.get("prefix_hits"), v.get("prefix_queries")

        l1 = Text.assemble(
            ("KV cache  ", "bold"), kv_cell,
            ("    Requests  ", "bold"),
            (f"running {running:3.0f}" if running is not None else "running  n/a", "green" if (running or 0) > 0 else "white"),
            ("   waiting ", "bright_black"),
            (f"{waiting:3.0f}" if waiting is not None else "n/a", "yellow" if (waiting or 0) > 0 else "green"),
        )
        l2 = Text.assemble(
            ("Throughput  ", "bold"),
            (f"gen {gen:7.1f} tok/s" if gen is not None else "gen      n/a", "cyan"),
            ("   ", ""),
            (f"prompt {prompt:6.1f} tok/s" if prompt is not None else "prompt     n/a", "cyan"),
        )
        lat = []
        if e2e is not None:
            lat.append(f"e2e {e2e * 1000:.0f}ms")
        if ttft is not None:
            lat.append(f"ttft {ttft * 1000:.0f}ms")
        if itl is not None:
            lat.append(f"itl {itl * 1000:.0f}ms")
        l3 = Text.assemble(
            ("Latency(avg)  ", "bold"),
            (", ".join(lat) if lat else "n/a", "white"),
            ("    Prefix cache  ", "bold"),
            (f"{hit:5.1f}% ({hits / 1e6:.1f}M/{queries / 1e6:.1f}M)" if hit is not None else "n/a", "green" if (hit or 0) > 50 else "white"),
        )
        return Panel(Group(l1, l2, l3), title=title, title_align="left",
                     border_style="cyan")

    def _proc_table(self, snap: Snapshot) -> Table:
        t = Table(title="Processes (GPU memory from xpu-smi)",
                  header_style="bold magenta",
                  expand=False, pad_edge=False)
        for col in ("PID", "USER", "NAME", "GPU-MEM", "CPU%", "RSS"):
            t.add_column(col, justify="right" if col in ("PID", "GPU-MEM", "CPU%", "RSS") else "left")
        procs = list(snap.procs)
        if self.sort_key == "cpu":
            procs.sort(key=lambda p: -(p["cpu"] or 0))
        else:
            procs.sort(key=lambda p: (-p["gpu_mem_kib"], -(p["cpu"] or 0)))
        shown = 0
        for p in procs:
            if p["gpu_mem_kib"] <= 0 and (p["cpu"] or 0) < 1.0 and shown > 0:
                continue
            if shown >= 12:
                break
            shown += 1
            style = "yellow" if p["gpu_mem_kib"] == max(
                x["gpu_mem_kib"] for x in snap.procs) and p["gpu_mem_kib"] > 0 else ""
            name = p["name"]
            if len(name) > 26:
                name = name[:25] + "…"
            t.add_row(
                str(p["pid"]),
                p.get("user", "?"),
                Text(name, style=style),
                Text(f"{p['gpu_mem_kib'] / 1024 / 1024:.2f}G",
                     style=style if p["gpu_mem_kib"] > 0 else "bright_black"),
                f"{p['cpu']:.1f}" if p["cpu"] is not None else "-",
                f"{p['rss_kib'] / 1024 / 1024:.2f}G",
            )
        if shown == 0:
            t.add_row("-", "-", "no processes with GPU usage", "-", "-", "-")
        return t

    def _system_panel(self, s: Dict[str, Any]) -> Panel:
        cpu = s.get("cpu")
        cpu_cell = Text.assemble(
            (f"{bar(cpu)} ", pct_color(cpu)),
            (f"{cpu:4.1f}%" if cpu is not None else "  n/a", "white"),
            (f"  ({s.get('cpu_count', '?')} cores)", "dim")) if cpu is not None else Text("n/a")
        rt, ru = s.get("ram_total_kib"), s.get("ram_used_kib")
        if rt and ru is not None:
            p = 100.0 * ru / rt
            ram_cell = Text.assemble(
                (f"{bar(p)} ", pct_color(p)),
                (f"{ru / 1024 / 1024:5.1f}/{rt / 1024 / 1024:5.1f}G", "white"))
        else:
            ram_cell = Text("n/a")
        load = s.get("load1")
        up = s.get("uptime")
        up_s = ""
        if up:
            h, rem = divmod(int(up), 3600)
            m = rem // 60
            up_s = f"uptime {h}h {m:02d}m"
        body = Group(
            Text.assemble(("CPU   ", "bold"), cpu_cell),
            Text.assemble(("RAM   ", "bold"), ram_cell),
            Text.assemble(
                ("LOAD  ", "bold"),
                (f"{load:.2f} {s.get('load5', 0):.2f} {s.get('load15', 0):.2f}" if load is not None else "n/a", "white"),
                (("    " + up_s), "dim")),
        )
        return Panel(body, title="System", title_align="left", border_style="grey50")

    def _footer(self, snap: Snapshot) -> Panel:
        left = "q quit · i/s refresh ±0.5s · p procs · l llm · y system · 1/2 sort · h help" if self.show_help else "h show keys"
        right = f"refresh {self.refresh:.1f}s"
        return Panel(
            Text.assemble((left, "bright_black"), ("  " + right, "cyan")),
            border_style="dim")


# --------------------------------------------------------------------------
# key input (Linux TTY)
# --------------------------------------------------------------------------

class KeyReader(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__()
        self.q: "queue.Queue[str]" = queue.Queue()
        # NOTE: must not be named "_stop" — that name is used internally by
        # threading.Thread and would break Thread.join().
        self._stop_event = threading.Event()
        self._old = None
        self._restored = False

    def run(self):
        try:
            import select
            import termios
            import tty
        except ImportError:
            return
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
        try:
            self._old = termios.tcgetattr(fd)
        except termios.error:
            return
        # NOTE: read the raw fd (not sys.stdin) so this thread never holds the
        # stdin TextIOWrapper lock while blocked — that lock deadlocks the
        # interpreter's finalization (close of sys.stdin) at exit.
        # NOTE: select() poll instead of a blocking read() so stop() can wake
        # us promptly; a plain Event cannot interrupt a blocking read.
        try:
            tty.setcbreak(fd, termios.TCSANOW)
            while not self._stop_event.is_set():
                try:
                    r, _, _ = select.select([fd], [], [], 0.05)
                except (OSError, ValueError):
                    break
                if not r:
                    continue
                try:
                    data = os.read(fd, 1)
                except (OSError, ValueError):
                    break
                if data:
                    self.q.put(data.decode("utf-8", "replace"))
        finally:
            self.restore()

    def restore(self) -> None:
        """Idempotently restore the saved terminal attributes."""
        if self._old is None or self._restored:
            return
        self._restored = True
        try:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._old)
        except (termios.error, OSError, ValueError):
            pass

    def stop(self, timeout: float = 1.5) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)
        self.restore()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def make_sampler(args: argparse.Namespace) -> Sampler:
    xsmi = None
    if not args.no_xpu_smi:
        path = args.xpu_smi or find_xpu_smi()
        if path:
            xsmi = XpuSmi(path)
    vurl = None
    if args.vllm and args.vllm.lower() != "off" and args.vllm.lower() != "auto":
        vurl = args.vllm if "://" in args.vllm else "http://" + args.vllm
    vllm = VllmMonitor(vurl, enabled=not args.no_llm and (args.vllm is None or args.vllm.lower() != "off"))
    return Sampler(xsmi, vllm)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="xvitop",
        description="Intel XPU (Arc) GPU monitor — an nvitop for Intel GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else None,
    )
    ap.add_argument("--once", action="store_true", help="print a single snapshot and exit")
    ap.add_argument("--json", action="store_true", help="print a single snapshot as JSON and exit")
    ap.add_argument("--watch", type=float, metavar="SECS",
                    help="plain-text refresh loop every SECS seconds (no TUI)")
    ap.add_argument("--refresh", type=float, default=2.0, help="TUI refresh interval in seconds (default 2)")
    ap.add_argument("--vllm", default=None, metavar="URL|auto|off",
                    help="vLLM endpoint (default: auto-detect local vllm serve)")
    ap.add_argument("--xpu-smi", dest="xpu_smi", default=None, metavar="PATH",
                    help="path to the xpu-smi binary")
    ap.add_argument("--no-xpu-smi", action="store_true", help="skip xpu-smi")
    ap.add_argument("--no-processes", action="store_true", help="hide process table")
    ap.add_argument("--no-llm", action="store_true", help="hide vLLM panel")
    ap.add_argument("--no-system", action="store_true", help="hide system panel")
    ap.add_argument("--debug", action="store_true", help="print debug info to stderr")
    args = ap.parse_args(argv)

    sampler = make_sampler(args)
    if args.debug:
        print(f"xpu-smi: {sampler.xsmi.path if sampler.xsmi else 'not found'}",
              file=sys.stderr)
        print(f"drm cards: {sampler.cards}", file=sys.stderr)

    if args.json:
        snap = sampler.sample()
        print(json.dumps(snap.as_dict(), indent=2, default=str))
        return 0

    ui = UI(args)

    if args.once:
        snap = sampler.sample()
        ui.console.print(ui.render(snap))
        return 0

    if args.watch is not None:
        interval = max(0.2, args.watch)
        try:
            while True:
                snap = sampler.sample()
                ui.console.clear()
                ui.console.print(ui.render(snap))
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        return 0

    # TUI
    if not sys.stdout.isatty():
        sys.stderr.write("xvitop: no TTY for the interactive UI; use --once or --watch.\n")
        return 2
    try:
        import termios  # noqa: F401  (Linux only; on other OS fall back to no keys)
        HAVE_TTY = True
    except ImportError:
        HAVE_TTY = False

    snap = sampler.sample()  # prewarm

    # Route SIGTERM (kill / ssh disconnect) through the same cleanup path so
    # the terminal is restored on every exit, not only on KeyboardInterrupt.
    import atexit
    import signal

    def _on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    old_sigterm = None
    try:
        old_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        old_sigterm = None

    kr: Optional[KeyReader] = None
    try:
        if HAVE_TTY:
            kr = KeyReader()
            kr.start()
            atexit.register(kr.restore)  # last-resort backstop
        with Live(ui.render(snap), console=ui.console, screen=True,
                  refresh_per_second=4) as live:
            while True:
                time.sleep(max(0.2, ui.refresh))
                snap = sampler.sample()
                if kr:
                    while not kr.q.empty():
                        ch = kr.q.get()
                        if ch == "q":
                            raise KeyboardInterrupt
                        elif ch == "i":
                            ui.refresh = min(10.0, ui.refresh + 0.5)
                        elif ch == "s":
                            ui.refresh = max(0.5, ui.refresh - 0.5)
                        elif ch == "p":
                            ui.show_procs = not ui.show_procs
                        elif ch == "l":
                            ui.show_llm = not ui.show_llm
                        elif ch == "y":
                            ui.show_system = not ui.show_system
                        elif ch == "h":
                            ui.show_help = not ui.show_help
                        elif ch == "1":
                            ui.sort_key = "gpu"
                        elif ch == "2":
                            ui.sort_key = "cpu"
                live.update(ui.render(snap))
    except KeyboardInterrupt:
        pass
    finally:
        if kr:
            kr.stop()      # joins the reader thread (restores termios in its finally)
            kr.restore()   # idempotent backstop
        if old_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, old_sigterm)
            except (ValueError, OSError):
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

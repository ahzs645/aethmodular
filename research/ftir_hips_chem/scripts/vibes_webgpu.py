"""WebGPU Khatri-Rao GEMM for batched VIBES. Experiment.

WGSL has no float64, so this runs only the one product a GPU is needed for:
rows-by-p float32 weights times a fixed p-by-q matrix, for the Hessian of
``vibes_batch.fit_elbo_newton`` and the system matrix of ``solve_map_batch``
(pass ``WebGpuGemm(...)`` as their ``gemm``). Objective, gradient, linear solves
and the MAP certificate stay float64 on the host. Runs through wgpu-py, i.e. the
same WGSL and WebGPU API a browser uses, without a browser.
"""

from __future__ import annotations

import math
import time

import numpy as np
import wgpu

WGSL = """
struct Params { rows: u32, pv: u32, q: u32, first_q: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x + P.first_q;
  if (c >= P.q) { return; }
  let r = id.y;
  let wo = r * P.pv;
  let ko = c * P.pv;
  var s = 0.0;
  for (var j = 0u; j < P.pv; j++) { s += dot(w[wo + j], K[ko + j]); }
  out[r * P.q + c] = s;
}
"""


class WebGpuGemm:
    """Holds the Khatri-Rao matrices on the device; call as ``gemm(name, weights)``."""

    def __init__(self, W, rows_per_dispatch=64, cols_per_dispatch=2048):
        W = np.asarray(W, np.float32)
        self.p, self.c = W.shape
        self.rows_per_dispatch = rows_per_dispatch
        self.cols_per_dispatch = cols_per_dispatch
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.adapter_info = dict(adapter.info)
        need = ["max-storage-buffer-binding-size", "max-buffer-size"]
        self.device = adapter.request_device_sync(required_limits={k: adapter.limits[k] for k in need})
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto", compute={"module": self.device.create_shader_module(code=WGSL), "entry_point": "main"})
        self.pv = math.ceil(self.p / 4)
        i0, i1 = np.triu_indices(self.c)
        W2 = W * W
        self.mats = {}
        for name, M in (("WW", W[:, i0] * W[:, i1]),
                        ("WW2", (W[:, :, None] * W2[:, None, :]).reshape(self.p, -1)),
                        ("W2W2", W2[:, i0] * W2[:, i1])):
            q = M.shape[1]
            packed = np.zeros((q, self.pv * 4), np.float32)
            packed[:, : self.p] = M.T
            buf = self.device.create_buffer_with_data(data=packed.tobytes(), usage=wgpu.BufferUsage.STORAGE)
            self.mats[name] = (buf, q)
        self.stats = {"calls": 0, "row_columns": 0, "seconds": 0.0}
        self.wbuf = self.obuf = self.rbuf = None
        self.ubuf = self.device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    def _ensure(self, rows, q):
        wbytes = rows * self.pv * 16
        obytes = rows * q * 4
        if self.wbuf is None or self.wbuf.size < wbytes:
            self.wbuf = self.device.create_buffer(size=wbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        if self.obuf is None or self.obuf.size < obytes:
            self.obuf = self.device.create_buffer(size=obytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            self.rbuf = self.device.create_buffer(size=obytes, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)

    def _run(self, name, w):
        kbuf, q = self.mats[name]
        rows = len(w)
        self._ensure(rows, q)
        packed = np.zeros((rows, self.pv * 4), np.float32)
        packed[:, : self.p] = w
        self.device.queue.write_buffer(self.wbuf, 0, packed.tobytes())
        bg = self.device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": self.ubuf}},
            {"binding": 1, "resource": {"buffer": self.wbuf}},
            {"binding": 2, "resource": {"buffer": kbuf}},
            {"binding": 3, "resource": {"buffer": self.obuf}}])
        step = self.cols_per_dispatch
        for first in range(0, q, step):  # bounded dispatches keep the GPU watchdog happy
            self.device.queue.write_buffer(self.ubuf, 0, np.array([rows, self.pv, q, first], np.uint32).tobytes())
            enc = self.device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self.pipeline)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(math.ceil(min(step, q - first) / 32), rows, 1)
            cp.end()
            self.device.queue.submit([enc.finish()])
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_buffer(self.obuf, 0, self.rbuf, 0, rows * q * 4)
        self.device.queue.submit([enc.finish()])
        self.rbuf.map_sync(wgpu.MapMode.READ, 0, rows * q * 4)
        out = np.frombuffer(self.rbuf.read_mapped(0, rows * q * 4, copy=True), np.float32).reshape(rows, q).copy()
        self.rbuf.unmap()
        return out

    def __call__(self, name, weights):
        t = time.perf_counter()
        w = np.asarray(weights, np.float32)
        parts = [self._run(name, w[i:i + self.rows_per_dispatch])
                 for i in range(0, len(w), self.rows_per_dispatch)]
        self.stats["calls"] += 1
        self.stats["row_columns"] += len(w) * self.mats[name][1]
        self.stats["seconds"] += time.perf_counter() - t
        return np.concatenate(parts, axis=0)

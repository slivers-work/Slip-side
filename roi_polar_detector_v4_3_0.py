#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4.3.0: one closed ROI, operator review and optional machine-coordinate export.

Dependencies: numpy opencv-python-headless scipy shapely (Pillow for the UI).
No JIT, skeleton, DTW, per-image warmup or filename rules.
Topology is discovered at a bounded scale; coordinates are refined on the
original image. Pixel sampling distance is not a measurement accuracy claim.
Cartesian files contain id,x,y; polar files contain 720 headerless angular
rows id,theta,r1,...,r_wafer, with no empty padding fields. Only ROI hits
strictly inside the physical wafer are retained; the last radius is its edge.
Optional OLS_v4_3_0.py affine calibration transforms ORIGINAL image pixels to machine
coordinates. Polar theta defaults to image-right / image-counterclockwise;
machine-X angle directions are an explicit alternative. Regression changes
distance units using transformed points and the transformed bright-ring center.
Machine tracking mode caps XY steps at 0.5.
Only selected images and coordinate files are written; no JSON sidecar is saved.
Bright-ring geometry is optional and independent of the main closed ROI.
"""
from __future__ import annotations

import argparse
import csv
import copy
import datetime as dt
import json
import math
import time
import hashlib
import heapq
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import grey_opening, maximum_filter1d, minimum_filter1d, median_filter
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, Point


class DetectionError(ValueError):
    pass


@dataclass(frozen=True)
class DetectorConfig:
    polarity: str = 'white'
    expected_count: int = 1
    region_mode: str = 'auto'
    search_circle: tuple[float, float, float] | None = None
    search_polygon: tuple[tuple[float, float], ...] | None = None
    gray_min: int | None = None
    gray_max: int | None = None
    black_threshold: int | None = None
    white_threshold: int | None = None
    gap_px: float | None = None
    min_area_ratio: float = .0015
    min_area_px: float | None = None
    max_width_ratio: float = .15
    sample_step_px: float = 1.
    scan_max_side: int = 2304
    threshold_levels: int = 10
    main_area_fraction: float = .60
    illumination_correction: bool = False
    feature_contrast_gray: float | None = None
    refine_profiles: bool = True
    regularize_path: bool = True
    path_smoothing: float = 1.0

    def validate(self):
        if self.polarity not in ('black', 'white'):
            raise ValueError('灰度特征必须是 black 或 white')
        if self.region_mode not in ('auto', 'full', 'circle', 'polygon'):
            raise ValueError('搜索方式必须是 auto、full、circle 或 polygon')
        if isinstance(self.expected_count, bool) or not isinstance(self.expected_count, int) or self.expected_count not in (0, 1):
            raise ValueError('单闭环模式只支持 1 圈；旧参数 0 也按 1 圈处理')
        if (self.gray_min is None) != (self.gray_max is None):
            raise ValueError('手动灰度范围需同时填写上下限')
        if self.gray_min is not None:
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (self.gray_min, self.gray_max)) or not 0 <= self.gray_min <= self.gray_max <= 255:
                raise ValueError('灰度范围应为 0～255，下限不得大于上限')
        for key in ('black_threshold', 'white_threshold'):
            v = getattr(self, key)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 254):
                raise ValueError(key + ' 应为 1～254 的整数或 null')
        for key, low, high in (('min_area_ratio', 0, .5), ('max_width_ratio', .001, .5),
                               ('sample_step_px', .1, 20), ('main_area_fraction', .05, 1)):
            v = getattr(self, key)
            if not math.isfinite(v) or not low <= v <= high or (key == 'min_area_ratio' and v == 0):
                raise ValueError(key + ' 超出有效范围')
        if self.min_area_px is not None and (not math.isfinite(self.min_area_px) or self.min_area_px <= 0):
            raise ValueError('最小面积应大于 0')
        if self.gap_px is not None and (not math.isfinite(self.gap_px) or not 0 <= self.gap_px <= 100):
            raise ValueError('断点修补半径应为 0～100 像素')
        if not isinstance(self.scan_max_side, int) or not 512 <= self.scan_max_side <= 4000:
            raise ValueError('搜索分辨率应为 512～4000')
        if not isinstance(self.threshold_levels, int) or not 4 <= self.threshold_levels <= 24:
            raise ValueError('自适应阈值级数应为 4～24')
        if self.feature_contrast_gray is not None and (not math.isfinite(self.feature_contrast_gray) or not 1 <= self.feature_contrast_gray <= 100):
            raise ValueError('局部灰度对比应为 1～100 或自动')
        for key in ('illumination_correction', 'refine_profiles', 'regularize_path'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(key + ' 必须为布尔值')
        if not math.isfinite(self.path_smoothing) or not 0 <= self.path_smoothing <= 3:
            raise ValueError('曲线整形强度应为 0～3')
        if self.region_mode == 'polygon':
            try:
                p = np.asarray(self.search_polygon, dtype=float)
                poly = Polygon(p)
            except Exception as exc:
                raise ValueError('请先绘制至少 3 点的手动 ROI') from exc
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all() or not poly.is_valid or poly.area < 4:
                raise ValueError('手动 ROI 需闭合、无自交，且至少包含 3 个有效点')
        if self.region_mode == 'circle':
            if self.search_circle is None or len(self.search_circle) != 3:
                raise ValueError('请先绘制圆形搜索区域')
            if any(not math.isfinite(float(v)) for v in self.search_circle) or self.search_circle[2] < 2:
                raise ValueError('圆心和半径应为有限数值，半径至少 2 像素')
        return self

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError('检测参数应为 JSON 对象')
        values = dict(values)
        # Old scan stability applies to V1's 64-level tracking, not V2.
        for key in ('stable_steps', 'feature_filter', 'deduplicate_shadows', 'halo_min_coverage', 'halo_max_width_ratio'):
            values.pop(key, None)
        if values.get('polarity') == 'both':
            values['polarity'] = 'black'
        if values.get('threshold_levels', 10) > 24:
            values['threshold_levels'] = 10
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('未知参数：' + ', '.join(sorted(unknown)))
        if values.get('search_circle') is not None:
            values['search_circle'] = tuple(values['search_circle'])
        if values.get('search_polygon') is not None:
            values['search_polygon'] = tuple(tuple(p) for p in values['search_polygon'])
        return cls(**values).validate()


@dataclass(frozen=True)
class ExportConfig:
    result_image: bool = True
    polar: bool = True
    cartesian: bool = True
    coordinate_format: str = 'csv'
    origin: str = 'center'
    use_regression: bool = False
    regression_coefficients: tuple | None = None
    worktable_name: str = ''
    calibration_shape: tuple | None = None
    regression_origin: str = 'top_left'
    regression_center: tuple | None = None
    simplify_px: float | None = None
    machine_step: float | None = None
    polar_angle_reference: str = 'image'

    def validate(self, require_output=False):
        if any(not isinstance(value, bool) for value in (self.result_image, self.polar, self.cartesian, self.use_regression)):
            raise ValueError('保存内容选项应为布尔值')
        if self.coordinate_format not in ('csv', 'txt'):
            raise ValueError('坐标文件格式应为 csv 或 txt')
        if self.polar_angle_reference not in ('image', 'machine'):
            raise ValueError('极坐标角度基准应为 image 或 machine')
        if self.origin not in ('top_left', 'bottom_left', 'center', 'ring_center'):
            raise ValueError('坐标原点应为图片左上、图片中心或亮圆环拟合圆心')
        if self.simplify_px is not None and (isinstance(self.simplify_px,bool) or self.simplify_px not in (.1,.25,.5,1.)):
            raise ValueError('像素选点应为原始点、0.1、0.25、0.5 或 1 像素')
        if require_output and not any((self.result_image, self.polar, self.cartesian)):
            raise ValueError('请至少选择一种保存内容')
        if self.machine_step is not None:
            if isinstance(self.machine_step,bool) or self.machine_step!=.5:
                raise ValueError('机台选点间距应为 0.5 机台单位')
            if not self.use_regression:raise ValueError('自适应机台模式需要勾选使用转换回归方程并选择已拟合工作台')
        if self.use_regression:
            if self.regression_origin not in ('top_left', 'bottom_left', 'center', 'ring_center'):
                raise ValueError('工作台回归原点无效')
            coefficients = np.asarray(self.regression_coefficients, dtype=float)
            if coefficients.shape != (2, 3) or not np.isfinite(coefficients).all() or np.linalg.matrix_rank(coefficients[:, :2]) < 2:
                raise ValueError('请先选择工作台并生成有效的二维回归方程')
            if not self.worktable_name.strip():
                raise ValueError('使用回归转换时工作台名称不能为空')
            if self.calibration_shape is None or len(self.calibration_shape) != 2 or any(not isinstance(v, int) or v < 16 for v in self.calibration_shape):
                raise ValueError('请先登记标定图像的原图尺寸')
        return self


@dataclass
class ROIResult:
    roi_id: str
    polarity: str
    points: np.ndarray
    threshold: int
    width_quartiles: list[float]
    review: np.ndarray
    info: dict
    reviewed: np.ndarray | None = None


@dataclass
class Analysis:
    gray: np.ndarray
    rois: list[ROIResult]
    metadata: dict
    _coordinate_cache: dict = field(default_factory=dict,repr=False)
    _geometry_cache: dict = field(default_factory=dict,repr=False)

    @property
    def status(self):
        return self.metadata['status']


def load_gray(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None or min(im.shape) < 16:
        raise DetectionError('无法解码图片或图片过小：' + str(path))
    return im


def save_image(path, im):
    path = Path(path)
    ok, data = cv2.imencode(path.suffix, im)
    if not ok:
        raise IOError('图像编码失败：' + str(path))
    data.tofile(str(path))


def warmup_accelerators():
    """Compatibility only. V2 has no just-in-time compilation or warmup."""
    return None


def resample(points, step=1.):
    p = np.asarray(points, float).reshape(-1, 2)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-8]]
    if len(p) > 1 and np.linalg.norm(p[-1] - p[0]) < 1e-8:
        p = p[:-1]
    if len(p) < 3:
        raise DetectionError('闭环有效点不足')
    q = np.r_[p, p[:1]]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    locations = np.linspace(0, arc[-1], max(8, math.ceil(arc[-1] / step)), endpoint=False)
    return np.c_[np.interp(locations, arc, q[:, 0]), np.interp(locations, arc, q[:, 1])]


def _sample(gray, positions):
    p = np.asarray(positions, np.float32)
    shape = p.shape[:-1]
    if len(shape) == 1:
        p = p[:, None, :]
    # cv2.remap's dimensions must be < 32767, even for very dense CSV output.
    chunks = []
    for start in range(0, len(p), 16000):
        a = p[start:start + 16000]
        chunks.append(cv2.remap(gray, a[..., 0], a[..., 1], cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101))
    return np.concatenate(chunks).reshape(shape).astype('float32')


def _normals(p, stride=2):
    smooth = gaussian_filter1d(np.asarray(p, float), .7, axis=0, mode='wrap')
    t = np.roll(smooth, -stride, axis=0) - np.roll(smooth, stride, axis=0)
    n = np.c_[t[:, 1], -t[:, 0]]
    if cv2.contourArea(np.asarray(p, np.float32), oriented=True) < 0:
        n *= -1
    return n / np.maximum(1e-8, np.linalg.norm(n, axis=1, keepdims=True))


def _context(gray, cfg):
    h, w = gray.shape
    shape = (max(16, round(w * min(1, cfg.scan_max_side / max(h, w)))),
             max(16, round(h * min(1, cfg.scan_max_side / max(h, w)))))
    g = cv2.resize(gray, shape, interpolation=cv2.INTER_AREA)
    sy, sx = g.shape[0] / h, g.shape[1] / w
    cx, cy = (w - 1) / 2, (h - 1) / 2
    radius = .49 * min(h, w)
    if cfg.region_mode == 'circle':
        cx, cy, radius = map(float, cfg.search_circle)
        if not (0 <= cx < w and 0 <= cy < h) or radius > 2 * math.hypot(w, h):
            raise ValueError('圆形搜索范围超出图像，请重新画圆')
    if cfg.region_mode == 'polygon':
        polygon = Polygon(cfg.search_polygon)
        if not polygon.within(Polygon([(0,0),(w-1,0),(w-1,h-1),(0,h-1)])):
            raise ValueError('手动 ROI 超出图像范围')
        seed = Point(cx, cy) if polygon.contains(Point(cx, cy)) else polygon.representative_point()
        cx, cy = seed.x, seed.y
    xx = (np.arange(g.shape[1], dtype='float32') + .5) / sx - .5 - cx
    yy = (np.arange(g.shape[0], dtype='float32') + .5) / sy - .5 - cy
    distance = yy[:, None] ** 2 + xx[None, :] ** 2
    domain = np.ones(g.shape, np.uint8) if cfg.region_mode == 'full' else (distance <= radius ** 2).astype('uint8')
    if cfg.region_mode == 'polygon':
        domain = np.zeros(g.shape, np.uint8)
        corners = (np.asarray(cfg.search_polygon) + .5) * [sx, sy] - .5
        cv2.fillPoly(domain, [np.rint(corners).astype('int32')], 1)
    core = (distance <= min(radius * .56, min(h, w) * .27) ** 2) & (domain > 0)
    if np.count_nonzero(core) < 16:
        core = domain > 0
    if np.count_nonzero(core) < 16:
        raise DetectionError('搜索区域太小')
    illumination_background = None
    if cfg.illumination_correction:
        bg = cv2.GaussianBlur(g, (0, 0), max(4., min(g.shape) * .025))
        illumination_background = bg
        g = np.clip(g.astype('float32') - bg + 128, 0, 255).astype('uint8')
    oriented = g if cfg.polarity == 'white' else 255 - g
    values = oriented[core]
    background = float(np.median(values))
    # Tail in the search region includes marker ink even if absent at the center.
    high = float(np.percentile(oriented[domain > 0][::4], 99.9))
    if high - background < 6:
        raise DetectionError('搜索区缺少可分离的笔迹灰度特征')
    hp = cv2.absdiff(oriented, cv2.GaussianBlur(oriented, (3, 3), .7))
    noise = 1.4826 * float(np.median(hp[core]))
    contrast = cfg.feature_contrast_gray if cfg.feature_contrast_gray is not None else float(max(1., min(max(noise * 3, (high-background) * .07), (high-background) * .4)))
    fixed = cfg.white_threshold if cfg.polarity == 'white' else cfg.black_threshold
    if fixed is not None:
        levels = [fixed if cfg.polarity == 'white' else 255 - fixed]
    else:
        levels = np.unique(np.clip(np.rint(background + (high - background) * np.linspace(.86, .20, cfg.threshold_levels)), 1, 254).astype(int))[::-1].tolist()
    if cfg.gray_min is not None:
        lo, hi = (cfg.gray_min, cfg.gray_max) if cfg.polarity == 'white' else (255 - cfg.gray_max, 255 - cfg.gray_min)
        levels = sorted(set([int(v) for v in levels if lo - 1 <= v < hi] + [max(0, lo - 1)]), reverse=True)
    return {'g': g, 'oriented': oriented, 'domain': domain, 'sx': sx, 'sy': sy,
            'center': ((cx + .5) * sx - .5, (cy + .5) * sy - .5),
            'levels': levels, 'contrast': contrast, 'background': background,
            'noise': noise, 'illumination_background': illumination_background,
            'region': {'requested_mode': cfg.region_mode,
             'mode': 'central_main_loop' if cfg.region_mode == 'auto' else cfg.region_mode,
             'center_xy': [cx, cy], 'radius_px': radius if cfg.region_mode in ('auto','circle') else None,
             'polygon_xy': list(cfg.search_polygon) if cfg.region_mode == 'polygon' else None}}


def _coarse_evidence(g, contour, contrast):
    p = contour[:, 0, :].astype('float32')[::max(1, len(contour) // 800)]
    p = gaussian_filter1d(p, 1, axis=0, mode='wrap')
    n = _normals(p)
    ds = np.arange(-10, 19, dtype='float32')
    v = _sample(g, p[:, None, :] + n[:, None, :] * ds[None, :, None])
    peak = v[:, 9:17].max(axis=1)
    bilateral = np.minimum(peak - v[:, :5].mean(axis=1), peak - v[:, -5:].mean(axis=1))
    return float(np.mean(bilateral > contrast)), float(np.median(bilateral))


def _discover(ctx, cfg, bridge_mask=None):
    g, domain = ctx['oriented'], ctx['domain']
    scale = min(ctx['sx'], ctx['sy'])
    gap = cfg.gap_px if cfg.gap_px is not None else max(1., min(g.shape[0] / ctx['sy'], g.shape[1] / ctx['sx']) * .00055)
    radius = 0 if gap == 0 else max(1, round(gap * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    eligible = domain > 0
    if cfg.gray_min is not None:
        eligible &= (ctx['g'] >= cfg.gray_min) & (ctx['g'] <= cfg.gray_max)
    # Gray morphology is shared by every threshold; no repeated full-size closing.
    field = np.where(eligible, g, 0).astype(np.uint8)
    if radius:
        field = cv2.morphologyEx(field, cv2.MORPH_CLOSE, kernel)
    minimum = cfg.min_area_px * ctx['sx'] * ctx['sy'] if cfg.min_area_px is not None else np.count_nonzero(domain) * cfg.min_area_ratio
    candidates, records = [], []
    for level in ctx['levels']:
        binary = ((field > level) & (domain > 0)).astype('uint8')
        if bridge_mask is not None:
            binary |= bridge_mask & domain
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        for index, contour in enumerate(contours):
            if hierarchy[0, index, 3] < 0:
                continue
            area = cv2.contourArea(contour)
            if area < minimum or cv2.pointPolygonTest(contour, ctx['center'], False) < 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if x <= 1 or y <= 1 or x + w >= g.shape[1] - 1 or y + h >= g.shape[0] - 1:
                continue
            fraction, contrast = _coarse_evidence(g, contour, ctx['contrast'])
            item = {'contour': contour, 'level': int(level), 'area': area,
                    'evidence_fraction': fraction, 'median_bilateral_contrast': contrast}
            candidates.append(item)
            records.append({key: value for key, value in item.items() if key != 'contour'})
    if not candidates:
        raise DetectionError('未发现包围搜索中心的闭合笔迹，请检查颜色、搜索区域或断点')
    credible = [c for c in candidates if c['evidence_fraction'] >= .40]
    if not credible:
        raise DetectionError('未发现具备双侧灰度证据的主体笔迹')
    largest = max(c['area'] for c in credible)
    principal = [c for c in candidates if c['area'] >= largest * cfg.main_area_fraction]
    best = max(c['evidence_fraction'] for c in principal)
    if best < .55:
        raise DetectionError('主体闭环缺少双侧笔迹证据，拒绝把背景圈或材料边缘当作 ROI')
    # At comparable ink support, the inner enclosed face wins. Small isolated
    # loops are excluded BEFORE this step; this is not "take the smallest hole".
    finalists = [c for c in principal if c['evidence_fraction'] >= best - .02]
    innermost_area = min(c['area'] for c in finalists)
    same_face = [c for c in finalists if c['area'] <= innermost_area + largest * .005]
    selected = max(same_face, key=lambda c: c['level'])
    diagnostics = {'levels_oriented': ctx['levels'], 'evaluated_levels': len(ctx['levels']),
        'candidate_count': len(candidates), 'main_candidate_count': len(principal),
        'main_area_fraction': cfg.main_area_fraction, 'selected_level_oriented': selected['level'],
        'coarse_scale_xy': [ctx['sx'], ctx['sy']], 'closing_radius_coarse_px': radius,
        'morphology_precomputed': True, 'candidates': records,
        'selected_bilateral_fraction': selected['evidence_fraction']}
    return selected, diagnostics, gap


def _guided_gaps(ctx, selected, cfg):
    """Propose short gap bridges, not a radial replacement for the ROI.

    A cyclic minimum-cost path finds nearby ink continuation. Only its weak
    intervals can become bridges. The topology stage then retraces the actual
    ink network, retaining non-star-shaped inward loops and bends.
    """
    g = ctx['oriented']
    r0 = math.sqrt(selected['area'] / math.pi)
    radii = np.arange(max(3, int(r0 * .5)), min(min(g.shape) * .48, int(r0 * 1.2)), dtype='float32')
    if len(radii) < 8 or cfg.gap_px == 0:
        return None, []
    count = 1024
    angles = np.arange(count, dtype='float32') * (2 * math.pi / count)
    cx, cy = ctx['center']
    xx = cx + np.cos(angles)[:, None] * radii
    yy = cy + np.sin(angles)[:, None] * radii
    diameter = max(5, round(ctx['stroke_width_hint_px']*min(ctx['sx'],ctx['sy'])*5)) | 1
    ridge = cv2.morphologyEx(g, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    response = cv2.remap(ridge, xx, yy, cv2.INTER_LINEAR).astype('float32')
    response = cv2.GaussianBlur(response, (3, 3), .5)
    ridge_scale=max(ctx['contrast'],float(np.percentile(response,97)))
    energy = 1 - np.clip(response / max(1.,ridge_scale), 0, 1)
    energy += .015 * ((radii[None, :] - r0) / (.2 * r0)) ** 2
    n, nr = energy.shape
    columns = np.arange(nr)
    shifts = np.arange(-3, 4, dtype=np.int8)

    def dynamic_path(seed=None):
        cost = energy[0].copy() if seed is None else np.full(nr, 1e6, 'float32')
        if seed is not None:
            cost[seed] = energy[0, seed]
        trace = np.zeros((n, nr), np.int8)
        alternatives = np.empty((7, nr), np.float32)
        for a in range(1, n):
            for index, delta in enumerate(shifts):
                d = int(delta)
                alternatives[index] = np.roll(cost, d) + .04 * abs(d)
                if d > 0:
                    alternatives[index, :d] = 1e6
                elif d < 0:
                    alternatives[index, d:] = 1e6
            ids = alternatives.argmin(axis=0)
            cost = alternatives[ids, columns] + energy[a]
            trace[a] = shifts[ids]
        index = int(cost.argmin()) if seed is None else seed
        path = np.empty(n, np.int32)
        for a in range(n - 1, -1, -1):
            path[a] = index
            index -= int(trace[a, index])
        return path

    path = dynamic_path()
    if abs(int(path[0]) - int(path[-1])) > 3:
        path = dynamic_path(int(path[0]))
    radius = radii[path]
    points = np.c_[cx + np.cos(angles) * radius, cy + np.sin(angles) * radius]
    strength = response[np.arange(n), path]
    weak = strength < max(ctx['contrast'],ridge_scale*.4)
    weak |= np.roll(weak, 1) & np.roll(weak, -1)
    if np.all(weak) or not np.any(weak):
        return None, []
    max_gap = min(g.shape) * .055 if cfg.gap_px is None else 2 * cfg.gap_px * min(ctx['sx'], ctx['sy'])
    mask = np.zeros(g.shape, np.uint8)
    records = []
    for start in np.flatnonzero(weak & ~np.roll(weak, 1)):
        ids = []
        index = int(start)
        while weak[index]:
            ids.append(index)
            index = (index + 1) % n
            if len(ids) == n:
                break
        a, b = (start - 2) % n, (index + 1) % n
        sequence = np.r_[a, (start - 1) % n, ids, index, b]
        local = points[sequence]
        length = float(np.linalg.norm(np.diff(local, axis=0), axis=1).sum())
        if not 2 < length <= max_gap:
            continue
        p0, p1 = local[0], local[-1]
        distance = float(np.linalg.norm(p1 - p0))
        tangent0 = p0 - points[(a - 3) % n]
        tangent1 = p1 - points[(b + 3) % n]
        tangent0 /= max(1e-6, np.linalg.norm(tangent0))
        tangent1 /= max(1e-6, np.linalg.norm(tangent1))
        control0, control1 = p0 + tangent0 * distance / 3, p1 + tangent1 * distance / 3
        t = np.linspace(0, 1, max(8, math.ceil(length * 2)))[:, None]
        curve = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * control0 + 3 * (1 - t) * t ** 2 * control1 + t ** 3 * p1
        cv2.polylines(mask, [np.rint(curve).astype('int32')], False, 1, 3)
        records.append({'start_xy_px': ((p0 + .5) / [ctx['sx'], ctx['sy']] - .5).tolist(),
                        'end_xy_px': ((p1 + .5) / [ctx['sx'], ctx['sy']] - .5).tolist(),
                        'length_px': length / min(ctx['sx'], ctx['sy'])})
    return mask if records else None, records


def _main_boundary(selected, ctx, radius=1):
    contour = selected['contour']
    face = np.zeros(ctx['g'].shape, np.uint8)
    cv2.drawContours(face, [contour], -1, 1, -1)
    # Fill subpixel-width dead-end incisions, while retaining broad inward lobes.
    diameter = radius * 2 + 1
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    contours, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    p = contour[:, 0, :].astype(float)
    poly = Polygon(p)
    repaired = not poly.is_valid
    if repaired:
        poly = poly.buffer(0)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda v: v.area)
        if poly.is_empty or poly.geom_type != 'Polygon':
            raise DetectionError('主体轮廓无法恢复为简单闭环')
        p = np.asarray(poly.exterior.coords)[:-1]
    p = (p + .5) / np.array([ctx['sx'], ctx['sy']]) - .5
    return resample(p, max(1.5, .8 / min(ctx['sx'], ctx['sy']))), repaired


def _estimate_stroke_scale(ctx,selected):
    contour=selected['contour'][:,0,:].astype(float)
    contour=contour[::max(1,len(contour)//700)]
    normal=_normals(contour)
    reach=max(16,min(ctx['g'].shape)*.025)
    offsets=np.arange(-4,reach+1,dtype=np.float32)
    values=_sample(ctx['oriented'],contour[:,None,:]+normal[:,None,:]*offsets[None,:,None])
    ink=values>selected['level'];columns=np.arange(len(offsets))[None,:]
    choose=np.where(ink&(offsets[None,:]>=-1)&(offsets[None,:]<=4),np.abs(offsets)[None,:],1e6).argmin(axis=1)
    left=np.where((~ink)&(columns<choose[:,None]),columns,-1).max(axis=1)
    right=np.where((~ink)&(columns>choose[:,None]),columns,len(offsets)).min(axis=1)
    valid=(left>=0)&(right<len(offsets))&ink[np.arange(len(choose)),choose]
    widths=(right-left)[valid]
    scale=min(ctx['sx'],ctx['sy'])
    return float(max(2/scale,np.median(widths)/scale if len(widths)>8 else 3/scale))


def _refine(gray, p, ctx, cfg):
    step = float(np.median(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)))
    width_hint=ctx['stroke_width_hint_px']
    n = _normals(p, max(1, round(width_hint*.6 / max(.1, step))))
    reach = max(width_hint*4,4/min(ctx['sx'],ctx['sy']))
    inward=max(width_hint*.8,2/min(ctx['sx'],ctx['sy']))
    pitch=max(.25,min(1.,width_hint/24))
    ds = np.linspace(-inward, reach, min(257,max(49,math.ceil((reach+inward)/pitch)+1)))
    positions = p[:, None, :] + n[:, None, :] * ds[None, :, None]
    raw = _sample(gray, positions)
    v = raw if cfg.polarity == 'white' else 255 - raw
    v = gaussian_filter1d(v, .65, axis=1)
    search = np.flatnonzero((ds >= -width_hint*.25) & (ds <= width_hint*1.8))
    half_window=max(2,round(width_hint*.8/max(.1,float(ds[1]-ds[0]))))
    side_size=2*half_window+1
    valley_left=minimum_filter1d(v,side_size,axis=1,origin=half_window,mode='nearest')
    valley_right=minimum_filter1d(v,side_size,axis=1,origin=-half_window,mode='nearest')
    local_baseline=np.maximum(valley_left,valley_right)
    local_prominence=v-local_baseline
    maximum_size=max(3,round(width_hint*.35/max(.1,float(ds[1]-ds[0]))))|1
    maxima=v>=maximum_filter1d(v,maximum_size,axis=1,mode='nearest')-.05
    allowed=maxima&(local_prominence>=ctx['contrast'])
    allowed &= (np.arange(len(ds))[None,:]>=search[0])&(np.arange(len(ds))[None,:]<=search[-1])
    choice=np.where(allowed,np.abs(ds[None,:]-width_hint*.35),1e9)
    peak_idx=choice.argmin(axis=1)
    found=allowed.any(axis=1)
    fallback=v[:,search].argmax(axis=1)+search[0]
    peak_idx=np.where(found,peak_idx,fallback)
    rows = np.arange(len(p))
    peak = v[rows, peak_idx]
    baseline=local_baseline[rows,peak_idx]
    prominence = peak - baseline
    half = baseline + np.maximum(0, prominence) * .5
    indices = np.arange(len(ds))[None, :]
    below = v <= half[:, None]
    left = np.where(below & (indices < peak_idx[:, None]), indices, -1).max(axis=1)
    right = np.where(below & (indices > peak_idx[:, None]), indices, len(ds)).min(axis=1)
    bounded = (left >= 0) & (right < len(ds))
    left = np.clip(left, 0, len(ds) - 2)
    right = np.clip(right, 1, len(ds) - 1)
    # Linear half-height crossings place a saturated plateau at its midpoint.
    a = v[rows, left]; b = v[rows, left + 1]
    l = ds[left] + (ds[left + 1] - ds[left]) * np.clip((half - a) / np.maximum(1e-6, b - a), 0, 1)
    a = v[rows, right - 1]; b = v[rows, right]
    r = ds[right - 1] + (ds[right] - ds[right - 1]) * np.clip((a - half) / np.maximum(1e-6, a - b), 0, 1)
    width = r - l
    offset = (l + r) * .5
    valid = found & bounded & (prominence >= ctx['contrast']) & (width > max(.35,width_hint*.06))
    if cfg.gray_min is not None:
        selected_gray = raw[rows, peak_idx]
        if ctx['illumination_background'] is not None:
            native_peak = positions[rows, peak_idx]
            coarse_peak = (native_peak + .5) * [ctx['sx'], ctx['sy']] - .5
            selected_gray = np.clip(selected_gray - _sample(ctx['illumination_background'], coarse_peak) + 128, 0, 255)
        valid &= (selected_gray >= cfg.gray_min) & (selected_gray <= cfg.gray_max)
    if np.count_nonzero(valid) < 10:
        raise DetectionError('原图中可确认的笔迹截面不足')
    widths = np.percentile(width[valid], [25, 50, 75])
    valid &= (width < max(width_hint, widths[2] * 3)) & (np.abs(offset) < max(width_hint, widths[2] * 1.25 + 2 / min(ctx['sx'], ctx['sy'])))
    offset = np.where(valid, offset, 0.)
    offset = gaussian_filter1d(offset, .65, mode='wrap')
    factors = (1., .5, .25, .125, 0.) if cfg.refine_profiles else (0.,)
    damping = None
    local_loops_pruned = 0
    pruned_area_fraction = 0.
    for factor in factors:
        trial = p + n * offset[:, None] * factor
        polygon = Polygon(trial)
        if polygon.is_valid and LineString(np.r_[trial, trial[:1]]).is_simple:
            p = trial
            damping = factor
            break
        if factor == 1.:
            # Opposite banks of an OPEN spur meet at the same ink centerline.
            # Keep the principal face after these zero/small-area appendages
            # collapse. This avoids damping the whole circumference for one
            # local hairpin. A meaningful area change is never accepted here.
            cleaned = polygon.buffer(0)
            pieces = list(cleaned.geoms) if cleaned.geom_type == 'MultiPolygon' else [cleaned]
            pieces = [piece for piece in pieces if piece.geom_type == 'Polygon' and not piece.is_empty]
            if pieces:
                principal = max(pieces, key=lambda v: v.area)
                total = sum(piece.area for piece in pieces)
                area_change = abs(principal.area - abs(polygon.area)) / max(1., principal.area)
                loss = (total - principal.area) / max(1., total)
                if area_change < .002 and loss < .002 and principal.is_valid:
                    cleaned_points = np.asarray(principal.exterior.coords)[:-1]
                    _, nearest = cKDTree(trial).query(cleaned_points)
                    valid = valid[nearest]
                    p = cleaned_points
                    damping = factor
                    local_loops_pruned = max(1, len(pieces) - 1)
                    pruned_area_fraction = area_change
                    break
    if damping is None:
        raise DetectionError('中线精修后出现自交，未输出坐标')
    diagnostics = {'enabled': cfg.refine_profiles, 'damping': damping,
        'valid_fraction': float(valid.mean()), 'median_shift_px': float(np.median(np.abs(offset)) * damping),
        'p95_shift_px': float(np.percentile(np.abs(offset), 95) * damping),
        'local_spur_cleanup': bool(local_loops_pruned),
        'cleanup_area_fraction': pruned_area_fraction,'stroke_width_hint_px':width_hint,
        'profile_inward_px':inward,'profile_outward_px':reach,
        'peak_selection':'nearest bounded local ridge with two local valleys'}
    return p, valid, widths, diagnostics


def _long_dark_gap(gray,p,valid,width,contrast,polarity):
    values=_sample(gray,p)
    if polarity=='black':values=255-values
    if not valid.any():return True
    ink=float(np.median(values[valid]))
    threshold=ink-max(contrast*2,float(np.percentile(values[valid],90)-np.percentile(values,20))*.25)
    weak=(~valid)&(values<threshold)
    if not weak.any():return False
    step=float(np.median(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)))
    # Merge tiny interruptions in an unpainted run; no location/name test.
    size=max(1,round(width/max(.25,step)))|1
    from scipy.ndimage import binary_closing
    padded=np.r_[weak[-size:],weak,weak[:size]]
    merged=binary_closing(padded,structure=np.ones(size,bool))[size:size+len(weak)]
    doubled=np.r_[merged,merged]
    edges=np.diff(np.r_[False,doubled,False].astype('int8'))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    required=max(width*10,len(p)*step*.08)
    return bool(len(lengths) and min(len(weak),int(lengths.max()))*step>required)


def learn_gray_range(gray, config=None):
    """Cheap optional preview only. The desktop learns during actual detection."""
    cfg = (config or DetectorConfig()).validate()
    if cfg.gray_min is not None:
        return {'available': True, 'automatic': False, 'min': cfg.gray_min, 'max': cfg.gray_max}
    ctx = _context(gray, cfg)
    level = int(ctx['levels'][len(ctx['levels']) // 2])
    lo, hi = (level + 1, 255) if cfg.polarity == 'white' else (0, 254 - level)
    return {'available': True, 'automatic': True, 'min': lo, 'max': hi, 'source': 'coarse_preview_only'}


def _open_resample(points, step):
    p = np.asarray(points, float).reshape(-1, 2)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
    if len(p) < 2:
        raise DetectionError('参考线有效点不足，请画出一段走向')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    loc = np.linspace(0., arc[-1], max(3, math.ceil(arc[-1] / max(.25, step)) + 1))
    return np.c_[np.interp(loc, arc, p[:, 0]), np.interp(loc, arc, p[:, 1])]


def _open_normals(points):
    tangent = np.gradient(points, axis=0)
    normal = np.c_[-tangent[:,1], tangent[:,0]]
    return normal / np.maximum(1e-8, np.linalg.norm(normal, axis=1, keepdims=True))


def trace_guided_arc(gray, guide, start, end, polarity='white', width=10., band=None,
                     rough=True,start_tangent=None,end_tangent=None):
    """Snap a rough gesture to one coherent ink path; never export the gesture.

    A Viterbi path crosses a strip around the gesture monotonically. Its states
    are normal offsets; transitions penalize jumping between neighboring ink
    bands. In broad, uninformative paint, continuity/gesture regularity wins.
    Start and end are exact anchors on the existing closed curve.
    """
    width = max(1., float(width))
    if not np.isfinite(np.asarray(guide,float)).all():raise DetectionError('参考线包含无效坐标')
    raw_guide = _open_resample(guide, max(.5, width / 3))
    start, end = np.asarray(start, float), np.asarray(end, float)
    if np.linalg.norm(raw_guide[0]-end)+np.linalg.norm(raw_guide[-1]-start) < np.linalg.norm(raw_guide[0]-start)+np.linalg.norm(raw_guide[-1]-end):
        raw_guide = raw_guide[::-1]
    if rough and len(raw_guide)>6:
        raw_guide=gaussian_filter1d(raw_guide,3.,axis=0,mode='nearest')
        front=max(3,len(raw_guide)//5)
        a=int(np.argmin(np.linalg.norm(raw_guide[:front]-start,axis=1)))
        b=len(raw_guide)-front+int(np.argmin(np.linalg.norm(raw_guide[-front:]-end,axis=1)))
        if b>a+2:raw_guide=raw_guide[a:b+1]
    # Gestures have approximate endpoints. Blend to the snapped curve anchors.
    raw_guide = np.r_[start[None], raw_guide, end[None]]
    path = _open_resample(raw_guide, max(1., width * .45))
    if len(path) > 6:
        smoothed = gaussian_filter1d(path, 2.2 if rough else 1., axis=0, mode='nearest')
        smoothed[0], smoothed[-1] = start, end
        path = _open_resample(smoothed, max(1., width * .45))
    path[0], path[-1] = start, end
    # Approximate clicks may land on a jagged vertex. Blend both joins using
    # the unchanged curve's direction instead of making a backward hook.
    if rough:
        join=min(max(3,round(width*2/max(1.,width*.45))),max(3,len(path)//5))
        def blend_anchor(a,b,tangent_a,tangent_b,count):
            distance=float(np.linalg.norm(b-a))
            ta=tangent_a/max(1e-8,np.linalg.norm(tangent_a));tb=tangent_b/max(1e-8,np.linalg.norm(tangent_b))
            c0=a+ta*distance/3;c1=b-tb*distance/3
            t=np.linspace(0,1,count)[:,None]
            return (1-t)**3*a+3*(1-t)**2*t*c0+3*(1-t)*t**2*c1+t**3*b
        if start_tangent is not None and len(path)>join+2:
            path[:join+1]=blend_anchor(start,path[join],np.asarray(start_tangent),path[join+1]-path[join],join+1)
        if end_tangent is not None and len(path)>join+2:
            path[-join-1:]=blend_anchor(path[-join-1],end,path[-join-1]-path[-join-2],np.asarray(end_tangent),join+1)
    normal = _open_normals(path)
    band = float(band if band is not None else max(width * 7, min(gray.shape) * .018))
    band = max(width, min(band, min(gray.shape) * .15))
    offset_step = max(.5, width * .22)
    half = max(4, min(100, math.ceil(band / offset_step)))
    offsets = np.arange(-half, half+1, dtype=np.float32) * offset_step
    positions = path[:,None,:] + normal[:,None,:] * offsets[None,:,None]
    raw = _sample(gray, positions)
    oriented = raw if polarity == 'white' else 255 - raw
    oriented = gaussian_filter1d(oriented, .65, axis=1)
    # One-dimensional top-hat rejects a broad material field while retaining
    # finite-width strokes. Its scale is estimated pen width, not image ID.
    kernel = max(5, round(width * 5 / offset_step)) | 1
    background = grey_opening(oriented, size=(1,kernel), mode='nearest')
    ridge = np.maximum(0., oriented-background)
    ridge = gaussian_filter1d(ridge, .65, axis=0)
    dynamic = max(1., float(np.percentile(ridge, 97)))
    evidence = np.clip(ridge/dynamic, 0, 1)
    lo = np.percentile(oriented,20,axis=1)
    hi = np.percentile(oriented,95,axis=1)
    brightness = np.clip((oriented-lo[:,None])/np.maximum(1.,hi-lo)[:,None],0,1)
    energy = 1 - .82*evidence - .18*brightness
    energy += .12*(offsets[None,:]/max(width*2,band*.4))**2
    n, states = energy.shape
    column = np.arange(states)
    center = half
    shifts = np.arange(-2,3,dtype=np.int8)
    cost = np.full(states,1e9,np.float32);cost[center]=energy[0,center]
    trace = np.zeros((n,states),np.int8)
    for row in range(1,n):
        alternatives = np.empty((5,states),np.float32)
        for index,delta in enumerate(shifts):
            d = int(delta)
            alternatives[index] = np.roll(cost,d)+.17*d*d
            if d>0:alternatives[index,:d]=1e9
            elif d<0:alternatives[index,d:]=1e9
        best = alternatives.argmin(axis=0)
        cost = alternatives[best,column]+energy[row]
        trace[row]=shifts[best]
    ids = np.empty(n,np.int32);state=center
    for row in range(n-1,-1,-1):
        ids[row]=state;state-=int(trace[row,state])
    selected_offset = offsets[ids].astype(float)
    selected_offset = gaussian_filter1d(selected_offset,.8,mode='nearest')
    selected_offset[0]=selected_offset[-1]=0
    proposed = path+normal*selected_offset[:,None]
    # Remove gesture-scale kinks without sorting by angle or flattening lobes.
    if len(proposed)>8:
        proposed = gaussian_filter1d(proposed,.65,axis=0,mode='nearest')
    proposed[0],proposed[-1]=start,end
    supported = evidence[np.arange(n),ids] > .25
    proposed = _open_resample(proposed,max(.5,width*.15))
    proposed[0],proposed[-1]=start,end
    return proposed, {'ink_supported_fraction':float(supported.mean()),
        'band_px':band,'estimated_width_px':width,'gesture_used_as_coordinates':False,
        'low_evidence_interpolation':bool(supported.mean()<.8)}


def _curve_flags_resample(points, flags, step, reviewed=None):
    p = np.asarray(points,float)
    if np.linalg.norm(p[-1]-p[0])<1e-8:
        p=p[:-1];flags=np.asarray(flags)[:len(p)]
        if reviewed is not None:reviewed=np.asarray(reviewed)[:len(p)]
    closed=np.r_[p,p[:1]]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
    q=resample(p,step)
    locations=np.arange(len(q))/len(q)*arc[-1]
    old=np.searchsorted(arc,locations,side='right')-1
    old=np.minimum(old,len(p)-1)
    new_flags=np.asarray(flags,bool)[old]
    new_reviewed=np.zeros(len(q),bool) if reviewed is None else np.asarray(reviewed,bool)[old]
    return q,new_flags,new_reviewed


def _copy_analysis(analysis):
    # Read-only image bytes are shared by previews and the undo history.
    result=Analysis(analysis.gray,copy.deepcopy(analysis.rois),copy.deepcopy(analysis.metadata))
    # Physical wafer/circle geometry is unchanged by a handwritten-ROI review.
    result._geometry_cache=analysis._geometry_cache.copy()
    return result


def _turns(points, closed=True):
    p=np.asarray(points,float)
    v=np.roll(p,-1,axis=0)-p if closed else np.diff(p,axis=0)
    v=v/np.maximum(1e-8,np.linalg.norm(v,axis=1,keepdims=True))
    before=np.roll(v,1,axis=0) if closed else v[:-1]
    after=v if closed else v[1:]
    return np.abs(np.arctan2(before[:,0]*after[:,1]-before[:,1]*after[:,0],np.sum(before*after,axis=1)))


def path_complexity(points,width):
    p=resample(points,max(1.,width*.3))
    turn=_turns(p)
    radius=max(2,round(width*3/max(1.,width*.3)))
    local=sum(np.roll(turn,k) for k in range(-radius,radius+1))
    chord=np.linalg.norm(np.roll(p,-radius,axis=0)-np.roll(p,radius,axis=0),axis=1)
    step=float(np.mean(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)))
    ratio=2*radius*step/np.maximum(step,chord)
    median=float(np.median(local));mad=1.4826*float(np.median(np.abs(local-median)))
    # These dimensionless regularizers do not encode any image location/name.
    tangled=(local>max(math.pi*2.5,median+5*mad))|(ratio>2.6)
    zigzag=(local>max(math.pi*1.35,median+3*mad))&(ratio<1.6)
    bad=tangled|zigzag
    # Long nearly straight strips can alternate between parallel strokes while
    # remaining simple. Detect repeated normal motion at pen-width scale.
    coarse=cv2.approxPolyDP(p.astype('float32'),width*1.3,True)[:,0,:]
    corners=np.unique(cKDTree(p).query(coarse)[1]);strip_count=0
    for a,b in zip(corners,np.r_[corners[1:],corners[:1]+len(p)]):
        ids=np.arange(a,b+1)%len(p);segment=p[ids]
        direction=segment[-1]-segment[0];length=float(np.linalg.norm(direction))
        if length<width*12 or len(segment)<10:continue
        normal=np.array([-direction[1],direction[0]])/max(1e-6,length)
        residual=(segment-segment[0])@normal
        repeated=float(np.sum(np.abs(np.diff(residual))))
        if repeated>width*2 and float(turn[ids].sum())>math.pi*2:
            bad[ids]=True;strip_count+=1
    return p,bad,{'turning_sum_rad':float(turn.sum()),
        'severe_return_fraction':float((ratio>2.6).mean()),'probe_step_px':step,
        'turn_noise_mad':mad,'estimated_width_px':float(width),'parallel_strip_candidates':strip_count}


def _regularize_auto(gray,points,flags,width,cfg):
    diagnostics={'enabled':bool(cfg.regularize_path and cfg.path_smoothing>0),'regions':[]}
    if not diagnostics['enabled']:
        return points,flags,diagnostics
    q,bad,before=path_complexity(points,width)
    diagnostics['before']=before
    if not bad.any() or bad.all():
        diagnostics['after']=before
        return points,flags,diagnostics
    expand=max(1,round(width/max(1.,before['probe_step_px'])))
    bad=maximum_filter1d(bad.astype('uint8'),2*expand+1,mode='wrap')>0
    if bad.all():
        diagnostics['after']=before
        return points,flags,diagnostics
    cut=int(np.flatnonzero(~bad)[len(np.flatnonzero(~bad))//2])
    start=int(cKDTree(points).query(q[cut])[1])
    p=np.roll(points,-start,axis=0);f=np.roll(flags,-start)
    q,bad,_=path_complexity(p,width)
    bad=maximum_filter1d(bad.astype('uint8'),2*expand+1,mode='wrap')>0
    edges=np.diff(np.r_[False,bad,False].astype('int8'))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
    total_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(np.r_[p,p[:1]],axis=0),axis=1))]
    parts=[];flag_parts=[];cursor=0
    for begin,end in zip(starts,ends):
        a=max(cursor,int(np.searchsorted(total_arc,begin/len(q)*total_arc[-1])-1))
        b=min(len(p)-1,int(np.searchsorted(total_arc,(end+1)/len(q)*total_arc[-1])+1))
        if b-a<4 or b-a>len(p)*.25:continue
        segment=p[a:b+1]
        ds=max(1.,width*.45)
        guide=_open_resample(segment,ds)
        guide=gaussian_filter1d(guide,max(.8,cfg.path_smoothing*1.6),axis=0,mode='nearest')
        guide[0],guide[-1]=segment[0],segment[-1]
        proposal,evidence=trace_guided_arc(gray,guide,segment[0],segment[-1],cfg.polarity,width,band=width*2.5,rough=False)
        displacement=float(cKDTree(segment).query(proposal)[0].max())
        old_turn=float(_turns(_open_resample(segment,max(1,width*.3)),False).sum())
        new_turn=float(_turns(_open_resample(proposal,max(1,width*.3)),False).sum())
        if new_turn>old_turn*.9 or displacement>width*(2+cfg.path_smoothing*2):continue
        trial=np.r_[p[:a],proposal,p[b+1:]]
        if not Polygon(trial).is_valid or not LineString(np.r_[trial,trial[:1]]).is_simple:continue
        parts.extend([p[cursor:a],proposal]);flag_parts.extend([f[cursor:a],np.ones(len(proposal),bool)])
        cursor=b+1
        diagnostics['regions'].append({'old_endpoints_xy':segment[[0,-1]].tolist(),
            'old_turning_rad':old_turn,'new_turning_rad':new_turn,'max_shift_px':displacement,
            'estimated_width_px':float(width),'needs_operator_review':True,**evidence})
    if not parts:
        diagnostics['after']=before
        return points,flags,diagnostics
    parts.append(p[cursor:]);flag_parts.append(f[cursor:])
    proposal=np.concatenate(parts);new_flags=np.concatenate(flag_parts)
    if not Polygon(proposal).is_valid or not LineString(np.r_[proposal,proposal[:1]]).is_simple:
        diagnostics['regions']=[];diagnostics['rejected_combined_geometry']=True;diagnostics['after']=before
        return points,flags,diagnostics
    proposal,new_flags,_=_curve_flags_resample(proposal,new_flags,cfg.sample_step_px)
    diagnostics['after']=path_complexity(proposal,width)[2]
    return proposal,new_flags,diagnostics


def select_review_arc(analysis, endpoints=None, polygon=None, complement=False):
    p=analysis.rois[0].points[:-1];n=len(p)
    if endpoints is not None:
        e=np.asarray(endpoints,float).reshape(-1,2)
        if len(e)!=2:raise DetectionError('请大致选择两个端点')
        distance,indices=cKDTree(p).query(e)
        tolerance=max(analysis.rois[0].width_quartiles[1]*10,min(analysis.gray.shape)*.04)
        if np.max(distance)>tolerance:raise DetectionError('端点离当前曲线过远，请在待替换段附近点击')
        a,b=map(int,indices)
        if a==b:raise DetectionError('两个端点过近，请重新选择')
        forward=(b-a)%n
        if forward>n//2:a,b=b,a
    elif polygon is not None:
        region=np.asarray(polygon,np.float32).reshape(-1,2)
        if len(region)<3:raise DetectionError('复核区域至少需要 3 点')
        # Bounding-box prefilter keeps 20k-point curves responsive.
        lo,hi=region.min(axis=0),region.max(axis=0)
        inside=np.zeros(n,bool)
        ids=np.flatnonzero(np.all((p>=lo)&(p<=hi),axis=1))
        inside[ids]=[cv2.pointPolygonTest(region,tuple(map(float,p[i])),False)>=0 for i in ids]
        if not inside.any():raise DetectionError('圈选区域没有覆盖当前曲线')
        if inside.all():raise DetectionError('圈选覆盖了整圈，请只圈选需要修改的一段')
        starts=np.flatnonzero(inside&~np.roll(inside,1));runs=[]
        for start in starts:
            length=0
            while inside[(start+length)%n]:length+=1
            runs.append((length,int(start)))
        length,a=max(runs);b=(a+length-1)%n
    else:raise DetectionError('请先选两个端点，或圈选复核区域')
    if complement:a,b=b,a
    margin=max(1,round(analysis.rois[0].width_quartiles[1]*1.5/analysis.metadata['parameters']['sample_step_px']))
    if (b-a)%n+2*margin<n-4:a,b=(a-margin)%n,(b+margin)%n
    indices=(a+np.arange((b-a)%n+1))%n
    return {'start_index':int(a),'end_index':int(b),'indices':indices,
        'points':p[indices].copy(),'endpoints':p[[a,b]].copy()}


def preview_review(analysis, selection, guide=None):
    segment=selection['points'];width=analysis.rois[0].width_quartiles[1]
    if guide is None:
        # A region-only review can start with a smooth suggested continuation.
        guide=gaussian_filter1d(_open_resample(segment,max(1,width/3)),3.,axis=0,mode='nearest')
    p=analysis.rois[0].points[:-1];n=len(p)
    a,b=selection['start_index'],selection['end_index']
    span=max(1,round(width*1.5/analysis.metadata['parameters']['sample_step_px']))
    tangent_a=p[a]-p[(a-span)%n];tangent_b=p[(b+span)%n]-p[b]
    points,info=trace_guided_arc(analysis.gray,guide,segment[0],segment[-1],analysis.rois[0].polarity,width,
                               start_tangent=tangent_a,end_tangent=tangent_b)
    if guide is not None:
        g=np.asarray(guide,float)
        info['rough_guide_xy']=g[::max(1,len(g)//256)].tolist()
    candidate=apply_review(analysis,selection,points,info,preview=True)
    return {'analysis':candidate,'points':points,'info':info,'selection':selection}


def apply_review(analysis, selection, replacement, evidence=None, preview=False):
    result=_copy_analysis(analysis);roi=result.rois[0]
    old=roi.points[:-1];n=len(old);a=selection['start_index'];b=selection['end_index']
    new=np.asarray(replacement,float).reshape(-1,2).copy()
    if len(new)<2 or not np.isfinite(new).all():raise DetectionError('替换线段无效')
    new[0],new[-1]=old[a],old[b]
    unchanged=(b+np.arange((a-b)%n+1))%n
    p=np.r_[new,old[unchanged][1:-1]]
    flags=np.r_[np.zeros(len(new),bool),roi.review[:-1][unchanged][1:-1]]
    reviewed=roi.reviewed[:-1] if roi.reviewed is not None else np.zeros(n,bool)
    reviewed=np.r_[np.ones(len(new),bool),reviewed[unchanged][1:-1]]
    h,w=result.gray.shape
    if np.any(p<0) or np.any(p[:,0]>w-1) or np.any(p[:,1]>h-1):raise DetectionError('替换线段超出图像范围')
    # Selection is in existing traversal order, so preserve that orientation.
    poly=Polygon(p)
    if not poly.is_valid or not LineString(np.r_[p,p[:1]]).is_simple:
        raise DetectionError('新线段与其余曲线相交，请调整参考走向；原结果未改动')
    step=result.metadata['parameters']['sample_step_px']
    p,flags,reviewed=_curve_flags_resample(p,flags,step,reviewed)
    if not Polygon(p).is_valid or not LineString(np.r_[p,p[:1]]).is_simple:
        raise DetectionError('采样后的替换线存在自交，原结果未改动，请调整参考走向')
    origin=np.asarray(result.metadata['origin_image_xy_px'])
    angle=np.arctan2(-(p[:,1]-origin[1]),p[:,0]-origin[0])
    shift=int(np.argmin(np.abs(angle)))
    p,flags,reviewed=(np.roll(v,-shift,axis=0) for v in (p,flags,reviewed))
    roi.points=np.r_[p,p[:1]];roi.review=np.r_[flags,flags[:1]];roi.reviewed=np.r_[reviewed,reviewed[:1]]
    poly=Polygon(p)
    roi.info.update(area_px2=poly.area,perimeter_px=poly.length,review_fraction=float(flags.mean()),
        max_adjacent_spacing_px=float(np.linalg.norm(np.diff(roi.points,axis=0),axis=1).max()),
        gray_quantiles_10_50_90=np.percentile(_sample(result.gray,p),[10,50,90]).tolist(),
        manual_reviewed_fraction=float(reviewed.mean()))
    result.metadata['review_complete']=False
    result.metadata.setdefault('review_history',[]).append({'endpoints_xy':selection['endpoints'].tolist(),
        'replacement_length_px':float(np.linalg.norm(np.diff(new,axis=0),axis=1).sum()),
        'operator_accepted':not preview,'evidence':evidence or {}})
    result.metadata['status']='REVIEW'
    result.metadata['warnings']=['已修改选中线段；请继续检查剩余曲线，完成后点击确认复核并保存']
    return result


def confirm_review(analysis):
    result=_copy_analysis(analysis)
    if not Polygon(result.rois[0].points).is_valid or not LineString(result.rois[0].points).is_simple:
        raise DetectionError('当前曲线几何无效，不能确认复核')
    roi=result.rois[0];roi.review[:]=False;roi.info['review_fraction']=0.
    result.metadata.update(status='OK',review_complete=True,warnings=[],
        operator_confirmed=True,confirmation_scope='entire_current_curve')
    return result


def analyze(gray, config=None):
    started = time.perf_counter()
    cfg = (config or DetectorConfig()).validate()
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 16:
        raise ValueError('analyze 需要 uint8 二维灰度图，图像边长至少 16')
    ctx = _context(gray, cfg)
    context_done = time.perf_counter()
    selected, scan, gap = _discover(ctx, cfg)
    ctx['stroke_width_hint_px']=_estimate_stroke_scale(ctx,selected)
    scan_done = time.perf_counter()
    p, repaired = _main_boundary(selected, ctx)
    p, valid, widths, profile = _refine(gray, p, ctx, cfg)
    repair = {'searched': False, 'applied': False, 'proposals': []}
    active_bridges = None
    dark_gap=selected['evidence_fraction']<.8 and _long_dark_gap(gray,p,valid,float(widths[1]),ctx['contrast'],cfg.polarity)
    if (float(valid.mean()) < .55 or dark_gap) and cfg.gap_px != 0:
        repair['searched'] = True
        bridge_mask, proposals = _guided_gaps(ctx, selected, cfg)
        if bridge_mask is not None:
            try:
                alternate, alternate_scan, _ = _discover(ctx, cfg, bridge_mask)
                if alternate['evidence_fraction'] > selected['evidence_fraction'] + .05:
                    alternate_p, alternate_repaired = _main_boundary(alternate, ctx)
                    q, q_valid, q_widths, q_profile = _refine(gray, alternate_p, ctx, cfg)
                    if q_valid.mean() > valid.mean() + .10:
                        selected, scan, p, valid, widths, profile = alternate, alternate_scan, q, q_valid, q_widths, q_profile
                        repaired |= alternate_repaired
                        active_bridges = bridge_mask
                        repair.update(applied=True, proposals=proposals)
            except DetectionError:
                pass
    cleanup_radius = max(1, math.ceil(widths[1] * min(ctx['sx'], ctx['sy']) * .65))
    effective_cleanup_radius = 1
    if cfg.refine_profiles and profile['damping'] < 1 and cleanup_radius > 1:
        cleaned, repaired_again = _main_boundary(selected, ctx, cleanup_radius)
        q, q_valid, q_widths, q_profile = _refine(gray, cleaned, ctx, cfg)
        if q_profile['damping'] > profile['damping'] and q_profile['valid_fraction'] >= profile['valid_fraction'] - .05:
            p, valid, widths, profile = q, q_valid, q_widths, q_profile
            repaired |= repaired_again
            effective_cleanup_radius = cleanup_radius
    profile['dead_end_cleanup_radius_coarse_px'] = effective_cleanup_radius
    if widths[1] / math.sqrt(selected['area'] / (ctx['sx'] * ctx['sy'])) > cfg.max_width_ratio:
        raise DetectionError('主体边界过宽，更接近材料色带而非笔迹')
    if float(valid.mean()) < .55:
        raise DetectionError('原图主体笔迹支持不足 55%，未接受背景圈')
    old_fraction = np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.r_[p, p[:1]], axis=0), axis=1))]
    old_fraction /= old_fraction[-1]
    points = resample(p, cfg.sample_step_px)
    locations = np.arange(len(points)) / len(points)
    flags = np.interp(locations, old_fraction, np.r_[~valid, ~valid[:1]].astype(float)) > .25
    interpolated_fraction = 0.
    if active_bridges is not None:
        coarse_points = (points + .5) * [ctx['sx'], ctx['sy']] - .5
        interpolated = _sample(active_bridges, coarse_points) > 0
        flags |= interpolated
        interpolated_fraction = float(interpolated.mean())
    points,flags,regularization=_regularize_auto(gray,points,flags,float(widths[1]),cfg)
    if cv2.contourArea(points.astype('float32'), oriented=True) > 0:
        points = points[::-1]
        flags = flags[::-1]
    h, w = gray.shape
    origin = np.array([(w - 1) / 2, (h - 1) / 2])
    angle = np.arctan2(-(points[:, 1] - origin[1]), points[:, 0] - origin[0])
    shift = int(np.argmin(np.abs(angle)))
    points = np.roll(points, -shift, axis=0)
    flags = np.roll(flags, -shift)
    points = np.r_[points, points[:1]]
    flags = np.r_[flags, flags[:1]]
    polygon = Polygon(points)
    if not polygon.is_valid or not LineString(points).is_simple:
        raise DetectionError('输出曲线不满足无自交约束')
    threshold = selected['level'] if cfg.polarity == 'white' else 255 - selected['level']
    gray_values = _sample(gray, points[:-1])
    info = {'area_px2': float(polygon.area), 'perimeter_px': float(polygon.length),
        'closed': True, 'simple_no_self_intersections': True, 'geometry_repaired': repaired,
        'branch_or_merged_component': False, 'supported_fraction': float(valid.mean()),
        'support_fraction_definition':'initial native profiles; regularized segments have their own evidence and remain pending review',
        'review_fraction': float(flags[:-1].mean()), 'gray_quantiles_10_50_90': np.percentile(gray_values, [10, 50, 90]).tolist(),
        'max_adjacent_spacing_px': float(np.linalg.norm(np.diff(points, axis=0), axis=1).max()),
        'profile_refinement': profile, 'coarse_bilateral_fraction': selected['evidence_fraction'],
        'interpolated_fraction': interpolated_fraction,
        'auto_regularization':regularization,
        'selection_rule': 'one main center-enclosing face, inner boundary at crossings; small disconnected loops excluded'}
    warnings = []
    if regularization['regions']:
        warnings.append(f"已抑制 {len(regularization['regions'])} 处异常折返或跳线，可人工复核走势")
    if repair['applied']:
        warnings.append('主体笔迹存在断口，已沿邻近走势进行有限长度补线；插值点已标记，需复核')
    if info['review_fraction'] > .10:
        warnings.append(f"约 {info['review_fraction']:.1%} 曲线点的双侧笔迹证据偏弱，请复核贴边或断口位置")
    if cfg.refine_profiles and profile['damping'] < 1:
        warnings.append('局部交点限制了中线位移，已减小精修幅度以保持无自交')
    if not cfg.refine_profiles:
        warnings.append('已关闭常规中线精修；异常走线整形仍按独立开关执行')
    if repaired:
        warnings.append('粗轮廓存在像素接触，已恢复主闭环')
    status = 'REVIEW' if warnings else 'OK'
    bounds = {'available': True, 'automatic': cfg.gray_min is None,
        'min': cfg.gray_min if cfg.gray_min is not None else (threshold + 1 if cfg.polarity == 'white' else 0),
        'max': cfg.gray_max if cfg.gray_max is not None else (255 if cfg.polarity == 'white' else threshold - 1),
        'source': 'manual_inclusive_range' if cfg.gray_min is not None else 'same_detection_main_loop_threshold',
        'note': '发现闭环使用该阈值；中线坐标使用原图局部灰度截面。'}
    ended = time.perf_counter()
    metadata = {'schema_version': 4, 'algorithm_version': 'v3_continuous_path_and_operator_review',
        'image_width_px': w, 'image_height_px': h, 'origin_image_xy_px': origin.tolist(),
        'angle_convention': '0 degrees right; counterclockwise; Cartesian y points up',
        'curve_definition': 'one main pen centerline, ordered polyline, repeated closing point',
        'status': status, 'roi_count': 1, 'warnings': warnings, 'parameters': asdict(cfg),
        'search_region': ctx['region'], 'effective_gap_radius_px': gap, 'threshold_scan': {cfg.polarity: scan},
        'gap_repair': repair,
        'gray_range': bounds, 'feature_evidence': {'enabled': True, 'estimated_texture_noise_gray': ctx['noise'],
            'local_contrast_gray': ctx['contrast'], 'native_profile_fraction': float(valid.mean())},
        'numba_available': False, 'requires_warmup': False, 'detect_seconds': ended - started,
        'stage_seconds': {'coarse_prepare': context_done - started, 'topology_select': scan_done - context_done,
                          'native_refine_and_validate': ended - scan_done},
        'limitations': ['Main loop must enclose the image center or the manually selected search-circle center.',
            'Main-area ratio is a scale heuristic, not semantic recognition of pen versus scratch.',
            'Merged ink or absent strokes can be ambiguous; REVIEW is not a pixel-accuracy guarantee.',
            'Sampling step is not measurement accuracy; no ground-truth annotation is assumed.']}
    roi = ROIResult('roi_01', cfg.polarity, points, int(threshold), widths.tolist(), flags, info)
    roi.reviewed=np.zeros(len(roi.points),bool)
    return Analysis(gray, [roi], metadata)


def detect_image(path, config=None):
    analysis = analyze(load_gray(path), config)
    analysis.metadata['source'] = str(Path(path))
    return analysis


DEFAULT_COLORS = ['#32E614', '#008CFF', '#FF5A00', '#CF80FF', '#FFD43B', '#33DDDD']


def coordinate_frame(shape, origin='center', ring_center=None):
    """Pixel centers: rightmost x=w-1, bottom y=h-1; never move ROI points."""
    h, w = shape[:2]
    if origin == 'top_left':
        return (0., 0.), 1.
    if origin == 'bottom_left':
        return (0., float(h - 1)), -1.
    if origin == 'center':
        return ((w - 1) / 2., (h - 1) / 2.), -1.
    if origin == 'ring_center':
        center=np.asarray(ring_center,dtype=float)
        if center.shape!=(2,) or not np.isfinite(center).all():
            raise ValueError('亮圆环圆心尚未拟合，不能使用该坐标原点')
        return tuple(center), -1.
    raise ValueError('未知坐标原点')


def make_export_stem(source, when=None, reviewed=False):
    when = when or dt.datetime.now()
    return f"{Path(source).stem}_{when:%Y%m%d_%H%M%S_%f}" + ('_reviewed' if reviewed else '')


def _circle_irls(points, model, weights, iterations=8):
    """Geometric circle fit, with Cauchy weights for occlusions/outliers."""
    model=np.asarray(model,float).copy();points=np.asarray(points,float)
    for _ in range(iterations):
        delta=model[:2]-points;distance=np.linalg.norm(delta,axis=1)
        error=distance-model[2]
        sigma=max(.2,1.4826*np.median(np.abs(error-np.median(error))))
        weight=np.asarray(weights)/(1+(error/(2*sigma))**2)
        jacobian=np.c_[delta/np.maximum(distance[:,None],1e-9),-np.ones(len(points))]
        step=np.linalg.lstsq(jacobian*np.sqrt(weight[:,None]),-error*np.sqrt(weight),rcond=None)[0]
        model+=step
        if np.linalg.norm(step)<1e-4:break
    return model


def fit_bright_ring(gray):
    """Fit the outer bright-ring edge; no fixed grayscale threshold or label mask.

    Otsu supplies only a coarse spatial seed. Radial falling-edge candidates,
    angular consensus and robust geometric residuals choose the outer circle.
    Occluded arcs are rejected, rather than following a label's boundary.
    The final fit samples original pixels. Failure never invalidates the main ROI.
    """
    started=time.perf_counter()
    def failed(reason):
        return {'status':'not_found','reason':reason,'fit_seconds':time.perf_counter()-started}
    if not isinstance(gray,np.ndarray) or gray.ndim!=2 or gray.dtype!=np.uint8 or min(gray.shape)<32:
        return failed('图像尺寸或类型不适合圆环拟合')
    h0,w0=gray.shape;scale=min(1.,1024/max(gray.shape))
    g=cv2.resize(gray,(round(w0*scale),round(h0*scale)),interpolation=cv2.INTER_AREA)
    h,w=g.shape;sx,sy=w/w0,h/h0
    learned_gray,mask=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return failed('未找到可支持圆环的亮区域')
    contour=max(contours,key=cv2.contourArea)
    x,y,bw,bh=cv2.boundingRect(contour);seed=np.array([x+(bw-1)/2,y+(bh-1)/2])
    rb=max(bw,bh)/2
    if rb<8 or cv2.contourArea(contour)<.01*h*w:return failed('亮区域不足以拟合完整圆环')
    count=540;angles=np.linspace(0,2*np.pi,count,endpoint=False)
    vectors=np.c_[np.cos(angles),np.sin(angles)]
    radii=np.arange(0,np.hypot(w,h),dtype=np.float32)
    positions=seed+vectors[:,None,:]*radii[None,:,None]
    profiles=cv2.remap(g,positions[:,:,0].astype('float32'),positions[:,:,1].astype('float32'),
        cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype('float32')
    profiles=gaussian_filter1d(profiles,1.2,axis=1)
    drop=-np.gradient(profiles,axis=1)
    valid=(positions[:,:,0]>=2)&(positions[:,:,0]<w-3)&(positions[:,:,1]>=2)&(positions[:,:,1]<h-3)
    valid&=(radii[None,:]>rb*.65)&(radii[None,:]<rb*1.3)
    peaks=drop>=maximum_filter1d(drop,9,axis=1)
    strength=np.where(valid&peaks,np.maximum(drop,0),0)
    k=5;indices=np.argpartition(strength,-k,axis=1)[:,-k:];rows=np.arange(count)[:,None]
    relative=strength[rows,indices]
    relative=relative/np.maximum(relative.max(axis=1,keepdims=True),1e-6)
    usable=relative.ravel()>.025
    points=positions[rows,indices].reshape(-1,2)[usable]
    groups=np.repeat(np.arange(count),k)[usable]
    scores=(.25+.75*np.sqrt(relative)).ravel()[usable]
    if len(points)<count:return failed('圆环边缘证据不足')
    histogram=np.zeros(len(radii));np.add.at(histogram,indices.ravel(),relative.ravel())
    modes=np.argsort(gaussian_filter1d(histogram,3))[-8:]
    models=[np.r_[seed,float(radius)] for radius in modes]
    rng=np.random.default_rng(331);tolerance=max(2.,rb*.006)
    for _ in range(650):
        ids=rng.choice(len(points),3,replace=False);q=points[ids]
        if len(set((groups[ids]//(count//12)).tolist()))<3:continue
        a=2*(q[1:]-q[0]);b=np.sum(q[1:]**2-q[0]**2,axis=1)
        if abs(np.linalg.det(a))<rb:continue
        center=np.linalg.solve(a,b);radius=np.linalg.norm(q[0]-center)
        if np.linalg.norm(center-seed)>rb*.3 or not rb*.65<radius<rb*1.3:continue
        models.append(np.r_[center,radius])
    ranked=[]
    for model in models:
        residual=np.abs(np.linalg.norm(points-model[:2],axis=1)-model[2])
        agreement=np.maximum(0,1-(residual/tolerance)**2)*scores
        per_angle=np.zeros(count);np.maximum.at(per_angle,groups,agreement)
        ranked.append((float(per_angle.sum()),model))
    best=max(score for score,_ in ranked)
    # Nearby concentric edges: take the outer one only with comparable support.
    model=max((model for score,model in ranked if score>=.92*best),key=lambda model:model[2]).copy()
    for _ in range(8):
        residual=np.abs(np.linalg.norm(points-model[:2],axis=1)-model[2])
        merit=np.where(residual<2*tolerance,scores/(1+residual),0)
        maxima=np.zeros(count);np.maximum.at(maxima,groups,merit)
        chosen=(merit>0)&(merit>=maxima[groups]-1e-9)
        if chosen.sum()<12:return failed('圆弧有效点不足')
        model=_circle_irls(points[chosen],model,scores[chosen],4)
    supported=np.zeros(count,bool);supported[groups[chosen]]=True
    coverage=float(supported.mean())
    sectors=len(np.unique(groups[chosen]//(count//12)))
    if coverage<.55 or sectors<9:return failed('遮挡或缺口过大，圆弧覆盖不足，未输出圆心')
    model=np.r_[(model[:2]+.5)/[sx,sy]-.5,model[2]/math.sqrt(sx*sy)]
    coarse=model.copy();band=max(6.,model[2]*.005)
    offsets=np.arange(-math.ceil(band),math.ceil(band)+1,dtype=float)
    positions=model[:2]+vectors[:,None,:]*(model[2]+offsets[None,:])[:,:,None]
    profiles=gaussian_filter1d(_sample(gray,positions),1.,axis=1)
    gradient=-np.gradient(profiles,axis=1)
    indices=np.argmax(gradient/(1+(offsets[None,:]/band)**2*.25),axis=1)
    peak=gradient[np.arange(count),indices]
    meaningful=(peak>np.percentile(np.maximum(peak,0),80)*.08)&supported
    meaningful&=(indices>0)&(indices<len(offsets)-1)
    inside=(positions[:,:,0].min(axis=1)>=0)&(positions[:,:,0].max(axis=1)<w0)
    inside&=(positions[:,:,1].min(axis=1)>=0)&(positions[:,:,1].max(axis=1)<h0)
    meaningful&=inside
    if meaningful.sum()>=count*.5:
        ii=indices[meaningful];vv=gradient[meaningful];rr=np.arange(len(ii))
        denominator=vv[rr,ii-1]-2*vv[rr,ii]+vv[rr,ii+1]
        adjustment=np.divide(.5*(vv[rr,ii-1]-vv[rr,ii+1]),denominator,out=np.zeros(len(ii)),where=np.abs(denominator)>1e-6)
        edge_points=model[:2]+vectors[meaningful]*(model[2]+offsets[ii]+np.clip(adjustment,-.5,.5))[:,None]
        weights=np.minimum(1.,peak[meaningful]/max(1e-6,np.percentile(peak[meaningful],75)))
        refined=_circle_irls(edge_points,model,weights,10)
        if np.linalg.norm(refined[:2]-coarse[:2])<=max(4.,coarse[2]*.02):model=refined
        errors=np.abs(np.linalg.norm(edge_points-model[:2],axis=1)-model[2])
        median_residual=float(np.median(errors));inlier_count=int(np.count_nonzero(errors<max(2.,model[2]*.003)))
    else:
        original_points=(points[chosen]+.5)/[sx,sy]-.5
        median_residual=float(np.median(np.abs(np.linalg.norm(original_points-model[:2],axis=1)-model[2])))
        inlier_count=int(chosen.sum())
    if not np.isfinite(model).all() or median_residual>model[2]*.02:
        return failed('圆弧偏离圆模型过大，未输出圆心')
    return {'status':'ok','center_image_xy_px':model[:2].tolist(),'radius_px':float(model[2]),
        'arc_coverage':coverage,'inlier_count':inlier_count,'median_residual_px':median_residual,
        'adaptive_seed_gray':float(learned_gray),'fit_seconds':time.perf_counter()-started,
        'method':'adaptive_radial_edges_angular_consensus_robust_circle',
        'note':'拟合圆心为图像像素坐标；遮挡处的圆环轮廓由圆模型补全'}


def radial_intersections(points, center, angles_deg, y_sign=-1.):
    """All positive ray/polyline intersections, including folds and tangencies.

    y_sign=-1 uses image pixels (upward angles); +1 uses numeric machine XY.
    Shared vertices are deduplicated geometrically, never by an angular bin.
    A collinear segment contributes its finite endpoints.
    """
    p=np.asarray(points,dtype=float);center=np.asarray(center,dtype=float)
    angles=np.asarray(angles_deg,dtype=float).reshape(-1)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
        raise ValueError('求交轮廓需包含至少三个有效坐标点')
    if center.shape!=(2,) or not np.isfinite(center).all() or not np.isfinite(angles).all():
        raise ValueError('射线圆心与角度必须为有限数值')
    if not np.allclose(p[0],p[-1],atol=1e-9,rtol=0):p=np.vstack([p,p[0]])
    a=p[:-1]-center;e=np.diff(p,axis=0)
    nonzero=np.linalg.norm(e,axis=1)>1e-10;a=a[nonzero];e=e[nonzero]
    cross_ae=a[:,0]*e[:,1]-a[:,1]*e[:,0]
    lengths=np.linalg.norm(e,axis=1)
    tolerance=max(1e-7,float(np.max(np.abs(p)))*1e-10)
    results=[]
    for start in range(0,len(angles),48):
        theta=np.radians(angles[start:start+48]);d=np.c_[np.cos(theta),y_sign*np.sin(theta)]
        den=d[:,0,None]*e[:,1]-d[:,1,None]*e[:,0]
        regular=np.abs(den)>np.maximum(1.,lengths)[None,:]*1e-11
        t=np.divide(cross_ae[None,:],den,out=np.full_like(den,np.nan),where=regular)
        cross_ad=a[:,0]*d[:,1,None]-a[:,1]*d[:,0,None]
        u=np.divide(cross_ad,den,out=np.full_like(den,np.nan),where=regular)
        valid=regular&(t>=-tolerance)&(u>=-1e-10)&(u<=1+1e-10)
        for j,vector in enumerate(d):
            values=t[j,valid[j]]
            collinear=(~regular[j])&(np.abs(cross_ad[j])<=tolerance)
            if collinear.any():
                endpoints=np.r_[a[collinear]@vector,(a[collinear]+e[collinear])@vector]
                values=np.r_[values,endpoints[endpoints>=-tolerance]]
            values=np.sort(np.maximum(values,0.))
            if len(values):values=values[np.r_[True,np.diff(values)>tolerance]]
            results.append(values)
    return results


def detect_wafer_outline(gray, bright_ring):
    """Extract the actual dark-wafer/bright-film exterior, not the ring circle.

    Learn the split inside the ring, keep a central connected wafer, ignore
    internal holes, and refine its exterior at original resolution. The fitted
    circle limits the search only; flat edges and notches remain in the contour.
    """
    started=time.perf_counter()
    if not bright_ring or bright_ring.get('status')!='ok':
        raise DetectionError('未找到亮圆环拟合圆心，不能提取晶圆外沿')
    center=np.asarray(bright_ring['center_image_xy_px'],float);radius=float(bright_ring['radius_px'])
    h,w=gray.shape;scale=min(1.,1536/max(h,w))
    g=cv2.resize(gray,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    sx,sy=g.shape[1]/w,g.shape[0]/h
    seed=(center+.5)*[sx,sy]-.5;r=radius*math.sqrt(sx*sy)
    domain=np.zeros_like(g);cv2.circle(domain,tuple(np.round(seed).astype(int)),round(r*.94),255,-1)
    values=g[domain>0]
    threshold,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=np.uint8((g<=threshold)&(domain>0))*255
    # Bridge narrow white ink cuts before taking the exterior; otherwise the
    # wafer contour follows the handwritten ROI deep into the wafer interior.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not .08*np.pi*r*r<area<.91*np.pi*r*r:continue
        if cv2.pointPolygonTest(contour,tuple(seed),False)<0:continue
        perimeter=cv2.arcLength(contour,True)
        circularity=4*np.pi*area/max(perimeter*perimeter,1)
        if circularity<.3:continue
        candidates.append((area*circularity,contour))
    if not candidates:raise DetectionError('未识别到与亮膜分离的晶圆真实外沿，未用拟合圆代替晶圆距离')
    contour=max(candidates,key=lambda item:item[0])[1].reshape(-1,2)
    p=(contour+.5)/[sx,sy]-.5
    polygon=Polygon(p)
    if not polygon.is_valid:polygon=polygon.buffer(0)
    if polygon.geom_type=='MultiPolygon':polygon=max(polygon.geoms,key=lambda part:part.area)
    if polygon.geom_type!='Polygon' or polygon.is_empty:raise DetectionError('晶圆外沿连通区域无效')
    # A narrow white ink loop can isolate an outer strip of real wafer from the
    # central component. Its convex hull is only a search seed: original-image
    # profiles locate the LAST supported wafer-to-film transition, including
    # disconnected outer strips. Neither hull nor fitted circle is exported.
    hull=np.asarray(polygon.convex_hull.exterior.coords)
    angles=np.arange(2880,dtype=float)*.125
    seeds=radial_intersections(hull,center,angles)
    if any(not len(v) for v in seeds):raise DetectionError('晶圆候选不能围住圆心')
    seed_r=np.array([v[-1] for v in seeds]);typical=float(np.median(seed_r))
    # An outer film shadow can join the central dark component, expanding its
    # hull far beyond the wafer. Cover the normal wafer radius even for these
    # outward-biased seeds; the hull still only limits a measured edge search.
    inner=max(80,math.ceil(typical*.15),
              math.ceil(float(seed_r.max()-np.percentile(seed_r,10))+typical*.15))
    outer=max(40,math.ceil(typical*.06))
    inner=min(inner,max(2,math.floor(float(seed_r.min())-1)))
    offsets=np.arange(-inner,outer+1,dtype=float)
    theta=np.radians(angles);vectors=np.c_[np.cos(theta),-np.sin(theta)]
    positions=center+vectors[:,None,:]*(seed_r[:,None]+offsets)[...,None]
    in_frame=(positions[:,:,0]>=1)&(positions[:,:,0]<=w-2)&(positions[:,:,1]>=1)&(positions[:,:,1]<=h-2)
    profiles=_sample(gray,positions)
    # Five neighboring 0.125-degree profiles suppress thin white ink cuts,
    # while each radial transition remains measured at original resolution.
    profiles=median_filter(profiles,size=(5,1),mode='wrap')
    profiles=gaussian_filter1d(profiles,.8,axis=1)
    # Learn the bright film inside the entire search band. Sampling only past
    # an inflated hull learns the outer shadow instead of the intervening film.
    if in_frame.all():film=np.percentile(profiles,90,axis=1)
    else:
        field=np.where(in_frame,profiles,np.nan)
        if not np.isfinite(field).any(axis=1).all():raise DetectionError('晶圆外侧亮膜超出图像范围，无法测量全部方向')
        film=np.nanpercentile(field,90,axis=1)
    # Holes can occupy nearly the entire inward band while a real outer strip
    # remains. Use the lower tail across the whole band, not its inner median.
    wafer=np.percentile(profiles,2,axis=1)
    level=film-np.maximum(3.,(film-wafer)*.25)
    dark=(profiles<level[:,None])
    above=profiles>=level[:,None]
    cumulative=np.pad(np.cumsum(above,axis=1),((0,0),(1,0)))
    runs=cumulative[:,6:]-cumulative[:,:-6]
    transition=dark[:,:-6]&(runs[:,1:]>=5)&(profiles[:,1:-5]>=level[:,None])
    if not in_frame.all():
        usable=np.pad(np.cumsum(in_frame,axis=1),((0,0),(1,0)))
        transition&=in_frame[:,:-6]&((usable[:,7:]-usable[:,1:-6])==6)
    indices=np.max(np.where(transition,np.arange(len(offsets)-6)[None,:],-1),axis=1)
    valid=(indices>=0)&(indices<len(offsets)-2)&(film>wafer+6)
    if not valid.all():
        raise DetectionError(f'{int((~valid).sum())}/{len(valid)} 个方向缺少可靠晶圆外沿；'
                             '请检查遮挡、外沿对比度或图像范围，未用拟合圆补齐距离')
    coarse_r=seed_r+offsets[indices]
    angular_median=median_filter(coarse_r,size=33,mode='wrap')
    spread=median_filter(np.abs(coarse_r-angular_median),size=33,mode='wrap')
    # Opposite inward/outward pen-stroke errors must not inflate the local
    # noise allowance until a several-hundred-pixel exterior spike is accepted.
    allowance=np.clip(spread*5,3.,max(12.,typical*.02))
    isolated=coarse_r>angular_median+allowance
    # Reject isolated dust outside the wafer by selecting an EARLIER measured
    # transition with angular support. Never substitute the angular median as
    # an invented edge, and preserve inward notches and extended flat edges.
    candidate_r=seed_r[:,None]+offsets[None,:-6]
    supported=transition&(candidate_r<=angular_median[:,None]+allowance[:,None])
    supported&=(candidate_r>=angular_median[:,None]-np.maximum(20.,allowance*3)[:,None])
    alternative=np.max(np.where(supported,np.arange(len(offsets)-6)[None,:],-1),axis=1)
    corrected=isolated&(alternative>=0);indices[corrected]=alternative[corrected]
    # A white pen stroke may cross the exterior, hiding the same-angle dark
    # side entirely. For these remaining isolated outward spikes only, suppress
    # the stroke with a wider angular profile filter. Accept a transition only
    # close to neighboring observed edges; do not replace it with a circle or
    # with an arbitrary median radius. Retain the recovery count for review.
    rows=np.arange(len(angles));left=profiles[rows,indices];right=profiles[rows,indices+1]
    surrounding_film=median_filter(film,size=33,mode='wrap')
    inward_ink=(coarse_r<angular_median-np.maximum(20.,allowance*3))
    inward_ink&=film>surrounding_film+np.maximum(10.,surrounding_film*.15)
    unresolved=(isolated&(alternative<0))|inward_ink
    recovered=np.zeros(len(angles),bool)
    for row in np.flatnonzero(unresolved):
        neighbors=(row+np.arange(-16,17))%len(angles)
        # Align neighboring observations at absolute radii, rather than at a
        # potentially shadow-biased hull offset. Only a few ink-affected rows
        # need this wider filter; do not smooth the whole physical exterior.
        radius_grid=seed_r[row]+offsets
        aligned=np.array([np.interp(radius_grid,seed_r[n]+offsets,profiles[n],left=np.nan,right=np.nan)
                          for n in neighbors])
        available=np.isfinite(aligned).sum(axis=0)>=17
        profile=np.nanmedian(aligned,axis=0)
        support=available&in_frame[row]
        if not support.any():continue
        bright=float(np.percentile(profile[support],90));dark=float(np.percentile(profile[support],2))
        if bright<=dark+6:continue
        cutoff=bright-max(3.,(bright-dark)*.25)
        above=profile>=cutoff;cumulative=np.r_[0,np.cumsum(above)]
        runs=cumulative[6:]-cumulative[:-6]
        possible=(~above[:-6])&(runs[1:]>=5)&above[1:-5]
        support_runs=np.r_[0,np.cumsum(support)]
        possible&=(support_runs[7:]-support_runs[:-7])==7
        possible&=(candidate_r[row]<=angular_median[row]+allowance[row])
        possible&=(candidate_r[row]>=angular_median[row]-max(20.,allowance[row]*3))
        candidates=np.flatnonzero(possible)
        if not len(candidates):continue
        index=int(candidates[-1]);indices[row]=index;level[row]=cutoff
        left[row]=profile[index];right[row]=profile[index+1];recovered[row]=True
    if (unresolved&~recovered).any():
        raise DetectionError('部分晶圆外沿受标记或遮挡影响，邻域图像仍缺少可靠边缘，请复核图片')
    fraction=np.clip(np.divide(level-left,right-left,out=np.zeros(len(angles)),where=np.abs(right-left)>1e-6),0,1)
    radii=seed_r+offsets[indices]+fraction
    p=center+vectors*radii[:,None]
    p=np.vstack([p,p[:1]])
    if not Polygon(p).covers(Point(center)):raise DetectionError('拟合圆心不在晶圆外沿内，无法生成完整 720 行晶圆距离')
    return {'points':p,'threshold_gray':float(threshold),'refined_fraction':float(valid.mean()),
            'dust_transition_rejections':int(corrected.sum()),
            'angular_profile_recoveries':int(recovered.sum()),
            'angular_profile_recovery_angles_deg':angles[recovered].tolist(),
            'search_inward_px':int(inner),'search_outward_px':int(outer),
            'point_count':len(p),'seconds':time.perf_counter()-started,
            'method':'adaptive_central_wafer_component_original_pixel_exterior'}


def ensure_wafer_outline(analysis):
    if 'wafer_outline' not in analysis._geometry_cache:
        ring=ensure_bright_ring(analysis,{},required=True)
        result=detect_wafer_outline(analysis.gray,ring)
        analysis._geometry_cache['wafer_outline']=result
        analysis.metadata['wafer_outline']={k:v for k,v in result.items() if k!='points'}
    return analysis._geometry_cache['wafer_outline']


def _polar_working_angles(angles, linear, angle_reference, use_regression):
    """Map labelled image directions to output XY without relabelling theta."""
    angles=np.asarray(angles,float)
    if not use_regression or angle_reference=='machine':return angles
    theta=np.radians(angles)
    directions=np.c_[np.cos(theta),-np.sin(theta)]@np.asarray(linear,float).T
    return np.degrees(np.arctan2(directions[:,1],directions[:,0]))%360


def _interior_roi_radii(radii, wafer_radius):
    """Keep interior hits only; a coincident/output-rounded edge is W once."""
    radii=np.asarray(radii,float)
    if not len(radii):return radii
    # Match CSV/TXT and the preview labels exactly. Two distinct floating-point
    # values may print as the same 3-decimal edge distance and must not create
    # an additional P point at that edge.
    rounded=np.fromiter((float(f'{radius:.3f}') for radius in radii),float,count=len(radii))
    return radii[(radii<float(wafer_radius))&(rounded<float(f'{wafer_radius:.3f}'))]


def polar_ray_table(analysis, options=None, roi_index=0):
    """720 rows of interior ROI hits followed by the physical wafer radius."""
    ring=ensure_bright_ring(analysis,{},required=True)
    options=(options or ExportConfig()).validate()
    if options.use_regression and tuple(analysis.gray.shape)!=tuple(options.calibration_shape):
        raise ValueError('当前图片尺寸与工作台标定尺寸不一致，不能转换机台极坐标')
    center=np.asarray(ring['center_image_xy_px'],float)
    p=np.asarray(analysis.rois[roi_index].points,float)
    linear,offset=_export_affine(analysis,options) if options.use_regression else (np.eye(2),np.zeros(2))
    working_center=center@linear.T+offset
    working_points=p@linear.T+offset
    y_sign=1. if options.use_regression else -1.
    reference=options.polar_angle_reference if options.use_regression else 'image'
    digest=hashlib.blake2b(np.ascontiguousarray(p).view(np.uint8),digest_size=12).digest()
    key=('polar_rays',roi_index,digest,tuple(center),options.use_regression,reference,tuple(linear.ravel()),tuple(offset))
    if key in analysis._geometry_cache:return analysis._geometry_cache[key]
    started=time.perf_counter();angles=np.arange(720,dtype=float)*.5
    working_angles=_polar_working_angles(angles,linear,reference,options.use_regression)
    outline=ensure_wafer_outline(analysis)
    working_outline=outline['points']@linear.T+offset
    outer_hits=radial_intersections(working_outline,working_center,working_angles,y_sign)
    if any(not len(hits) for hits in outer_hits):raise DetectionError('部分角度缺少晶圆真实外沿交点，未生成不完整的极坐标文件')
    outer=np.array([hits[-1] for hits in outer_hits])
    raw_hits=radial_intersections(working_points,working_center,working_angles,y_sign)
    # The physical exterior is the terminal cut position. Do not emit ROI hits
    # at or past W, including duplicates after 3-decimal output rounding.
    hits=[_interior_roi_radii(values,r) for values,r in zip(raw_hits,outer)]
    maximum=max(map(len,hits),default=0)
    headers=['id','theta']+[f'r{i+1}' for i in range(maximum)]+['r_wafer']
    rows=[[i+1,float(angle),*values.tolist(),float(r)]
          for i,(angle,values,r) in enumerate(zip(angles,hits,outer))]
    directions=np.c_[np.cos(np.radians(working_angles)),y_sign*np.sin(np.radians(working_angles))]
    image_scale=np.linalg.norm(directions@np.linalg.inv(linear).T,axis=1)
    result={'center_image_xy_px':center.tolist(),'center_output_xy':working_center.tolist(),
            'center_machine_xy':working_center.tolist() if options.use_regression else None,
            'space':'machine' if options.use_regression else 'image','unit':'机台单位' if options.use_regression else 'px',
            'linear':linear,'offset':offset,'y_sign':y_sign,
            'angles_deg':angles,'working_angles_deg':working_angles,'angle_reference':reference,
            'roi_radii':hits,'wafer_radii':outer,
            'roi_radii_px':[r*s for r,s in zip(hits,image_scale)],'wafer_radii_px':outer*image_scale,
            'headers':headers,'rows':rows,'max_roi_intersections':maximum,
            'roi_point_count':sum(map(len,hits)),
            'discarded_roi_point_count':sum(map(len,raw_hits))-sum(map(len,hits)),
            'wafer_reference_count':len(outer),
            'outside_roi_intersections':sum(int(np.count_nonzero(values>r+1e-6)) for values,r in zip(raw_hits,outer)),
            'roi_outside_wafer_max':max((max(0.,float(values[-1]-r)) for values,r in zip(raw_hits,outer) if len(values)),default=0.),
            'roi_outside_wafer_max_px':max((max(0.,float(values[-1]-r))*s for values,r,s in zip(raw_hits,outer,image_scale) if len(values)),default=0.),
            'seconds':time.perf_counter()-started,'angle_step_deg':.5,'row_count':720}
    # Keep only a few review revisions, never grow with every edit/angle.
    for old in [k for k in analysis._geometry_cache if isinstance(k,tuple) and k[0]=='polar_rays'][:-2]:
        analysis._geometry_cache.pop(old,None)
    analysis._geometry_cache[key]=result
    return result


def polar_ray_at_angle(analysis, angle_deg, options=None, roi_index=0):
    angle=float(angle_deg)%360
    if not math.isfinite(angle):raise ValueError('查看角度必须为有限数值')
    table=polar_ray_table(analysis,options,roi_index)
    working_angle=float(_polar_working_angles([angle],table['linear'],table['angle_reference'],table['space']=='machine')[0])
    index=int(round(angle*2))%720
    if abs(angle-index*.5)<1e-9:
        radii=table['roi_radii'][index];outer=float(table['wafer_radii'][index])
    else:
        linear=table['linear'];offset=table['offset'];center=table['center_output_xy'];sign=table['y_sign']
        outer=float(radial_intersections(ensure_wafer_outline(analysis)['points']@linear.T+offset,center,[working_angle],sign)[0][-1])
        values=radial_intersections(analysis.rois[roi_index].points@linear.T+offset,center,[working_angle],sign)[0]
        radii=_interior_roi_radii(values,outer)
    center=np.asarray(table['center_image_xy_px'],float);theta=math.radians(working_angle)
    direction=np.array([math.cos(theta),table['y_sign']*math.sin(theta)])
    image_direction=np.linalg.solve(table['linear'],direction)
    image_scale=float(np.linalg.norm(image_direction))
    return {'angle_deg':angle,'space':table['space'],'unit':table['unit'],
            'angle_reference':table['angle_reference'],
            'machine_angle_deg':working_angle if table['space']=='machine' else None,
            'image_angle_deg':math.degrees(math.atan2(-image_direction[1],image_direction[0]))%360,
            'center_image_xy_px':center.tolist(),'center_machine_xy':table['center_machine_xy'],
            'roi_radii':radii.tolist(),'wafer_radius':outer,
            'roi_radii_px':(radii*image_scale).tolist(),'wafer_radius_px':outer*image_scale,
            'roi_points_image_xy_px':(center+radii[:,None]*image_direction).tolist(),
            'wafer_point_image_xy_px':(center+outer*image_direction).tolist()}


def _coordinate_system(overlay, origin, y_sign, mode, machine_linear=None, angle_reference='image'):
    h,w=overlay.shape[:2];length=min(h,w);line=max(1,round(length/2600))
    font=max(.5,length/2200);color=(205,155,75);margin=max(12,round(length*.012))
    ox,oy=origin;center=(round(ox),round(oy))
    def text(label,at):
        size,_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,font,line)
        x=max(3,min(w-size[0]-3,round(at[0])));y=max(size[1]+3,min(h-4,round(at[1])))
        cv2.putText(overlay,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,font,color,line,cv2.LINE_AA)
    def interval(value):
        power=10**math.floor(math.log10(max(value,1e-12)))
        return next(factor*power for factor in (1,2,5,10) if factor*power>=value)
    if mode=='polar' and machine_linear is not None:
        linear=np.asarray(machine_linear,float);inverse=np.linalg.inv(linear)
        corners=np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],float)-[ox,oy]
        maximum=float(np.linalg.norm(corners@linear.T,axis=1).max());step=interval(maximum/5)
        angles=np.linspace(0,2*np.pi,241);vectors=np.c_[np.cos(angles),np.sin(angles)]@inverse.T
        for radius in np.arange(step,maximum+step*.1,step):
            contour=np.asarray(origin)+radius*vectors
            cv2.polylines(overlay,[np.round(contour).astype('int32')],True,color,line,cv2.LINE_AA)
            for label in contour[::30]:
                if margin<label[0]<w-margin and margin<label[1]<h-margin:
                    text(f'{radius:g}',label+[margin/3,-margin/3]);break
        for angle in range(0,360,45):
            rad=math.radians(angle)
            if angle_reference=='image':
                vector=np.array([math.cos(rad),-math.sin(rad)])
                vector/=np.linalg.norm(linear@vector)
            else:vector=inverse@np.array([math.cos(rad),math.sin(rad)])
            end=tuple(np.round(np.asarray(origin)+vector*maximum*3).astype(int))
            visible,a,b=cv2.clipLine((0,0,w,h),center,end)
            if visible:
                cv2.line(overlay,a,b,color,line,cv2.LINE_AA)
                text(f'{angle} deg',np.asarray(b)*.94+np.asarray(center)*.06)
    elif mode=='polar':
        maximum=max(math.hypot(x-ox,y-oy) for x in (0,w-1) for y in (0,h-1))
        step=interval(maximum/5)
        for radius in np.arange(step,maximum+1,step):
            cv2.circle(overlay,center,round(radius),color,line,cv2.LINE_AA)
            for angle in (45,0,90,135,225,315,180,270):
                rad=math.radians(angle)
                label=(ox+radius*math.cos(rad),oy+y_sign*radius*math.sin(rad))
                if margin<label[0]<w-margin and margin<label[1]<h-margin:
                    text(f'{radius:g}',(label[0]+margin/3,label[1]-margin/3));break
        for angle in range(0,360,45):
            rad=math.radians(angle);end=(round(ox+maximum*math.cos(rad)),round(oy+y_sign*maximum*math.sin(rad)))
            visible,a,b=cv2.clipLine((0,0,w,h),center,end)
            if not visible or np.linalg.norm(np.asarray(a)-b)<margin:continue
            cv2.line(overlay,a,b,color,line,cv2.LINE_AA)
            label=np.asarray(b)*.94+np.asarray(center)*.06
            text(f'{angle} deg',label)
    else:
        cv2.line(overlay,(0,round(oy)),(w-1,round(oy)),color,line,cv2.LINE_AA)
        cv2.line(overlay,(round(ox),0),(round(ox),h-1),color,line,cv2.LINE_AA)
        xe=(w-1,round(oy));ye=(round(ox),h-1 if y_sign>0 else 0)
        if w-1-ox>margin:cv2.arrowedLine(overlay,center,xe,color,line,cv2.LINE_AA,tipLength=.015)
        if abs(ye[1]-oy)>margin:cv2.arrowedLine(overlay,center,ye,color,line,cv2.LINE_AA,tipLength=.015)
        text('X (px)',(w-margin*6,oy-margin/2 if oy>margin*2 else oy+margin*2))
        text('Y (px)',(ox+margin/2,h-margin if y_sign>0 else margin*2))
        step=interval(max(h,w)/6)
        for x in np.arange(ox%step,w,step):
            if abs(x-ox)<step*.05:continue
            cv2.line(overlay,(round(x),round(oy-margin/3)),(round(x),round(oy+margin/3)),color,line)
            text(f'{x-ox:g}',(x+margin/3,oy+margin*1.5))
        for y in np.arange(oy%step,h,step):
            if abs(y-oy)<step*.05:continue
            cv2.line(overlay,(round(ox-margin/3),round(y)),(round(ox+margin/3),round(y)),color,line)
            text(f'{(y-oy)*y_sign:g}',(ox+margin/2,y-margin/3))


def render_image(gray, style=None, points=None, bright_ring=None):
    """Render cached auxiliary geometry; rendering never runs a circle fit."""
    style = style or {}
    colors = style.get('colors', [DEFAULT_COLORS[0]])
    width = int(style.get('width', 3))
    if not colors or not 1 <= width <= 30:
        raise ValueError('需要至少一种颜色；标注线宽应为 1～30')
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = colors[0].lstrip('#')
    bgr = tuple(int(color[j:j + 2], 16) for j in (4, 2, 0))
    center=bright_ring.get('center_image_xy_px') if bright_ring and bright_ring.get('status')=='ok' else None
    try:origin,y_sign=coordinate_frame(gray.shape,style.get('coordinate_origin','center'),center)
    except ValueError:origin=None;y_sign=-1.
    axes_mode=style.get('axes_mode','cartesian')
    axes_origin=center if axes_mode=='polar' else origin
    if (style.get('show_axes',False) and axes_origin is not None and
            not (axes_mode=='polar' and style.get('polar_machine') and style.get('polar_linear') is None)):
        _coordinate_system(overlay,axes_origin,-1. if axes_mode=='polar' else y_sign,axes_mode,
                           style.get('polar_linear') if axes_mode=='polar' and style.get('polar_machine') else None,
                           style.get('polar_angle_reference','image'))
    if bright_ring and bright_ring.get('status')=='ok':
        center=tuple(round(v) for v in bright_ring['center_image_xy_px'])
        ring_color=(255,205,45);ring_width=max(1,round(min(gray.shape)/2200))
        if style.get('show_ring_roi',False):cv2.circle(overlay,center,round(bright_ring['radius_px']),ring_color,ring_width,cv2.LINE_AA)
        if style.get('show_ring_center',False):cv2.drawMarker(overlay,center,ring_color,cv2.MARKER_TILTED_CROSS,max(16,round(min(gray.shape)*.009)),ring_width)
    if points is not None:
        cv2.polylines(overlay,[np.round(points[:-1]).astype('int32')],True,bgr,width,cv2.LINE_AA)
    if style.get('show_wafer_outline') and style.get('wafer_outline') is not None:
        exterior=np.asarray(style['wafer_outline'],float)
        cv2.polylines(overlay,[np.round(exterior).astype('int32')],True,(220,80,255),max(1,width),cv2.LINE_AA)
    if style.get('show_origin', True) and origin is not None:
        x,y=origin
        cv2.drawMarker(overlay, (round(x), round(y)), (0, 0, 230), cv2.MARKER_CROSS, 20, 2)
    ray=style.get('polar_ray')
    if style.get('show_polar_ray') and ray:
        thickness=max(1,round(min(gray.shape)/1800));size=max(12,round(min(gray.shape)/250))
        c=tuple(np.round(ray['center_image_xy_px']).astype(int))
        endpoint=tuple(np.round(ray['wafer_point_image_xy_px']).astype(int))
        cv2.line(overlay,c,endpoint,(40,210,255),thickness,cv2.LINE_AA)
        cv2.drawMarker(overlay,c,(255,205,45),cv2.MARKER_CROSS,size,thickness)
        locations=ray['roi_points_image_xy_px']+[ray['wafer_point_image_xy_px']]
        radii=ray['roi_radii']+[ray['wafer_radius']]
        for i,(p,radius) in enumerate(zip(locations,radii),1):
            xy=tuple(np.round(p).astype(int));outer=i==len(locations)
            color=(220,80,255) if outer else (0,210,255)
            cv2.drawMarker(overlay,xy,color,cv2.MARKER_DIAMOND if outer else cv2.MARKER_CROSS,size,thickness)
            label=f'{"W" if outer else "P"+str(i)} {radius:.3f}'
            cv2.putText(overlay,label,(xy[0]+size//2,xy[1]-size if i%2 else xy[1]+size*2),
                        cv2.FONT_HERSHEY_SIMPLEX,max(.5,min(gray.shape)/4200),color,thickness,cv2.LINE_AA)
    return overlay


def render_analysis(analysis, style=None):
    return render_image(analysis.gray,style,analysis.rois[0].points,analysis.metadata.get('bright_ring'))


def ensure_bright_ring(analysis, style=None, required=False):
    style=style or {}
    if (required or any(style.get(key,False) for key in ('show_ring_center','show_ring_roi','require_ring','show_polar_ray','show_wafer_outline'))
            or style.get('coordinate_origin')=='ring_center') and 'bright_ring' not in analysis.metadata:
        try:analysis.metadata['bright_ring']=fit_bright_ring(analysis.gray)
        except Exception as exc:analysis.metadata['bright_ring']={'status':'not_found','reason':'圆环拟合失败：'+str(exc)}
    result=analysis.metadata.get('bright_ring')
    if required and (not result or result.get('status')!='ok'):
        raise DetectionError('极坐标/圆心原点需要可信的亮圆环拟合：'+(result or {}).get('reason','尚未拟合'))
    return result


def _span_distances(p, a, b):
    v=b-a;length=float(v@v)
    if length<1e-24:return np.linalg.norm(p-a,axis=1)
    t=np.clip((p-a)@v/length,0,1)
    return np.linalg.norm(p-a-t[:,None]*v,axis=1)


def _span_costs(p,starts,ends,rounding):
    """Certify many original arcs in one NumPy batch, not a Python point loop."""
    starts=np.asarray(starts,int);ends=np.asarray(ends,int)
    lengths=ends-starts+1
    offsets=np.r_[0,np.cumsum(lengths)[:-1]]
    groups=np.repeat(np.arange(len(starts)),lengths)
    positions=np.repeat(starts,lengths)+np.arange(int(lengths.sum()))-np.repeat(offsets,lengths)
    v=p[ends]-p[starts];vv=np.maximum(np.sum(v*v,axis=1),1e-24)
    relative=p[positions]-p[starts[groups]]
    t=np.clip(np.sum(relative*v[groups],axis=1)/vv[groups],0,1)
    difference=relative-t[:,None]*v[groups]
    squared=np.sum(difference*difference,axis=1)
    deviation=np.sqrt(np.maximum.reduceat(squared,offsets))
    return deviation+np.maximum(rounding[starts],rounding[ends])


def machine_affine(shape,coefficients,origin='center',ring_center=None):
    """Convert ORIGINAL image pixel coordinates into the calibrated machine XY."""
    coef=np.asarray(coefficients,float)
    if coef.shape!=(2,3) or not np.isfinite(coef).all() or np.linalg.matrix_rank(coef[:,:2])<2:
        raise ValueError('工作台仿射方程无效或退化')
    offset,sign=coordinate_frame(shape,origin,ring_center)
    linear=coef[:,:2]@np.diag([1.,sign])
    return linear,coef[:,2]-linear@np.asarray(offset)


def _export_affine(analysis, options):
    use=options.use_regression
    key=options.regression_origin if use else options.origin
    center=ensure_bright_ring(analysis,required=True)['center_image_xy_px'] if key=='ring_center' else None
    origin,sign=coordinate_frame(analysis.gray.shape,key,center)
    frame=np.diag([1.,sign])
    if use:
        return machine_affine(analysis.gray.shape,options.regression_coefficients,key,center)
    else:
        linear=frame;offset=-linear@np.asarray(origin)
    return linear,offset


def machine_spaced_path(points,linear,offset,step=.5):
    """Ordered machine path: <=0.5 between rounded output XY, <=1px shape error.

    Uniform machine arc locations reduce dense runs. Image extrema and local
    error-driven refinements retain corners. No global X sorting is performed.
    The source curve is never changed; inserted points lie on its segments.
    """
    started=time.perf_counter();p=np.asarray(points,float)
    linear=np.asarray(linear,float);offset=np.asarray(offset,float);inverse=np.linalg.inv(linear)
    xy=p@linear.T+offset
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    if not np.isfinite(arc[-1]) or arc[-1]<=0:raise ValueError('机台轨迹长度无效')
    # Two endpoints each round by at most sqrt(2)*0.0005 machine units.
    spacing=step-math.sqrt(2)*.001-1e-6
    if spacing<=0:raise ValueError('机台间距小于三位小数可保证的精度')
    count=max(1,math.ceil(float(arc[-1])/spacing))
    if count>2_000_000:raise ValueError('机台轨迹尺度过大，0.5 模式将产生过多点，请核对回归单位')
    mandatory=sorted({0,len(p)-1,int(np.argmin(p[:-1,0])),int(np.argmax(p[:-1,0])),
                      int(np.argmin(p[:-1,1])),int(np.argmax(p[:-1,1])),
                      int(np.argmin(xy[:-1,0])),int(np.argmax(xy[:-1,0])),
                      int(np.argmin(xy[:-1,1])),int(np.argmax(xy[:-1,1]))})
    base=np.unique(np.r_[np.linspace(0,arc[-1],count+1),arc[mandatory]])
    augmented=np.unique(np.r_[arc,base])
    source=np.c_[np.interp(augmented,arc,p[:,0]),np.interp(augmented,arc,p[:,1])]
    machine=np.round(source@linear.T+offset,3);machine[np.abs(machine)<.0005]=0
    restored=(machine-offset)@inverse.T
    rounding=np.linalg.norm(restored-source,axis=1)
    initial=np.searchsorted(augmented,base)

    def select(tolerance):
        if np.any(rounding[initial]>tolerance):raise ValueError('机台三位小数精度不足以保持轨迹的像素误差要求')
        keep=set(map(int,initial));values=_span_costs(source,initial[:-1],initial[1:],rounding)
        stack=[(int(a),int(b)) for a,b,value in zip(initial[:-1],initial[1:],values) if value>tolerance-1e-9]
        while stack:
            a,b=stack.pop()
            distances=_span_distances(source[a:b+1],source[a],source[b])
            if float(distances.max())+max(rounding[a],rounding[b])<=tolerance-1e-9:continue
            if b-a<=1:raise ValueError('机台三位小数舍入无法满足轨迹精度')
            mid=a+1+int(np.argmax(distances[1:-1]))
            if rounding[mid]>tolerance:raise ValueError('机台坐标精度不足，请核对工作台回归比例')
            keep.add(mid);stack.extend(((a,mid),(mid,b)))
        indices=np.array(sorted(keep),int)
        # Remove exact repeated exported positions when doing so retains the
        # same original-arc error certificate and the closing point.
        unique=np.r_[True,np.any(np.diff(machine[indices],axis=0)!=0,axis=1)];unique[-1]=True
        candidate=indices[unique]
        if len(candidate)>=4 and _span_costs(source,candidate[:-1],candidate[1:],rounding).max()<=tolerance:
            indices=candidate
        # Refining a corner can leave a uniform sample very close beside it.
        # Remove such redundant nodes only when BOTH the rounded 0.5 machine
        # step and the original-image error certificate remain satisfied.
        m=len(indices);left=np.arange(m)-1;right=np.arange(m)+1
        active=np.ones(m,bool);versions=np.zeros(m,int);heap=[]
        protected=set(np.searchsorted(augmented,arc[mandatory]).tolist())
        def offer(k):
            if k<=0 or k>=m-1 or not active[k] or indices[k] in protected:return
            versions[k]+=1
            a,b=int(left[k]),int(right[k]);ia,ib=int(indices[a]),int(indices[b])
            if np.linalg.norm(machine[ib]-machine[ia])>step+1e-10:return
            error=float(_span_distances(source[ia:ib+1],source[ia],source[ib]).max())+max(rounding[ia],rounding[ib])
            if error>tolerance-1e-9:return
            proximity=min(float(np.linalg.norm(machine[indices[k]]-machine[ia])),
                          float(np.linalg.norm(machine[ib]-machine[indices[k]])))
            heapq.heappush(heap,(proximity,k,int(versions[k])))
        for k in range(1,m-1):offer(k)
        while heap:
            _,k,version=heapq.heappop(heap)
            if not active[k] or version!=versions[k]:continue
            a,b=int(left[k]),int(right[k]);active[k]=False;right[a]=b;left[b]=a
            offer(a);offer(b)
        indices=indices[active]
        return indices

    for tolerance in (1.,.5,.2,.05):
        indices=select(tolerance)
        q=restored[indices]
        if Polygon(q).is_valid and LineString(q).is_simple:break
    else:raise ValueError('机台选点后无法保持无自交闭环，请检查轮廓或回归精度')
    certified=float(_span_costs(source,indices[:-1],indices[1:],rounding).max())
    steps=np.linalg.norm(np.diff(machine[indices],axis=0),axis=1)
    if float(steps.max())>step+1e-10:raise ValueError('机台相邻点间距校验失败')
    nearest=np.clip(np.searchsorted(arc,augmented[indices],side='right')-1,0,len(p)-1)
    nearest[-1]=len(p)-1
    return {'image_points':restored[indices],'source_image_points':source[indices],
            'indices':nearest,'source_arc_machine':augmented[indices],
            'cartesian':machine[indices],'polar':None,'original_count':len(p),'point_count':len(indices),
            'tolerance_px':tolerance,'certified_error_px':certified,'selection_mode':'machine',
            'max_machine_step':step,'actual_max_machine_step':float(steps.max()),
            'actual_min_machine_step':float(steps.min()),'prepare_seconds':time.perf_counter()-started}


def coordinate_path(analysis, options=None, roi_index=0):
    """One cached path for preview, manual save, review save and TCP output.

    Tolerance is in original-image pixels, including quantization in the
    selected output systems. The detection/review point array is never edited.
    """
    options=(options or ExportConfig()).validate()
    if options.use_regression and tuple(analysis.gray.shape)!=tuple(options.calibration_shape):
        raise ValueError('当前图片尺寸与工作台标定尺寸不一致，请重新标定')
    p=np.asarray(analysis.rois[roi_index].points,dtype=float)
    if len(p)<4 or not np.allclose(p[0],p[-1],atol=1e-8,rtol=0):
        raise ValueError('待导出的 ROI 必须为完整闭环')
    digest=hashlib.blake2b(np.ascontiguousarray(p).view(np.uint8),digest_size=12).digest()
    key=(roi_index,digest,tuple(analysis.gray.shape),options.simplify_px,options.machine_step,options.origin,options.use_regression,
         tuple(np.asarray(options.regression_coefficients).ravel()) if options.use_regression else (),
         options.regression_origin,options.cartesian)
    cache=analysis._coordinate_cache
    if key in cache:return cache[key]
    started=time.perf_counter()
    linear,offset=_export_affine(analysis,options)
    if options.machine_step is not None:
        result=machine_spaced_path(p,linear,offset,options.machine_step)
        if len(cache)>=6:cache.clear()
        cache[key]=result;return result
    inverse=np.linalg.inv(linear)
    xy=p@linear.T+offset
    views={};numbers={}
    # Polar rays are computed separately from the complete ROI. Point selection
    # and its error certificate now concern the Cartesian closed path only.
    cart=np.round(xy,3);cart[np.abs(cart)<.0005]=0
    numbers['cartesian']=cart
    views['cartesian']=(cart-offset)@inverse.T
    rounding=np.maximum.reduce([np.linalg.norm(q-p,axis=1) for q in views.values()])
    n=len(p)-1
    anchors=sorted({0,n,int(np.argmin(p[:-1,0])),int(np.argmax(p[:-1,0])),
                    int(np.argmin(p[:-1,1])),int(np.argmax(p[:-1,1]))})
    tolerance=options.simplify_px

    def cost(a,b):
        return float(_span_distances(p[a:b+1],p[a],p[b]).max())+max(rounding[a],rounding[b])

    def valid(indices):
        return all(Polygon(q[indices]).is_valid and LineString(q[indices]).is_simple for q in views.values())

    def precision_error():
        return ValueError('当前坐标保留三位小数后的舍入误差无法满足 '+str(tolerance)+
            ' 像素；请选更大的像素档，或核对工作台回归比例')

    if tolerance is None:
        indices=np.arange(len(p))
        certified=float(rounding.max())
    else:
        if any(rounding[i]>tolerance for i in anchors):raise precision_error()
        # OpenCV performs the initial RDP in C. Exact source vertices and
        # full-resolution span checks below remove float32/cleanup ambiguity.
        tree=cKDTree(p[:-1])
        def select(epsilon):
            keep=set(anchors)
            for a,b in zip(anchors[:-1],anchors[1:]):
                approximate=cv2.approxPolyDP(p[a:b+1].astype(np.float32),epsilon,False).reshape(-1,2)
                _,near=tree.query(approximate)
                keep.update(int(i) for i in near if a<=i<=b)
            initial=np.array(sorted(keep),int)
            values=_span_costs(p,initial[:-1],initial[1:],rounding)
            stack=[(a,b) for a,b,value in zip(initial[:-1],initial[1:],values) if value>tolerance-1e-10]
            while stack:
                a,b=stack.pop()
                if cost(a,b)<=tolerance-1e-10:continue
                if b-a<=1:raise precision_error()
                distances=_span_distances(p[a+1:b],p[a],p[b])
                mid=a+1+int(np.argmax(distances))
                if rounding[mid]>tolerance:raise precision_error()
                keep.add(mid);stack.extend(((a,mid),(mid,b)))
            return np.array(sorted(keep),int)

        epsilon=max(tolerance*.05,tolerance-float(rounding.max())-1e-6)
        indices=select(epsilon)
        original_indices=indices
        # Remove locally redundant RDP vertices under the SAME original-arc
        # error bound. Mandatory extrema and the original seam remain fixed.
        m=len(indices);left=np.arange(m)-1;right=np.arange(m)+1
        active=np.ones(m,bool);versions=np.zeros(m,int);heap=[];protected=set(anchors)
        def offer(k,value=None):
            if k<=0 or k>=m-1 or not active[k] or indices[k] in protected:return
            if value is None:value=cost(indices[left[k]],indices[right[k]])
            versions[k]+=1;heapq.heappush(heap,(value,k,int(versions[k])))
        initial_costs=_span_costs(p,indices[:-2],indices[2:],rounding)
        for k,value in enumerate(initial_costs,1):offer(k,float(value))
        while heap:
            value,k,version=heapq.heappop(heap)
            if not active[k] or versions[k]!=version:continue
            if value>tolerance-1e-10:break
            a,b=int(left[k]),int(right[k]);active[k]=False;right[a]=b;left[b]=a
            offer(a);offer(b)
        indices=indices[active]
        if not valid(indices):indices=original_indices
        if not valid(indices):
            for factor in (.5,.2,.05):
                indices=select(max(.00001,epsilon*factor))
                if valid(indices):break
            else:raise ValueError('选点后无法保持无自交闭环，请使用更小像素档或原始点')
        certified=float(_span_costs(p,indices[:-1],indices[1:],rounding).max())
        if certified>tolerance+1e-8:raise ValueError('坐标选点误差校验失败')
    if not valid(indices):raise ValueError('坐标舍入后出现自交，请检查轮廓或回归精度')
    primary='cartesian' if 'cartesian' in views else 'polar'
    result={'image_points':views[primary][indices], 'source_image_points':p[indices],
            'indices':indices,'cartesian':numbers.get('cartesian',np.empty((0,2)))[indices] if 'cartesian' in numbers else None,
            'polar':numbers['polar'][indices] if 'polar' in numbers else None,
            'original_count':len(p),'point_count':len(indices),'tolerance_px':tolerance,
            'certified_error_px':float(certified),'prepare_seconds':time.perf_counter()-started}
    if len(cache)>=6:cache.clear()
    cache[key]=result
    return result


def export_analysis(analysis, out_dir, stem=None, style=None, options=None):
    started = time.perf_counter()
    options = (options or ExportConfig()).validate(require_output=True)
    use=options.use_regression
    if use and tuple(analysis.gray.shape) != tuple(options.calibration_shape):
        raise ValueError(f'图片尺寸 {analysis.gray.shape[1]}×{analysis.gray.shape[0]} 与工作台“{options.worktable_name}”标定尺寸 '
                         f'{options.calibration_shape[1]}×{options.calibration_shape[0]} 不一致，请重新标定')
    stem = stem or make_export_stem(analysis.metadata.get('source', 'result'))
    paths=[coordinate_path(analysis,options,i) for i in range(len(analysis.rois))]
    polar_tables=[polar_ray_table(analysis,options,i) for i in range(len(analysis.rois))] if options.polar else []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ring=analysis.metadata.get('bright_ring',{})
    origin, y_sign = coordinate_frame(analysis.gray.shape, options.origin,ring.get('center_image_xy_px'))
    columns = {'cartesian': ['id', 'x', 'y'], 'polar': polar_tables[0]['headers'] if polar_tables else ['id','theta','r_wafer']}
    if use:
        columns['cartesian'] = ['id', 'X', 'Y']
    exports = []; saved_files = []
    def flagged_ranges(flags):
        if flags is None:return []
        changes = np.diff(np.r_[False, np.asarray(flags, bool)[:-1], False].astype('int8'))
        return [[int(a), int(b - 1)] for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
    for roi_index,(roi,path) in enumerate(zip(analysis.rois,paths)):
        p=path['image_points']
        filenames = {}
        prefix = stem if len(analysis.rois) == 1 else f'{stem}_{roi.roi_id}'
        for kind, enabled in (('cartesian', options.cartesian), ('polar', options.polar)):
            if not enabled:continue
            kind_suffix = 'machine_'+kind if use else kind
            target = out / f'{prefix}_{kind_suffix}.{options.coordinate_format}'
            with target.open('w', newline='', encoding='utf-8-sig' if options.coordinate_format == 'csv' and kind=='cartesian' else 'utf-8') as f:
                writer = csv.writer(f, delimiter=',' if options.coordinate_format == 'csv' else '\t', lineterminator='\n')
                if kind=='polar':
                    table=polar_tables[roi_index]
                    writer.writerows([row[0]]+[f'{v:.3f}' for v in row[1:]] for row in table['rows'])
                else:
                    writer.writerow(columns[kind])
                    writer.writerows((i, f'{a:.3f}', f'{b:.3f}') for i, (a, b) in enumerate(path[kind], start=1))
            filenames[kind] = target.name; saved_files.append(target.name)
        exports.append({'roi_id': roi.roi_id, 'polarity': roi.polarity, 'coordinate_files': filenames,
            'review_point_ranges': flagged_ranges(np.asarray(roi.review)[path['indices']]),
            'operator_reviewed_point_ranges': flagged_ranges(np.asarray(roi.reviewed)[path['indices']] if roi.reviewed is not None else None),
            'threshold': roi.threshold, 'width_quartiles_px': roi.width_quartiles, **roi.info,
            'point_count':len(p),'original_point_count':len(roi.points),
            'selection_tolerance_px':path['tolerance_px'],'selection_error_bound_px':path['certified_error_px']})
        if options.machine_step is not None:
            exports[-1]['machine_spacing']={k:path[k] for k in ('max_machine_step','actual_max_machine_step','actual_min_machine_step')}
        if options.polar:
            table=polar_tables[roi_index]
            exports[-1]['polar']={k:table[k] for k in ('headers','row_count','angle_step_deg','max_roi_intersections',
                'roi_point_count','discarded_roi_point_count','wafer_reference_count','outside_roi_intersections','seconds')}
    overlay = None
    if options.result_image:
        overlay = f'{stem}.jpg'
        ensure_bright_ring(analysis,style)
        image_style={**(style or {}),'coordinate_origin':options.origin}
        image_style['polar_machine']=use
        image_style['polar_angle_reference']=options.polar_angle_reference if use else 'image'
        if use and analysis.metadata.get('bright_ring',{}).get('status')=='ok':
            image_style['polar_linear']=_export_affine(analysis,options)[0]
        try:
            if image_style.get('show_polar_ray'):
                image_style['polar_ray']=polar_ray_at_angle(analysis,image_style.get('polar_angle_deg',0),options)
            if image_style.get('show_wafer_outline'):image_style['wafer_outline']=ensure_wafer_outline(analysis)['points']
        except DetectionError as exc:
            # An optional viewing overlay must not invalidate Cartesian output.
            # Required polar output was fully validated before any file writes.
            image_style['show_polar_ray']=False;image_style['show_wafer_outline']=False
            analysis.metadata['polar_preview_warning']=str(exc)
        save_image(out / overlay, render_image(analysis.gray,image_style,paths[0]['image_points'],analysis.metadata.get('bright_ring')))
        saved_files.append(overlay)
    result = {**analysis.metadata, 'rois': exports, 'overlay': overlay, 'coordinate_columns': columns,
              'origin_image_xy_px': list(origin),
              'detection_origin_image_xy_px': analysis.metadata.get('origin_image_xy_px'),
              'export_options': asdict(options), 'export_origin_image_xy_px': list(origin),
              'coordinate_convention': {'x_positive': 'right', 'y_positive': 'down' if y_sign > 0 else 'up',
                  'angle_zero': 'right', 'angle_positive': 'clockwise' if y_sign > 0 else 'counterclockwise',
                  'distance_unit': 'pixel', 'angle_unit': 'degree', 'id_start': 1,
                  'closing_point_repeated': True, 'txt_delimiter': 'tab'},
              'saved_files': saved_files,
              'export_seconds': time.perf_counter() - started}
    if use:
        result['coordinate_system'] = 'machine'
        result['worktable'] = options.worktable_name
        result['regression_image_convention'] = 'worktable_origin_original_pixels'
        result['regression_image_origin'] = options.regression_origin
        result['coordinate_convention'].update(x_positive='machine_X', y_positive='machine_Y',
            angle_zero='machine_X_positive', angle_positive='counterclockwise_in_machine_XY', distance_unit='machine_unit')
    if options.polar:
        table=polar_tables[0]
        result['polar_convention']={'origin':'bright_ring_fitted_center','origin_image_xy_px':ring['center_image_xy_px'],
            'origin_machine_XY':table['center_machine_xy'],
            'angle_reference':table['angle_reference'],
            'angle_zero':'machine_X_positive' if table['angle_reference']=='machine' else 'image_right',
            'angle_positive':'counterclockwise_in_machine_XY' if table['angle_reference']=='machine' else 'counterclockwise_in_image',
            'angle_step_deg':.5,
            'angle_range_deg':[0.,359.5],'row_count':720,'distance_unit':'machine_unit' if use else 'pixel','angle_unit':'degree',
            'columns':'id,theta,near_to_far_ROI_distances,r_wafer','last_column':'r_wafer',
            'roi_boundary_policy':'strictly_inside_wafer; omit_edge_duplicates_at_3_decimals',
            'roi_point_count':table['roi_point_count'],'point_count_excludes_wafer_references':True,
            'variable_length_rows':True,'header_written':False,'unused_ROI_columns':'omitted',
            'closing_point_repeated':False,'regression_applies':use,
            'wafer_boundary':'measured_exterior_not_fitted_ring_circle'}
    primary=next((item['coordinate_files']['cartesian'] for item in exports if 'cartesian' in item['coordinate_files']),None)
    primary=primary or next((item['coordinate_files']['polar'] for item in exports if 'polar' in item['coordinate_files']),None)
    primary=primary or overlay
    return {'directory':str(out.resolve()),'primary':primary,'overlay':overlay,'files':saved_files,'metadata':result}


def main():
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--out', type=Path, default=Path('roi_result'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--polarity', choices=('black', 'white'))
    parser.add_argument('--black-threshold', type=int)
    parser.add_argument('--white-threshold', type=int)
    parser.add_argument('--step', type=float)
    parser.add_argument('--coordinate-format', choices=('csv', 'txt'), default='csv')
    parser.add_argument('--origin', choices=('top_left', 'center', 'ring_center'), default='center')
    parser.add_argument('--no-result-image', action='store_true')
    parser.add_argument('--no-polar', action='store_true')
    parser.add_argument('--no-cartesian', action='store_true')
    parser.add_argument('--simplify-px',type=float,choices=(.1,.25,.5,1.),help='Maximum added contour error including 3-decimal rounding')
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    for name in ('expected_count', 'polarity', 'black_threshold', 'white_threshold'):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    if args.step is not None:
        values['sample_step_px'] = args.step
    options = ExportConfig(result_image=not args.no_result_image, polar=not args.no_polar,
        cartesian=not args.no_cartesian, coordinate_format=args.coordinate_format, origin=args.origin,simplify_px=args.simplify_px).validate(require_output=True)
    analysis = detect_image(args.image, DetectorConfig.from_dict(values))
    analysis.metadata['source'] = str(args.image)
    analysis.metadata['detected_at'] = dt.datetime.now().isoformat(timespec='microseconds')
    output = export_analysis(analysis, args.out, make_export_stem(args.image), options=options)
    print(json.dumps({'status': analysis.status, 'roi_count': len(analysis.rois), 'out': output['directory'],
        'detect_seconds': analysis.metadata['detect_seconds'], 'warnings': analysis.metadata['warnings']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

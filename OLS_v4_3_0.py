#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4.3.0 image markers and ordinary least-squares affine calibration.

Marker detection uses original image pixels. Each worktable selects its own
coordinate origin; calibration, correspondence tables and exports share it.
Machine units are those entered in X,Y. No row matching is inferred from a grid.
Run python OLS_v4_3_0.py for a standalone calibration editor.
"""
from __future__ import annotations

import configparser
import copy
import csv
import io
import math
import hashlib
import os
from pathlib import Path
import tempfile
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

VERSION = "4.3.0"
ORIGINS = {'图片中心': 'center', '图片左上角': 'top_left', '亮圆环拟合圆心': 'ring_center'}


def image_frame(shape, origin='top_left', ring_center=None):
    if origin not in (*ORIGINS.values(),'bottom_left'):
        raise ValueError('未知图像原点')
    if origin == 'top_left':
        return np.array([0., 0.]), 1.
    if origin=='ring_center':
        center=np.asarray(ring_center,dtype=float)
        if center.shape!=(2,) or not np.isfinite(center).all():raise ValueError('请先载入图片并拟合亮圆环圆心')
        return center, -1.
    if shape is None or len(shape) != 2 or min(shape) < 16:
        raise ValueError('请先载入标定图片或填写原图宽、高，再换算坐标原点')
    h, w = shape
    if origin == 'bottom_left':
        return np.array([0., h - 1.]), -1.
    return np.array([(w - 1.) / 2, (h - 1.) / 2]), -1.


def image_to_frame(points, shape, origin='top_left', ring_center=None):
    p = _points(points, '原图坐标')
    offset, y_sign = image_frame(shape, origin, ring_center)
    return (p - offset) * np.array([1., y_sign])


def frame_to_image(points, shape, origin='top_left', ring_center=None):
    p = _points(points, '图像坐标')
    offset, y_sign = image_frame(shape, origin, ring_center)
    return p * np.array([1., y_sign]) + offset


def change_table_origin(table, origin, shape=None, ring_center=None):
    """Atomically re-express existing points; a change invalidates the fit."""
    if origin not in ORIGINS.values():
        raise ValueError('未知图像原点')
    if origin=='ring_center':image_frame(shape,'ring_center',ring_center)
    previous = table.get('origin', 'top_left')
    if previous == origin and (origin!='ring_center' or (table.get('ring_center') is not None and ring_center is not None
                               and np.allclose(table['ring_center'],ring_center,atol=1e-6,rtol=0))):
        return
    rows = [dict(row) for row in table['rows']]
    shape = shape or table.get('image_shape')
    if any(str(row.get(k, '')).strip() for row in rows for k in ('u', 'v')):
        old_offset, old_y = image_frame(shape, previous, table.get('ring_center'))
        new_offset, new_y = image_frame(shape, origin, ring_center)
        for row in rows:
            for j, key in enumerate(('u', 'v')):
                value = str(row.get(key, '')).strip()
                if not value:
                    continue
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError('请先修正无效坐标，再切换原点')
                raw = number * (old_y if j == 1 else 1) + old_offset[j]
                row[key] = f'{(raw - new_offset[j]) * (new_y if j == 1 else 1):.8f}'
    table.update(rows=rows, origin=origin, fit=None)
    if origin=='ring_center':
        offset,_=image_frame(shape,origin,ring_center);table['ring_center']=offset.tolist()
    if shape is not None:
        table['image_shape'] = tuple(map(int, shape))


def _fit_frame_center(gray):
    from roi_polar_detector_v4_3_0 import fit_bright_ring
    result=fit_bright_ring(gray)
    if result.get('status')!='ok':raise ValueError('亮圆环拟合失败：'+result.get('reason','未找到可信圆心'))
    return result['center_image_xy_px']


def _points(value, title):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError(title + "必须为有限数值的 N×2 数组")
    return a


def transform_points(points, coefficients):
    p = _points(points, "图像坐标")
    c = np.asarray(coefficients, dtype=np.float64)
    if c.shape != (2, 3) or not np.isfinite(c).all():
        raise ValueError("回归系数必须为有限数值的 2×3 矩阵")
    if np.linalg.matrix_rank(c[:, :2]) < 2:
        raise ValueError("回归方程退化，无法完成二维坐标转换")
    return p @ c[:, :2].T + c[:, 2]


def inverse_transform_points(machine_xy, coefficients):
    xy=_points(machine_xy,'机台坐标')
    c=np.asarray(coefficients,dtype=float)
    if c.shape!=(2,3) or not np.isfinite(c).all() or np.linalg.matrix_rank(c[:,:2])<2:
        raise ValueError('回归方程无效或退化，不能反算')
    if np.linalg.cond(c[:,:2])>1e10:raise ValueError('回归方程接近退化，反算不稳定')
    return np.linalg.solve(c[:,:2],(xy-c[:,2]).T).T


def verification_error(image_uv, actual_xy, coefficients, tolerance=None):
    uv=_points(image_uv,'图像坐标');actual=_points(actual_xy,'实测机台坐标')
    if uv.shape!=actual.shape:raise ValueError('复检两侧的点数必须一致')
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance<=0):
        raise ValueError('允许偏差应大于 0，留空时只显示数值')
    predicted=transform_points(uv,coefficients);delta=predicted-actual
    radial=np.linalg.norm(delta,axis=1)
    return {'predicted':predicted,'delta':delta,'errors':radial,
            'passed':None if tolerance is None else radial<=tolerance}


def compare_coordinate_pair(image_uv, machine_xy, coefficients, space='image', tolerance=None):
    """Both inputs are independently measured; only the compared side changes."""
    uv=_points(image_uv,'图像实测坐标');xy=_points(machine_xy,'机台实测坐标')
    if uv.shape!=xy.shape:raise ValueError('图像与机台实测点数必须一致')
    if space not in ('image','machine'):raise ValueError('复检模式无效')
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance<=0):
        raise ValueError('允许偏差应大于 0，留空时只显示数值')
    predicted_uv=inverse_transform_points(xy,coefficients)
    predicted_xy=transform_points(uv,coefficients)
    predicted=predicted_uv if space=='image' else predicted_xy
    measured=uv if space=='image' else xy
    delta=predicted-measured;errors=np.linalg.norm(delta,axis=1)
    return {'predicted_uv':predicted_uv,'predicted_xy':predicted_xy,'measured':measured,
            'predicted':predicted,'delta':delta,'errors':errors,'space':space,
            'unit':'px' if space=='image' else '机台单位',
            'passed':None if tolerance is None else errors<=tolerance}


@dataclass
class AffineFit:
    coefficients: tuple
    predicted: np.ndarray
    residuals: np.ndarray
    radial_errors: np.ndarray
    rms: float | None
    max_error: float | None
    axis_rms: tuple
    loo_rms: float | None
    condition: float
    count: int

    def equations(self, digits=10):
        lines = []
        for name, row in zip(("X", "Y"), self.coefficients):
            a, b, c = row
            lines.append(f"{name} = {a:.{digits}g}·u {b:+.{digits}g}·v {c:+.{digits}g}")
        return "\n".join(lines)


def evaluate_equation(image_uv, machine_xy, coefficients):
    """Validate a manual affine equation and measure it against available rows."""
    uv = np.asarray(image_uv, dtype=float).reshape(-1, 2)
    xy = np.asarray(machine_xy, dtype=float).reshape(-1, 2)
    if uv.shape != xy.shape or not np.isfinite(xy).all():
        raise ValueError('对应坐标数量不一致或包含无效数值')
    predicted = transform_points(uv, coefficients)
    residuals = predicted - xy
    radial = np.linalg.norm(residuals, axis=1)
    c = tuple(tuple(float(v) for v in row) for row in np.asarray(coefficients))
    return AffineFit(c, predicted, residuals, radial,
        float(np.sqrt(np.mean(radial ** 2))) if len(uv) else None,
        float(radial.max()) if len(uv) else None,
        tuple(np.sqrt(np.mean(residuals ** 2, axis=0))) if len(uv) else (None, None),
        None, float(np.linalg.cond(np.asarray(c)[:, :2])), len(uv))


def fit_affine(image_uv, machine_xy):
    """Unweighted OLS using every row; numerical normalization, no rejection."""
    uv, xy = _points(image_uv, "图像坐标"), _points(machine_xy, "机台坐标")
    if len(uv) != len(xy) or len(uv) < 3:
        raise ValueError("至少需要 3 组完整的对应坐标，且两侧数量相同")
    if len(np.unique(uv, axis=0)) != len(uv):
        raise ValueError("图像坐标存在重复点，请检查点号对应关系")
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("机台坐标存在重复点，请检查点号对应关系")
    mean, scale = uv.mean(axis=0), uv.std(axis=0)
    if np.any(scale < 1e-9):
        raise ValueError("图像点不能共线；请分布在二维区域内")
    z = (uv - mean) / scale
    design = np.c_[z, np.ones(len(z))]
    condition = float(np.linalg.cond(design))
    if np.linalg.matrix_rank(design) < 3 or condition > 1e5:
        raise ValueError("图像点共线或近似共线，无法稳定拟合")
    centered = xy - xy.mean(axis=0)
    xy_scale = centered.std(axis=0)
    if np.any(xy_scale < 1e-12) or np.linalg.matrix_rank(centered / xy_scale) < 2:
        raise ValueError("机台点不能共线或重合")
    coef_z, _, _, _ = np.linalg.lstsq(design, xy, rcond=None)
    linear = coef_z[:2] / scale[:, None]
    offset = coef_z[2] - mean @ linear
    coefficients = tuple(tuple(map(float, r)) for r in np.r_[linear, offset[None]].T)
    predicted = transform_points(uv, coefficients)
    residuals = predicted - xy
    radial = np.linalg.norm(residuals, axis=1)
    # Analytic leave-one-out residuals; unavailable if a removal loses rank.
    leverage = np.sum(design * np.linalg.solve(design.T @ design, design.T).T, axis=1)
    loo = None
    if len(uv) > 3 and np.all(1 - leverage > 1e-9):
        loo = float(np.sqrt(np.mean(np.sum((residuals / (1 - leverage[:, None])) ** 2, axis=1))))
    return AffineFit(coefficients, predicted, residuals, radial,
                     float(np.sqrt(np.mean(radial ** 2))), float(radial.max()),
                     tuple(np.sqrt(np.mean(residuals ** 2, axis=0)).tolist()),
                     loo, condition, len(uv))


@dataclass
class Marker:
    u: float
    v: float
    contour: np.ndarray
    area: float
    mode: str
    score: float = 0.
    details: dict = field(default_factory=dict)

    @property
    def center(self):
        return (self.u, self.v)


def _gray(image):
    g = np.asarray(image)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.ndim != 2 or g.dtype != np.uint8 or min(g.shape) < 16:
        raise ValueError("请提供至少 16×16 的 8 位灰度图像")
    return g


def _smooth_contour(contour):
    p = np.asarray(contour, float).reshape(-1, 2)
    if len(p) > 10:
        p = gaussian_filter1d(p, .9, axis=0, mode="wrap")
    return p.astype(np.float32)


def marker_from_polygon(points, mode="centroid"):
    p = _points(points, "手绘边界")
    if len(p) < 3:
        raise ValueError("请至少圈出三个边界点")
    from shapely.geometry import Polygon
    poly = Polygon(p)
    if not poly.is_valid or poly.area < 2:
        raise ValueError("边界不可自交，且必须包围有效面积")
    return Marker(float(poly.centroid.x), float(poly.centroid.y), p.astype(np.float32),
                  float(poly.area), mode, details={"manual_boundary": True})


def _extract_puncture_once(gray, seed, radius=None):
    """Enclose BOTH bright and dark local deformation, then area centroid.

    Segmentation is locally noise/contrast adaptive. A seed only selects the
    region, never becomes the reported center. Internal gaps are filled by the
    external contour, and fine contour noise is smoothed at subpixel scale.
    """
    g = _gray(gray)
    radius = int(radius or max(20, min(g.shape) * .012))
    if not 16 <= radius <= 1000:
        raise ValueError("搜索半径应为 16～1000 原图像素")
    u, v = map(float, seed)
    if not (math.isfinite(u) and math.isfinite(v) and 0 <= u < g.shape[1] and 0 <= v < g.shape[0]):
        raise ValueError("点选位置超出图片")
    x0, y0 = max(0, round(u) - radius), max(0, round(v) - radius)
    x1, y1 = min(g.shape[1], round(u) + radius + 1), min(g.shape[0], round(v) + radius + 1)
    cut = g[y0:y1, x0:x1].astype(np.float32)
    raw_noise=float(np.median(np.abs(cut-cv2.GaussianBlur(cut,(0,0),.8)))*1.4826)
    sm = cv2.GaussianBlur(cut, (0, 0), 1.1 if raw_noise>6 else .8)
    residual = sm - cv2.GaussianBlur(sm, (0, 0), max(5., radius * .3))
    noise = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    threshold = max(1.5, 4 * noise, float(np.percentile(np.abs(residual), 99.5)) * .15)
    mask = (np.abs(residual) > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    seed_local = (u - x0, v - y0)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(4,12*(min(g.shape)/5120)**2) or area > cut.size * .3:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + w >= cut.shape[1] - 1 or y + h >= cut.shape[0] - 1:
            continue
        signed_distance = cv2.pointPolygonTest(contour, seed_local, True)
        if signed_distance < -radius * .45:
            continue
        m = cv2.moments(contour)
        cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
        candidates.append((max(0., -signed_distance) + .03 * math.hypot(cx - seed_local[0], cy - seed_local[1]), contour))
    if not candidates:
        raise ValueError("附近未找到完整的戳破区域，请缩放后点选或手绘边界")
    contour = min(candidates, key=lambda item: item[0])[1]
    p = _smooth_contour(contour) + np.array([x0, y0], np.float32)
    m = cv2.moments(p)
    area = float(m['m00'])
    inside = np.zeros(cut.shape, np.uint8)
    cv2.drawContours(inside, [contour], -1, 255, -1)
    values = residual[inside > 0]
    bright, dark = float(np.percentile(values, 95)), float(-np.percentile(values, 5))
    return Marker(float(m['m10'] / area), float(m['m01'] / area), p, area,
                  "centroid", min(bright, dark),
                  {"threshold": threshold, "noise": noise, "bright": bright, "dark": dark,
                   "search_radius": radius, "centroid_basis": "enclosed_deformation_area"})


def extract_puncture(gray, seed, radius=None):
    # Recenter the local background window on the extracted region rather than
    # on the operator's click. Nearby clicks then converge to the same mask.
    mark = _extract_puncture_once(gray, seed, radius)
    previous = tuple(map(round, seed))
    for _ in range(4):
        center = tuple(map(round, mark.center))
        if center == previous:
            break
        previous = center
        mark = _extract_puncture_once(gray, center, radius)
    return mark


def detect_punctures(gray):
    """Unknown-location candidates from locally normalized, bipolar scales.

    No grid, filename or target-count prior is used. Native-resolution masks
    determine each area centroid. Compact look-alike texture can remain a
    candidate; score is a quality heuristic rather than a physical-hole label.
    """
    g=_gray(gray);factor=min(1.,2048/max(g.shape))
    size=(round(g.shape[1]*factor),round(g.shape[0]*factor))
    small=cv2.resize(g,size,interpolation=cv2.INTER_AREA)
    sx,sy=g.shape[1]/size[0],g.shape[0]/size[1]
    _,film=cv2.threshold(small,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    film=cv2.morphologyEx(film,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(film)
    if n>1:film=(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.uint8)*255
    contours,_=cv2.findContours(film,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return []
    film=np.zeros_like(film);cv2.drawContours(film,[max(contours,key=cv2.contourArea)],-1,255,-1)
    film=cv2.erode(film,np.ones((13,13),np.uint8))
    if not np.any(film):return []
    sm=cv2.GaussianBlur(small.astype(np.float32),(0,0),.7)
    seeds=[]
    for scale in (5.5,12.):
        residual=sm-cv2.GaussianBlur(sm,(0,0),scale)
        sample=residual[::4,::4][film[::4,::4]>0]
        if not len(sample):sample=residual[film>0]
        noise=max(.35,float(np.median(np.abs(sample)))*1.4826)
        local=np.maximum(noise,cv2.GaussianBlur(np.abs(residual),(0,0),7))
        threshold=np.maximum(1.2,3.*local)
        positive=cv2.dilate(residual,np.ones((15,15),np.uint8))
        negative=cv2.dilate(-residual,np.ones((15,15),np.uint8))
        paired=(positive>np.maximum(2.,local*3.5))&(negative>np.maximum(2.,local*3.5))&(film>0)
        for sign in (1.,-1.):
            mask=((sign*residual>threshold)&paired).astype(np.uint8)
            n,lab,stats,centers=cv2.connectedComponentsWithStats(mask)
            for rect,center in zip(stats[1:],centers[1:]):
                x,y,w,h,area=rect
                if not (3<=area<=600 and min(w,h)>=2 and max(w,h)<=45):continue
                u,v=(center[0]+.5)*sx-.5,(center[1]+.5)*sy-.5
                cx,cy=int(round(center[0])),int(round(center[1]))
                strength=min(positive[cy,cx],negative[cy,cx])/max(local[cy,cx],.35)
                seeds.append((float(strength),u,v))
    candidates=[];attempted={};seed_gap=max(3.,min(g.shape)*.0015)
    for strength,u,v in sorted(seeds,reverse=True):
        if any(math.hypot(u-m.u,v-m.v)<.65*math.sqrt(m.area/math.pi)*2 for m in candidates):continue
        cell=(int(u/seed_gap),int(v/seed_gap))
        nearby=[p for x in range(cell[0]-1,cell[0]+2) for y in range(cell[1]-1,cell[1]+2) for p in attempted.get((x,y),())]
        if any(math.hypot(u-p[0],v-p[1])<seed_gap for p in nearby):continue
        attempted.setdefault(cell,[]).append((u,v))
        try:mark=extract_puncture(g,(u,v))
        except ValueError:continue
        diameter=math.sqrt(mark.area/math.pi)*2
        if not min(g.shape)/1000<=diameter<=min(g.shape)/45:continue
        bright,dark=mark.details['bright'],mark.details['dark']
        noise=max(mark.details['noise'],.35)
        if min(bright,dark)<max(2.,4*noise) or min(bright,dark)/max(bright,dark)<.12:continue
        contour=mark.contour.astype(np.float32)
        (_, _),(w,h),_=cv2.minAreaRect(contour)
        perimeter=cv2.arcLength(contour,True)
        compactness=4*math.pi*mark.area/max(perimeter**2,1)
        aspect=max(w,h)/max(min(w,h),1)
        if aspect>5 or compactness<.12:continue
        if any(np.linalg.norm(np.subtract(mark.center,m.center))<diameter*.8 for m in candidates):continue
        snr=min(bright,dark)/noise
        quality=min(1.,snr/12)*min(1.,compactness/.55)*min(1.,2./max(aspect,1))
        mark.details.update(contrast_snr=snr,compactness=compactness,aspect_ratio=aspect,
                            quality_score=quality,candidate_quality='较高' if quality>=.65 else '疑似')
        candidates.append(mark)
    return sorted(candidates,key=lambda m:(m.v,m.u))


def _ring_candidates(gray, max_area_fraction=.02):
    # Full-resolution threshold contours are refined on the inner aperture only.
    g = _gray(gray)
    smooth = cv2.GaussianBlur(g, (0, 0), .75)
    _, dark = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    candidates = []
    for i, contour in enumerate(contours):
        parent = hierarchy[0, i, 3]
        if parent < 0 or len(contour) < 12:
            continue
        area = cv2.contourArea(contour)
        if not 20 < area < g.size * max_area_fraction:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 5 or min(w, h) / max(w, h) < .65:
            continue
        outer_area = cv2.contourArea(contours[parent])
        ratio = area / max(outer_area, 1)
        circularity = 4 * math.pi * area / max(cv2.arcLength(contour, True) ** 2, 1)
        if not .12 < ratio < .85 or circularity < .65:
            continue
        ellipse = cv2.fitEllipse(contour)
        (u, v), (a, b), angle = ellipse
        if min(a, b) / max(a, b) < .65:
            continue
        p = contour[:, 0].astype(np.float32)
        candidates.append(Marker(float(u), float(v), p, float(area), "ring", circularity,
                                 {"ellipse_axes": [a, b], "ellipse_angle": angle,
                                  "center_basis": "inner_aperture_ellipse"}))
    return candidates


def detect_rings(gray):
    return sorted(_ring_candidates(gray), key=lambda m: (m.v, m.u))


def extract_ring(gray, seed, radius=None):
    g = _gray(gray)
    radius = int(radius or max(60, min(g.shape) * .045))
    u, v = map(float, seed)
    if not (math.isfinite(u) and math.isfinite(v) and 0 <= u < g.shape[1] and 0 <= v < g.shape[0]):
        raise ValueError("点选位置超出图片")
    x0, y0 = max(0, round(u) - radius), max(0, round(v) - radius)
    x1, y1 = min(g.shape[1], round(u) + radius + 1), min(g.shape[0], round(v) + radius + 1)
    marks = _ring_candidates(g[y0:y1, x0:x1], .4)
    if not marks:
        raise ValueError("附近未找到黑色圆环的内孔，请点在内孔中心附近或扩大搜索半径")
    result = min(marks, key=lambda m: math.hypot(m.u + x0 - u, m.v + y0 - v))
    if math.hypot(result.u + x0 - u, result.v + y0 - v) > radius * .6:
        raise ValueError("最近的内孔离点选位置过远，请重新点选")
    result.u += x0
    result.v += y0
    result.contour += np.array([x0, y0], np.float32)
    return result


def detect_markers(gray, mode="centroid"):
    if mode not in ("centroid", "ring"):
        raise ValueError("坐标提取模式应为 centroid 或 ring")
    return detect_punctures(gray) if mode == "centroid" else detect_rings(gray)


def extract_marker(gray, seed, mode="centroid", radius=None):
    if mode not in ("centroid", "ring"):
        raise ValueError("坐标提取模式应为 centroid 或 ring")
    return extract_puncture(gray, seed, radius) if mode == "centroid" else extract_ring(gray, seed, radius)


def empty_row(identifier):
    return {"id": str(identifier), "u": "", "v": "", "X": "", "Y": ""}


def complete_rows(rows):
    result = []
    identifiers = set()
    for i, row in enumerate(rows, 1):
        identifier = str(row.get("id", "")).strip()
        values = [str(row.get(k, "")).strip() for k in ("u", "v", "X", "Y")]
        if not any(values):
            continue
        if not identifier or identifier in identifiers:
            raise ValueError(f"第{i}行点号为空或重复")
        if not all(values):
            raise ValueError(f"点 {identifier} 尚未填完整 u、v、X、Y")
        try:
            numbers = list(map(float, values))
        except ValueError as exc:
            raise ValueError(f"点 {identifier} 包含非数值坐标") from exc
        if not np.isfinite(numbers).all():
            raise ValueError(f"点 {identifier} 包含无效坐标")
        identifiers.add(identifier)
        result.append((i - 1, identifier, *numbers))
    return result


def default_state_path():
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    return base / "RoiPolar" / "v4_3_0" / "worktables.ini"


class WorktableStore:
    """Atomic INI state; result folders still contain only images/coordinates."""
    def __init__(self, path=None, seed_path=None):
        self.path = Path(path) if path else default_state_path()
        self.tables = []
        self.active_id = ""
        self.load_error = ""
        legacy=next((self.path.parent.parent/v/'worktables.ini' for v in ('v4_2_9','v4_2_8','v4_2_7','v4_2_6','v4_2_5','v4_2_4','v4_2_3','v4_2_2','v4_2_1')
                     if (self.path.parent.parent/v/'worktables.ini').exists()),None)
        source = self.path if self.path.exists() else (legacy if path is None and legacy else Path(seed_path) if seed_path else None)
        if source and source.exists():
            try:
                self._load(source)
            except Exception as exc:
                self.load_error = "读取工作台失败：" + str(exc)
        if not self.tables:
            self.add("工作台1")
        if self.active_id not in {t['id'] for t in self.tables}:
            self.active_id = self.tables[0]['id']

    def add(self, name):
        name = str(name).strip()
        if not name or any(t['name'] == name for t in self.tables):
            raise ValueError("工作台名称不能为空或重复")
        table = {"id": uuid.uuid4().hex, "name": name, "rows": [empty_row(i) for i in range(1, 10)],
                 "image_shape": None, "fit": None, "origin": 'center', 'ring_center':None,'equation_source': 'ols'}
        self.tables.append(table)
        self.active_id = table['id']
        return table

    @property
    def active(self):
        return next(t for t in self.tables if t['id'] == self.active_id)

    def _load(self, path):
        ini = configparser.ConfigParser(interpolation=None)
        ini.read(path, encoding="utf-8-sig")
        tables = []
        for section in ini.sections():
            if not section.startswith("table:"):
                continue
            d = ini[section]
            rows = []
            for values in csv.reader(io.StringIO(d.get('rows', ''))):
                if not values:
                    continue
                if len(values) != 5:
                    raise ValueError("工作台坐标行需包含 id,u,v,X,Y 五列")
                rows.append(dict(zip(("id", "u", "v", "X", "Y"), values)))
            shape = tuple(map(int, d.get('image_shape', '').split(','))) if d.get('image_shape') else None
            if shape and (len(shape) != 2 or min(shape) < 1):
                raise ValueError("标定图像尺寸无效")
            origin = d.get('origin', 'top_left')
            if origin not in (*ORIGINS.values(),'bottom_left'):
                raise ValueError('工作台原点设置无效')
            ring_center=[float(v) for v in d.get('ring_center','').split(',')] if d.get('ring_center') else None
            if origin=='ring_center':image_frame(shape,origin,ring_center)
            source = d.get('equation_source', 'ols')
            if source not in ('ols', 'manual'): raise ValueError('方程来源设置无效')
            table = {'id': section[6:], 'name': d['name'], 'rows': rows, 'image_shape': shape,
                     'fit': None, 'origin': origin, 'ring_center':ring_center,'equation_source': source}
            if d.getboolean('fitted', fallback=False):
                valid = complete_rows(rows)
                if source == 'manual':
                    values = [float(v) for v in next(csv.reader([d.get('coefficients', '')]))]
                    if len(values) != 6: raise ValueError('手动方程需要六个系数')
                    table['fit'] = evaluate_equation([r[2:4] for r in valid], [r[4:6] for r in valid], np.asarray(values).reshape(2, 3))
                else:
                    table['fit'] = fit_affine([r[2:4] for r in valid], [r[4:6] for r in valid])
            if origin=='bottom_left':
                # Migrate the removed choice without changing machine positions.
                if shape is None and not any(row.get('u') or row.get('v') for row in rows):
                    table['origin']='center';tables.append(table);continue
                fitted=table['fit'];old_offset,old_sign=image_frame(shape,'bottom_left')
                new_offset,new_sign=image_frame(shape,'center')
                change_table_origin(table,'center',shape)
                if fitted:
                    c=np.asarray(fitted.coefficients);shift=(new_offset-old_offset)*[1,old_sign]
                    linear=c[:,:2]@np.diag([1,old_sign/new_sign]);offset=c[:,2]+c[:,:2]@shift
                    values=complete_rows(table['rows'])
                    table['fit']=evaluate_equation([r[2:4] for r in values],[r[4:6] for r in values],np.c_[linear,offset])
            tables.append(table)
        if len({t['name'] for t in tables}) != len(tables):
            raise ValueError("工作台名称重复")
        self.tables = tables
        self.active_id = ini.get('calibration', 'active_id', fallback='')

    def save(self):
        if self.load_error and self.path.exists():
            raise ValueError(self.load_error + "；原文件已保留，请修复配置或使用其他状态文件")
        ini = configparser.ConfigParser(interpolation=None)
        ini['calibration'] = {'version': VERSION, 'active_id': self.active_id,
                              'image_convention': 'per_table_origin_original_pixels'}
        for table in self.tables:
            stream = io.StringIO()
            csv.writer(stream, lineterminator='\n').writerows(
                [row.get(k, '') for k in ('id', 'u', 'v', 'X', 'Y')] for row in table['rows'])
            ini['table:' + table['id']] = {'name': table['name'], 'rows': stream.getvalue(),
                'image_shape': ','.join(map(str, table['image_shape'])) if table.get('image_shape') else '',
                'origin': table.get('origin', 'top_left'), 'fitted': str(table.get('fit') is not None),
                'ring_center': ','.join(format(float(v),'.17g') for v in table['ring_center']) if table.get('ring_center') is not None else '',
                'equation_source': table.get('equation_source', 'ols'),
                'coefficients': ','.join(format(v, '.17g') for row in table['fit'].coefficients for v in row) if table.get('fit') else ''}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='worktables_', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                ini.write(f)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def _column_key(value):
    import re, unicodedata
    text = unicodedata.normalize('NFKC', str(value or '')).strip().casefold()
    text = re.sub(r'\([^)]*\)|\[[^]]*\]', '', text)
    text = re.sub(r'[\s_\-./:：]+', '', text)
    text = re.sub(r'(?:pixels?|px|mm)$', '', text)
    aliases = {
        'id': ('id','pointid','point','no','序号','点号','编号','点位','位置'),
        'u': ('u','图像u','图像坐标u','像素u','像素坐标u','图像x','图像坐标x','像素x','像素坐标x','imageu','imagex','pixelx'),
        'v': ('v','图像v','图像坐标v','像素v','像素坐标v','图像y','图像坐标y','像素y','像素坐标y','imagev','imagey','pixely'),
        'X': ('x','机台x','机台坐标x','机器x','机械x','工作台x','平台x','machine x','machinex','stagex','x坐标'),
        'Y': ('y','机台y','机台坐标y','机器y','机械y','工作台y','平台y','machiney','stagey','y坐标'),
        'origin': ('origin','原点','图像原点','坐标原点'),
        'image_width': ('imagewidth','图宽','图片宽度','图像宽度'),
        'image_height': ('imageheight','图高','图片高度','图像高度'),
        'origin_x': ('originx','圆心x','原点x'),
        'origin_y': ('originy','圆心y','原点y'),
    }
    return next((key for key, names in aliases.items() if text in names), None)


def read_correspondences(path, with_metadata=False):
    """Import identified columns or documented 2/3/4/5-column positional data."""
    path = Path(path)
    if path.suffix.lower() == '.xlsx':
        try: import openpyxl
        except ImportError as exc: raise ValueError('读取 XLSX 需要 openpyxl，也可另存为 CSV') from exc
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try: records = list(wb.active.values)
        finally: wb.close()
    else:
        try: text = path.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError: text = path.read_text(encoding='gb18030')
        if not text.strip(): raise ValueError('文件为空')
        sample = '\n'.join(text.splitlines()[:20])
        try: delimiter = csv.Sniffer().sniff(sample, delimiters=',\t;').delimiter
        except csv.Error: delimiter = '\t' if '\t' in sample else ','
        records = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    records = [list(row) for row in records if any(v is not None and str(v).strip() for v in row)]
    if not records: raise ValueError('文件为空')
    header = None; data = None
    for index, record in enumerate(records[:30]):
        keys = [_column_key(v) for v in record]
        known = [k for k in keys if k]
        if 'X' in known and 'Y' in known:
            if len(known) != len(set(known)):
                raise ValueError('表中含重复坐标列或多组 X、Y，请将每个工作台分别整理成一张表后导入。')
            header, data = keys, records[index+1:]; break
    if header is None:
        trimmed = []
        for record in records:
            row = list(record)
            while row and (row[-1] is None or str(row[-1]).strip() == ''): row.pop()
            trimmed.append(row)
        widths = {len(row) for row in trimmed}
        if len(widths) != 1 or next(iter(widths)) not in (2,3,4,5):
            raise ValueError('无法识别列。推荐表头 id,u,v,X,Y；无表头支持 XY、idXY、uvXY、iduvXY。')
        n = next(iter(widths)); header = {2:['X','Y'],3:['id','X','Y'],4:['u','v','X','Y'],5:['id','u','v','X','Y']}[n]
        # Numeric coordinates distinguish positional data from unknown headings.
        for row in trimmed:
            for j, key in enumerate(header):
                if key == 'id' or row[j] is None or str(row[j]).strip() == '': continue
                try: value = float(row[j])
                except (ValueError, TypeError) as exc: raise ValueError('无法识别表头，请使用 id,u,v,X,Y 或对应的中英文列名。') from exc
                if not math.isfinite(value): raise ValueError('坐标包含无效数值')
        data = trimmed
    if not data: raise ValueError('没有数据行')
    rows = []
    for i, record in enumerate(data, 1):
        row = empty_row(i)
        for key in row:
            if key in header:
                j = header.index(key); row[key] = str(record[j]).strip() if j < len(record) and record[j] is not None else ''
        rows.append(row)
    if not with_metadata: return rows
    metadata = {}
    for key in ('origin','image_width','image_height','origin_x','origin_y'):
        if key not in header: continue
        j = header.index(key)
        values = {str(row[j]).strip() for row in data if j < len(row) and row[j] is not None and str(row[j]).strip()}
        if len(values) > 1: raise ValueError('对应表中的原点或图片尺寸不一致')
        if values: metadata[key] = values.pop()
    origin = metadata.get('origin')
    origin = {'图片左上':'top_left','图片左下':'bottom_left',**ORIGINS}.get(origin, origin)
    if origin and origin not in (*ORIGINS.values(),'bottom_left'): raise ValueError('对应表中的原点无效')
    shape = None
    if metadata.get('image_width') or metadata.get('image_height'):
        numbers = [float(metadata.get('image_height',0)), float(metadata.get('image_width',0))]
        if any(not math.isfinite(v) or v != int(v) or v < 16 for v in numbers): raise ValueError('对应表图片尺寸无效')
        shape = tuple(map(int,numbers))
    center=None
    if metadata.get('origin_x') or metadata.get('origin_y'):
        center=[float(metadata.get('origin_x','nan')),float(metadata.get('origin_y','nan'))]
        image_frame(shape,'ring_center',center)
    if origin=='ring_center' and center is None:raise ValueError('以圆心为原点的对应表需包含 origin_x、origin_y 两列')
    if origin=='bottom_left':
        table={'rows':rows,'origin':origin,'image_shape':shape}
        change_table_origin(table,'center',shape);rows=table['rows'];origin='center'
    return {'rows': rows, 'origin': origin, 'image_shape': shape,'ring_center':center}


def write_correspondences(path, rows, origin=None, image_shape=None, ring_center=None):
    path = Path(path)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter='\t' if path.suffix.lower() == '.txt' else ',', lineterminator='\n')
        extra = [origin, image_shape[1] if image_shape else '', image_shape[0] if image_shape else ''] if origin else []
        extra_headers=['origin','image_width','image_height'] if origin else []
        if origin=='ring_center':
            center,_=image_frame(image_shape,origin,ring_center)
            extra+=center.tolist();extra_headers+=['origin_x','origin_y']
        writer.writerow(['id', 'u', 'v', 'X', 'Y'] + extra_headers)
        writer.writerows([row.get(k, '') for k in ('id', 'u', 'v', 'X', 'Y')] + extra for row in rows)


# UI imports remain below the computation functions so importing OLS never
# creates a window, starts a scan, or accesses Tk variables from a worker.
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter import font as tkfont
import queue
import threading
from PIL import Image, ImageTk


class TickCheckbutton(tk.Frame):
    """Explicit vector tick; selected controls never become a theme-dependent X."""
    def __init__(self,parent,text,variable,command=None):
        super().__init__(parent,bg="#181818",highlightthickness=0,takefocus=1)
        self.variable=variable;self.command=command;self.enabled=True
        self.indicator=tk.Canvas(self,width=18,height=18,bg="#181818",highlightthickness=0)
        self.indicator.pack(side="left",padx=(0,5))
        self.label=tk.Label(self,text=text,bg="#181818",fg="#EDEDED")
        self.label.pack(side="left")
        for widget in (self,self.indicator,self.label):widget.bind("<Button-1>",self._toggle)
        self.bind("<space>",self._toggle);self.bind("<Return>",self._toggle)
        variable.trace_add("write",lambda *a:self._draw());self._draw()

    def _draw(self):
        c=self.indicator;c.delete("all")
        c.create_rectangle(2,2,16,16,outline="#76B4FF" if self.enabled else "#666666",fill=("#1B3D27" if self.enabled else "#303030") if self.variable.get() else "#242424")
        self.label.configure(fg="#EDEDED" if self.enabled else "#777777")
        if self.variable.get():c.create_line(4,9,8,13,14,5,fill="#C6FFD0" if self.enabled else "#888888",width=2)

    def set_enabled(self,enabled):
        self.enabled=bool(enabled);self.configure(takefocus=int(self.enabled));self._draw()

    def _toggle(self,event=None):
        if self.enabled:
            self.variable.set(not self.variable.get());self.focus_set()
            if self.command:self.command()
        return "break"


class FoldSection(ttk.Frame):
    def __init__(self,parent,title):
        super().__init__(parent)
        self.title=title;self.open=False
        self.pack(fill='x',pady=(0,10))
        self.button=ttk.Button(self,text='▶ '+title,command=self.toggle)
        self.button.pack(fill='x')
        self.body=ttk.Frame(self,padding=(8,10))

    def toggle(self):
        self.open=not self.open
        self.button.configure(text=('▼ ' if self.open else '▶ ')+self.title)
        if self.open:self.body.pack(fill='x')
        else:self.body.pack_forget()


class WorktableNotebook(ttk.Notebook):
    """Fixed-size page buttons with Tk-managed, unclipped content height."""
    def __init__(self,parent,**kwargs):
        self.tab_bar=tk.Frame(parent,bg='#181818',height=38)
        self.tab_bar.pack(fill='x',pady=(0,7));self.tab_bar.grid_propagate(False)
        self.tab_bar.grid_rowconfigure(0,weight=1)
        style=ttk.Style(parent);style.layout('WorkFixed.TNotebook.Tab',[])
        style.configure('WorkFixed.TNotebook',background='#181818',borderwidth=0)
        kwargs.pop('style',None)
        # A fixed height measured while a page is hidden can be only 1px.
        # Natural geometry tracks the requested size as its children initialize.
        kwargs['height']=0
        super().__init__(parent,style='WorkFixed.TNotebook',**kwargs)
        self.page_buttons={}
        self.bind('<<NotebookTabChanged>>',self._paint_buttons,add='+')

    def add(self,child,**kwargs):
        text=kwargs.get('text','');super().add(child,**kwargs)
        column=len(self.page_buttons)
        self.tab_bar.grid_columnconfigure(column,weight=1,uniform='worktable_pages')
        button=tk.Button(self.tab_bar,text=text,width=1,bg='#242424',fg='#EDEDED',
                         activebackground='#333333',activeforeground='#EDEDED',relief='flat',
                         borderwidth=0,highlightthickness=0,padx=8,pady=6,cursor='hand2',
                         command=lambda page=child:self.select(page))
        button.grid(row=0,column=column,sticky='nsew',padx=(0,3) if column==0 else (3,0))
        self.page_buttons[str(child)]=button;self._paint_buttons()

    def _paint_buttons(self,event=None):
        selected=self.select()
        for page,button in self.page_buttons.items():
            color='#295B35' if page==selected else '#242424'
            button.configure(bg=color,activebackground=color if page==selected else '#333333')


class CalibrationPanel(ttk.Frame):
    """One sidebar and one embedded preview; no extraction Toplevel windows."""
    def __init__(self, parent, on_change=None, image_provider=None, state_path=None,
                 preview_parent=None, on_preview_info=None):
        super().__init__(parent)
        self.on_change = on_change; self.image_provider = image_provider
        self.on_preview_info = on_preview_info
        self.store = WorktableStore(state_path)
        self.name = tk.StringVar(); self.width = tk.StringVar(); self.height = tk.StringVar()
        self.origin_var = tk.StringVar(); self.origin_hint = tk.StringVar()
        self.message = tk.StringVar(value=self.store.load_error or '先选图片和检测方式，再自动提取；点击左侧候选点直接写入当前行。')
        self.statistics = tk.StringVar(); self.image_info = tk.StringVar(value='尚未选择标定图片')
        self.target_info = tk.StringVar(); self.mode = tk.StringVar(value='质心（戳破薄膜）')
        self.radius = tk.StringVar(value=''); self.advance = tk.BooleanVar(value=True)
        self.manual = False; self._auto_done = False; self.gray = None; self.source = ''; self.markers = []
        self._refreshing = False; self._cell_column = 1; self._editor = None
        self._size_draft=None; self._equation_drafts={}; self._equation_refreshing=False
        self.coefficient_vars={key:tk.StringVar() for key in 'abcdef'}
        self.equation_text=tk.StringVar(); self.equation_hint=tk.StringVar()
        self._undo = []; self.events = queue.Queue(); self._busy = False
        self._job_id = 0; self._closed = False
        self.preview = MarkerImageView(preview_parent or self, self._pick, self._polygon_done)
        self.preview.on_cancel = self._cancel_manual
        self._build_controls()
        for variable in self.coefficient_vars.values():variable.trace_add('write',self._coefficients_edited)
        self._refresh(select_index=0)
        self.bind('<Destroy>', self._destroyed, add='+')
        self._poll_after = self.after(60, self._poll)

    def _group(self, title):
        group = ttk.LabelFrame(self, text=title, padding=8)
        group.pack(fill='x', pady=(0, 8)); return group

    def _build_controls(self):
        group = self._group('工作台')
        self.selector = ttk.Combobox(group, textvariable=self.name, state='readonly')
        self.selector.pack(fill='x'); self.selector.bind('<<ComboboxSelected>>', self._select_table)
        row = ttk.Frame(group); row.pack(fill='x', pady=(5, 0))
        for label, callback in [('新增工作台', self._add_table), ('改名', self._rename_table), ('删除', self._delete_table)]:
            ttk.Button(row, text=label, command=callback).pack(side='left', expand=True, fill='x', padx=2)

        self._build_equation_controls()
        group = self._group('图像坐标提取')
        row = ttk.Frame(group); row.pack(fill='x')
        self.open_button = ttk.Button(row, text='选择标定图片', command=self.choose_image)
        self.open_button.pack(side='left', fill='x', expand=True, padx=(0, 3))
        self.current_button = ttk.Button(row, text='使用检测主图', command=self.use_current_image)
        self.current_button.pack(side='left', fill='x', expand=True)
        ttk.Label(group, textvariable=self.image_info, wraplength=370, foreground='#AAAAAA').pack(fill='x', pady=(4, 2))
        row = ttk.Frame(group); row.pack(fill='x', pady=3)
        ttk.Label(row, text='检测方式').pack(side='left')
        self.mode_combo = ttk.Combobox(row, textvariable=self.mode, values=('质心（戳破薄膜）', '圆环（黑色圆环内孔）'), state='readonly', width=22)
        self.mode_combo.pack(side='right'); self.mode_combo.bind('<<ComboboxSelected>>', self._mode_changed)
        self.extract_button = ttk.Button(group, text='自动提取图像坐标', command=self.auto_detect)
        self.extract_button.pack(fill='x', pady=(5, 4))
        row = ttk.Frame(group); row.pack(fill='x')
        self.manual_button = ttk.Button(row, text='手动补点（漏检）', command=self.toggle_manual)
        self.manual_button.pack(side='left', fill='x', expand=True, padx=(0, 3))
        self.undo_button = ttk.Button(row, text='撤销上次取点', command=self.undo_assignment)
        self.undo_button.pack(side='left', fill='x', expand=True)
        self.manual_settings = ttk.Frame(group)
        row = ttk.Frame(self.manual_settings); row.pack(fill='x', pady=(5, 2))
        ttk.Label(row, text='局部搜索半径 / px（留空自动）').pack(side='left')
        ttk.Entry(row, textvariable=self.radius, width=6).pack(side='right')
        self.polygon_button = ttk.Button(self.manual_settings, text='手绘形变边界，闭合后直接写入', command=self._arm_polygon)
        self.polygon_button.pack(fill='x', pady=(3, 0))
        ttk.Label(group, textvariable=self.message, wraplength=370, foreground='#76B4FF', justify='left').pack(fill='x', pady=(5, 0))

        group = self._group('坐标对应关系')
        row = ttk.Frame(group); row.pack(fill='x')
        ttk.Label(row, text='图像坐标原点').pack(side='left')
        self.origin_combo = ttk.Combobox(row, textvariable=self.origin_var, values=list(ORIGINS), state='readonly', width=13)
        self.origin_combo.pack(side='right'); self.origin_combo.bind('<<ComboboxSelected>>', self._origin_changed)
        ttk.Label(group, textvariable=self.origin_hint, wraplength=370, foreground='#AAAAAA').pack(fill='x', pady=(3, 4))
        row = ttk.Frame(group); row.pack(fill='x', pady=(0, 4))
        ttk.Label(row, text='标定图宽×高 / px').pack(side='left')
        ttk.Entry(row, textvariable=self.width, width=7).pack(side='left', padx=(6, 2))
        ttk.Label(row, text='×').pack(side='left')
        ttk.Entry(row, textvariable=self.height, width=7).pack(side='left', padx=2)
        for var in (self.width, self.height): var.trace_add('write', self._shape_edited)
        row = ttk.Frame(group); row.pack(fill='x', pady=(0, 3))
        ttk.Label(row, textvariable=self.target_info, foreground='#76B4FF').pack(side='left')
        TickCheckbutton(row, text='写入后下一行', variable=self.advance).pack(side='right')
        box = ttk.Frame(group); box.pack(fill='x')
        self.tree = ttk.Treeview(box, columns=('id', 'u', 'v', 'X', 'Y', 'error'), show='headings', height=7, selectmode='extended')
        for key, title, width in [('id', '点号', 43), ('u', 'u / px', 67), ('v', 'v / px', 67), ('X', '机台 X', 65), ('Y', '机台 Y', 65), ('error', '残差', 63)]:
            self.tree.heading(key, text=title); self.tree.column(key, width=width, minwidth=38, stretch=True, anchor='center')
        scroll = ttk.Scrollbar(box, orient='vertical', command=self.tree.yview)
        self.table_scrollbar=scroll
        self._bind_table_wheel(self.tree);self._bind_table_wheel(scroll)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); self.tree.pack(fill='x', expand=True)
        self.tree.bind('<Double-1>', self._edit_cell)
        self.tree.bind('<Button-1>', self._remember_column, add='+')
        self.tree.bind('<<TreeviewSelect>>', self._target_changed)
        self.tree.bind('<Control-v>', self._paste)
        row = ttk.Frame(group); row.pack(fill='x', pady=(5, 0))
        for label, callback in [('增加行', self.add_row), ('删除行', self.delete_rows), ('粘贴', self._paste)]:
            ttk.Button(row, text=label, command=callback).pack(side='left', expand=True, fill='x', padx=2)
        row = ttk.Frame(group); row.pack(fill='x', pady=(3, 0))
        for label, callback in [('导入对应表', self._import), ('导出对应表', self._export)]:
            ttk.Button(row, text=label, command=callback).pack(side='left', expand=True, fill='x', padx=2)
        row = ttk.Frame(group); row.pack(fill='x', pady=(3, 0))
        for label, callback in [('清空图像坐标', self.clear_image_points), ('清空机台坐标', self.clear_machine_points)]:
            ttk.Button(row, text=label, command=callback).pack(side='left', expand=True, fill='x', padx=2)
        self.fit_button = ttk.Button(self, text='拟合并保存当前工作台', command=self.fit_current)
        self.fit_button.pack(fill='x', pady=(0, 4))
        ttk.Label(self, textvariable=self.statistics, wraplength=375, foreground='#AAAAAA', justify='left').pack(fill='x')

    @property
    def equation_dirty(self):
        return self.store.active_id in self._equation_drafts

    def _build_equation_controls(self):
        self.equation_section=FoldSection(self,'拟合方程')
        group=self.equation_section.body
        ttk.Label(group,textvariable=self.equation_text,wraplength=370,justify='left').pack(fill='x',pady=(0,4))
        for axis,keys in [('X',('a','b','c')),('Y',('d','e','f'))]:
            row=ttk.Frame(group);row.pack(fill='x',pady=2)
            ttk.Label(row,text=axis+' =',width=4).pack(side='left')
            for key,suffix in zip(keys,('·u +','·v +','')):
                entry=ttk.Entry(row,textvariable=self.coefficient_vars[key],width=9)
                entry.pack(side='left',fill='x',expand=True)
                if suffix:ttk.Label(row,text=suffix).pack(side='left',padx=2)
        row=ttk.Frame(group);row.pack(fill='x',pady=(4,0))
        self.apply_equation_button=ttk.Button(row,text='应用并保存方程',command=self.apply_equation)
        self.apply_equation_button.pack(side='left',fill='x',expand=True,padx=(0,3))
        self.revert_equation_button=ttk.Button(row,text='撤销未应用修改',command=self.revert_equation)
        self.revert_equation_button.pack(side='left',fill='x',expand=True)
        ttk.Label(group,textvariable=self.equation_hint,wraplength=370,foreground='#AAAAAA').pack(fill='x',pady=(3,0))

    def _refresh_equation(self):
        draft=self._equation_drafts.get(self.store.active_id)
        fit=self.active_table.get('fit')
        values=draft if draft is not None else ([format(v,'.17g') for row in fit.coefficients for v in row] if fit else ['']*6)
        self._equation_refreshing=True
        try:
            for key,value in zip('abcdef',values):self.coefficient_vars[key].set(value)
        finally:self._equation_refreshing=False
        self._render_equation()

    def _render_equation(self):
        values=[self.coefficient_vars[k].get().strip() for k in 'abcdef']
        try:
            c=np.asarray([float(v) for v in values]).reshape(2,3)
            if not np.isfinite(c).all():raise ValueError('invalid')
            lines=[]
            for name,row in zip(('X','Y'),c):
                a,b,offset=row;lines.append(f'{name} = {a:.12g}·u {b:+.12g}·v {offset:+.12g}')
            self.equation_text.set('\n'.join(lines))
        except ValueError:
            self.equation_text.set('X = a·u + b·v + c\nY = d·u + e·v + f')
        if self.equation_dirty:
            self.equation_hint.set('修改尚未应用；请应用保存或撤销修改后再转换导出。')
        elif self.active_table.get('fit'):
            self.equation_hint.set('当前为'+('手动方程' if self.active_table.get('equation_source')=='manual' else '最小二乘拟合')+'；系数可编辑，应用保存后生效。')
        else:self.equation_hint.set('拟合后显示系数；也可填写六个数值后应用。')
        self.revert_equation_button.configure(state='normal' if self.equation_dirty else 'disabled')

    def _coefficients_edited(self,*args):
        if self._equation_refreshing:return
        self._equation_drafts[self.store.active_id]=[self.coefficient_vars[k].get() for k in 'abcdef']
        self._render_equation()
        if self.on_change:self.on_change()

    def apply_equation(self):
        self._finish_editor()
        try:
            coefficients=np.asarray([float(self.coefficient_vars[k].get()) for k in 'abcdef']).reshape(2,3)
            shape=self._shape_from_fields(required=True)
            if self.gray is not None and shape!=self.gray.shape:raise ValueError('标定尺寸必须与当前原图一致')
            valid=complete_rows(self.active_table['rows'])
            fitted=evaluate_equation([r[2:4] for r in valid],[r[4:6] for r in valid],coefficients)
            self.active_table.update(fit=fitted,equation_source='manual',image_shape=shape)
            self._equation_drafts.pop(self.store.active_id,None)
            self.message.set('手动方程已应用并保存；转换导出使用当前系数。')
            self._refresh();self._notify()
        except (ValueError,TypeError) as exc:
            self.message.set('方程未应用：'+str(exc))
            messagebox.showerror('拟合方程',str(exc),parent=self)

    def revert_equation(self):
        self._equation_drafts.pop(self.store.active_id,None)
        self._refresh_equation()
        self.message.set('已撤销未应用的方程修改。')
        if self.on_change:self.on_change()

    @property
    def active_table(self): return self.store.active

    @property
    def origin_key(self): return self.active_table.get('origin', 'top_left')

    @property
    def frame_center(self):return self.active_table.get('ring_center')

    def export_snapshot(self):
        table = self.active_table
        if self.equation_dirty:raise ValueError('方程修改尚未应用，请先应用保存或撤销修改')
        if table['fit'] is None:
            raise ValueError('工作台“' + table['name'] + '”尚未拟合，或坐标/原点已修改，请重新拟合并保存')
        if not table.get('image_shape'):
            raise ValueError('请填写标定图宽、高并重新拟合')
        return {'coefficients': table['fit'].coefficients, 'worktable': table['name'],
                'image_shape': tuple(table['image_shape']), 'origin': self.origin_key,'ring_center':self.frame_center}

    def _notify(self):
        try: self.store.save()
        except Exception as exc: self.message.set('工作台设置未能保存：' + str(exc))
        if self.on_change: self.on_change()

    def _shape_from_fields(self, required=False):
        w, h = self.width.get().strip(), self.height.get().strip()
        if not w and not h:
            if not required:return None
            shape=self.active_table.get('image_shape')
            if shape:return tuple(shape)
        try: shape = (int(h), int(w))
        except ValueError as exc: raise ValueError('请载入图片或填写有效的原图宽、高') from exc
        if min(shape) < 16: raise ValueError('图片宽、高至少为 16 像素')
        return shape

    def _refresh(self, select_index=None, preserve_selection=True):
        if select_index is None and preserve_selection:
            selected = self.tree.selection()
            if selected: select_index = int(selected[0])
        self._refreshing = True
        try:
            table = self.active_table
            self.selector.configure(values=[t['name'] for t in self.store.tables]); self.name.set(table['name'])
            self.origin_var.set(next(k for k, v in ORIGINS.items() if v == self.origin_key))
            shape=self.gray.shape if self.gray is not None else None
            width,height=(str(shape[1]),str(shape[0])) if shape else (self._size_draft or ('',''))
            self.width.set(width);self.height.set(height)
            self.origin_hint.set('u 向右；v ' + ('向下' if self.origin_key == 'top_left' else '向上') + '，单位 px。直角坐标使用此原点；极坐标固定用亮圆环圆心。')
            self.tree.delete(*self.tree.get_children())
            for i, row in enumerate(table['rows']):
                values = [row.get(k, '') for k in ('id', 'u', 'v', 'X', 'Y')]
                for j in (1, 2):
                    try: values[j] = f'{float(values[j]):.2f}'
                    except (ValueError, TypeError): pass
                self.tree.insert('', 'end', iid=str(i), values=values + [''])
            fit = table.get('fit')
            if fit:
                for row, error in zip(complete_rows(table['rows']), fit.radial_errors): self.tree.set(str(row[0]), 'error', f'{error:.4g}')
                if fit.count:
                    source='手动方程' if table.get('equation_source')=='manual' else '已拟合'
                    self.statistics.set(f'{fit.count} 组，{source}；二维残差 RMS {fit.rms:.6g}，最大 {fit.max_error:.6g}（机台单位）')
                else:self.statistics.set('手动方程已保存；没有完整对应点，未计算残差。')
            else: self.statistics.set('至少 3 组不共线点；修改坐标或原点后需重新拟合。')
            if select_index is not None and str(select_index) in self.tree.get_children():
                self.tree.selection_set(str(select_index)); self.tree.focus(str(select_index)); self.tree.see(str(select_index))
            else: self.tree.focus('')
        finally: self._refreshing = False
        self._target_changed(); self._update_assigned(); self._sync_buttons();self._refresh_equation()

    def _invalidate(self, text='坐标已修改，请重新拟合', clear_undo=True, select_index=None):
        self.active_table['fit'] = None
        self._equation_drafts.pop(self.store.active_id,None)
        if clear_undo: self._undo.clear()
        self.message.set(text); self._refresh(select_index); self._notify()

    def _shape_edited(self, *args):
        if self._refreshing: return
        self._size_draft=(self.width.get(),self.height.get())
        self.active_table['fit'] = None; self._undo.clear()
        self._equation_drafts.pop(self.store.active_id,None);self._refresh_equation()
        self.statistics.set('图片尺寸已修改，请重新拟合。')
        for iid in self.tree.get_children(): self.tree.set(iid, 'error', '')
        self._sync_buttons()
        if self.on_change: self.on_change()

    def _origin_changed(self, event=None):
        selected_origin = ORIGINS[self.origin_var.get()]
        self._finish_editor()
        previous = self.origin_key
        if selected_origin=='ring_center':
            if self.gray is None:
                self.origin_var.set(next(k for k,v in ORIGINS.items() if v==previous))
                self.message.set('请先载入标定图片，再选择亮圆环拟合圆心。');return
            self.message.set('正在拟合标定图片的亮圆环圆心…')
            gray=self.gray;self._start_work('origin_fitted',lambda:_fit_frame_center(gray));return
        self._job_id+=1;self._busy=False
        try:
            shape = self._shape_from_fields()
            if self.gray is not None and shape is not None and shape != self.gray.shape:
                raise ValueError('标定图宽、高必须与左侧原图一致。')
            change_table_origin(self.active_table, selected_origin, shape)
            self._equation_drafts.pop(self.store.active_id,None)
            self._undo.clear(); self.message.set('原点已切换，已有图像坐标同步换算；请重新拟合。')
            self._refresh(); self._notify()
        except Exception as exc:
            self.origin_var.set(next(k for k, v in ORIGINS.items() if v == previous)); self.message.set(str(exc))

    def _reset_extraction(self):
        self._job_id += 1; self._busy = False; self._undo.clear()
        self.gray = None; self.source = ''; self.markers = []; self.manual = False; self._auto_done = False
        self._size_draft=None
        self.preview.clear('选择当前工作台的标定图片后，自动提取坐标')
        self.image_info.set('尚未选择标定图片'); self._sync_buttons(); self._preview_info_changed()

    def _select_table(self, event=None):
        selected_name = self.name.get(); self._finish_editor()
        self.store.active_id = next(t['id'] for t in self.store.tables if t['name'] == selected_name)
        self._reset_extraction(); self.message.set('当前工作台：' + selected_name)
        self._refresh(0, preserve_selection=False); self._notify()

    def _add_table(self):
        self._finish_editor()
        name = simpledialog.askstring('新增工作台', '工作台名称', initialvalue=f'工作台{len(self.store.tables)+1}', parent=self)
        if name is None: return
        try:
            self.store.add(name); self._reset_extraction(); self.message.set('新工作台默认九行；先输入机台坐标，再提取图像坐标。')
            self._refresh(0, preserve_selection=False); self._notify()
        except Exception as exc: messagebox.showerror('工作台', str(exc), parent=self)

    def _rename_table(self):
        name = simpledialog.askstring('工作台改名', '工作台名称', initialvalue=self.name.get(), parent=self)
        if name is None: return
        name = name.strip()
        if not name or any(t['name'] == name and t['id'] != self.store.active_id for t in self.store.tables):
            messagebox.showerror('工作台', '名称不能为空或与其他工作台重复', parent=self); return
        self.active_table['name'] = name; self._refresh(); self._notify()

    def _delete_table(self):
        self._finish_editor()
        if not messagebox.askyesno('删除工作台', '删除“' + self.name.get() + '”及其对应关系？', parent=self): return
        self.store.tables.remove(self.active_table)
        if not self.store.tables: self.store.add('工作台1')
        self.store.active_id = self.store.tables[0]['id']; self._reset_extraction()
        self._refresh(0, preserve_selection=False); self._notify()

    def add_row(self):
        self._finish_editor(); used = {str(r['id']) for r in self.active_table['rows']}; number = 1
        while str(number) in used: number += 1
        self.active_table['rows'].append(empty_row(number))
        self._invalidate(select_index=len(self.active_table['rows'])-1)

    def delete_rows(self):
        self._finish_editor(); selected = {int(i) for i in self.tree.selection()}
        if not selected: return
        self.active_table['rows'] = [r for i, r in enumerate(self.active_table['rows']) if i not in selected]
        self._invalidate('所选行已删除，请重新拟合', select_index=0)

    def _bind_table_wheel(self,widget):
        # Widget bindings run before ttk class and global page bindings.
        for sequence in ('<MouseWheel>','<Button-4>','<Button-5>'):
            widget.bind(sequence,self._table_wheel)

    def _table_wheel(self,event):
        number=getattr(event,'num',None)
        delta=getattr(event,'delta',0)
        if number in (4,5):
            units=-3 if number==4 else 3
        elif delta:
            steps=max(1,int(abs(delta)/120))
            units=(-3 if delta>0 else 3)*steps
        else:return 'break'
        self.tree.yview_scroll(units,'units')
        # Consume both directions even at either boundary; the outer page
        # must not jump when the pointer remains inside the table.
        return 'break'

    def _remember_column(self, event):
        col = self.tree.identify_column(event.x)
        if col and col != '#0': self._cell_column = min(4, int(col[1:])-1)

    def _target_changed(self, event=None):
        selection = self.tree.selection()
        if selection and int(selection[0]) < len(self.active_table['rows']):
            self.target_info.set('当前写入：点 ' + str(self.active_table['rows'][int(selection[0])]['id']))
        else: self.target_info.set('请在表格选择写入行')

    def _edit_cell(self, event):
        self._finish_editor(); iid = self.tree.identify_row(event.y); column = self.tree.identify_column(event.x)
        if not iid or not column or not 1 <= int(column[1:]) <= 5: return
        j = int(column[1:])-1; self._cell_column = j; bounds = self.tree.bbox(iid, column)
        if not bounds: return
        key = ('id', 'u', 'v', 'X', 'Y')[j]; editor = ttk.Entry(self.tree)
        self._bind_table_wheel(editor)
        editor.insert(0, str(self.active_table['rows'][int(iid)].get(key, '')))
        editor.place(x=bounds[0], y=bounds[1], width=bounds[2], height=bounds[3]); editor.focus_set(); editor.select_range(0, 'end')
        self._editor = (editor, int(iid), key)
        editor.bind('<Return>', lambda e: self._finish_editor()); editor.bind('<FocusOut>', lambda e: self._finish_editor())
        editor.bind('<Escape>', lambda e: self._finish_editor(False)); editor.bind('<Control-v>', lambda e: self._paste_from_editor())

    def _paste_from_editor(self):
        try: content = self.clipboard_get()
        except tk.TclError: return 'break'
        if '\t' in content or '\n' in content.strip():
            if self._editor:
                _, i, _ = self._editor; self.tree.selection_set(str(i)); self.tree.focus(str(i))
            self._finish_editor(False); return self._paste()
        return None

    def _finish_editor(self, save=True):
        if self._editor is None: return 'break'
        editor, index, key = self._editor; self._editor = None; value = editor.get().strip(); editor.destroy()
        if save and index < len(self.active_table['rows']) and value != str(self.active_table['rows'][index].get(key, '')):
            self.active_table['rows'][index][key] = value; self._invalidate(select_index=index)
        return 'break'

    def _paste(self, event=None):
        self._finish_editor()
        try:
            text = self.clipboard_get().strip('\r\n'); data = list(csv.reader(io.StringIO(text), delimiter='\t' if '\t' in text else ','))
            if not data: return 'break'
            start = int(self.tree.focus() or (self.tree.selection()[0] if self.tree.selection() else 0)); column = self._cell_column
            if any(len(row) > 5-column for row in data): raise ValueError('粘贴列数超出范围，请先单击正确的起始单元格。')
            rows = self.active_table['rows']; keys = ('id', 'u', 'v', 'X', 'Y')
            while len(rows) < start+len(data): rows.append(empty_row(len(rows)+1))
            for i, row in enumerate(data, start):
                for j, value in enumerate(row, column): rows[i][keys[j]] = value.strip()
            self._invalidate('已粘贴，请核对点号和当前原点，再重新拟合。', select_index=start)
        except Exception as exc: messagebox.showerror('粘贴坐标', str(exc), parent=self)
        return 'break'

    def _import(self):
        self._finish_editor()
        path = filedialog.askopenfilename(parent=self, filetypes=[('坐标表', '*.csv *.txt *.xlsx')])
        if not path: return
        try:
            data = read_correspondences(path, with_metadata=True)
            if any(any(str(r.get(k, '')).strip() for k in ('u', 'v', 'X', 'Y')) for r in self.active_table['rows']):
                if not messagebox.askyesno('导入坐标', '用导入表替换当前工作台的对应关系？', parent=self): return
            origin=data['origin'] or self.origin_key
            center=data.get('ring_center') if data['origin'] else self.frame_center
            if origin=='ring_center':image_frame(data['image_shape'],'ring_center',center)
            self.active_table.update(rows=data['rows'], image_shape=data['image_shape'], origin=origin,
                                     ring_center=center)
            self._invalidate('已导入；无原点信息的旧表按当前原点解释，请核对后拟合。', select_index=0)
        except Exception as exc: messagebox.showerror('导入坐标', str(exc), parent=self)

    def _export(self):
        self._finish_editor()
        path = filedialog.asksaveasfilename(parent=self, defaultextension='.csv', initialfile='calibration.csv', filetypes=[('CSV', '*.csv'), ('TXT', '*.txt')])
        if path:
            try: write_correspondences(path, self.active_table['rows'], self.origin_key, self.active_table.get('image_shape'),self.frame_center)
            except Exception as exc: messagebox.showerror('导出坐标', str(exc), parent=self)

    def _clear_coordinates(self, keys, title, question):
        self._finish_editor()
        if not messagebox.askyesno(title, question, parent=self): return
        for row in self.active_table['rows']:
            for key in keys: row[key] = ''
        self._invalidate(title+'已完成；请补齐后重新拟合。', select_index=0)

    def clear_image_points(self):
        self._clear_coordinates(('u','v'), '清空图像坐标', '清空全部行的 u、v，保留点号、机台 X、Y 和原点？')

    def clear_machine_points(self):
        self._clear_coordinates(('X','Y'), '清空机台坐标', '清空全部行的机台 X、Y，保留点号、图像 u、v 和原点？')

    def fit_current(self):
        self._finish_editor()
        try:
            valid = complete_rows(self.active_table['rows']); shape = self._shape_from_fields(required=True)
            if self.gray is not None and tuple(shape) != self.gray.shape:
                raise ValueError('标定图宽、高必须与左侧原图一致。')
            raw = frame_to_image([r[2:4] for r in valid], shape, self.origin_key,self.frame_center)
            if np.any(raw < -1e-6) or np.any(raw[:, 0] > shape[1]-1+1e-6) or np.any(raw[:, 1] > shape[0]-1+1e-6):
                raise ValueError('部分图像坐标超出图片范围，请核对尺寸及原点。')
            self.active_table['fit'] = fit_affine([r[2:4] for r in valid], [r[4:6] for r in valid]); self.active_table['image_shape'] = shape
            self.active_table['equation_source']='ols';self._equation_drafts.pop(self.store.active_id,None)
            self.message.set('当前工作台已拟合并保存。'); self._refresh(); self._notify()
        except Exception as exc:
            self.active_table['fit'] = None; self._refresh(); self._notify()
            messagebox.showerror('坐标回归', str(exc), parent=self)

    def assign_marker(self, index, marker, shape):
        table = self.active_table
        if not 0 <= index < len(table['rows']): raise ValueError('请先选择有效的写入行')
        old_shape = table.get('image_shape')
        if old_shape and tuple(old_shape) != tuple(shape) and any(str(r.get('u', '')).strip() or str(r.get('v', '')).strip() for r in table['rows']):
            raise ValueError('标定图片尺寸与已有点不一致；请新建工作台或先清空图像坐标。')
        uv = image_to_frame([marker.center], shape, self.origin_key,self.frame_center)[0]
        for i, row in enumerate(table['rows']):
            if i == index: continue
            try: pair = [float(row['u']), float(row['v'])]
            except (ValueError, KeyError, TypeError): continue
            if np.linalg.norm(np.subtract(pair, uv)) < 1:
                raise ValueError('该位置已写入点 ' + str(row['id']) + '；如需修改，请先选中该行。')
        self._undo.append((copy.deepcopy(table), index)); self._undo = self._undo[-50:]
        identifier = str(table['rows'][index]['id'])
        table['rows'][index].update(u=f'{uv[0]:.8f}', v=f'{uv[1]:.8f}'); table['image_shape'] = tuple(map(int, shape)); table['fit'] = None
        self._equation_drafts.pop(self.store.active_id,None)
        next_index = index+1 if self.advance.get() else index
        if next_index >= len(table['rows']): next_index = None
        self.message.set('已写入点 '+identifier+'；'+('请点击下一处。' if next_index is not None else '已到最后一行；可选择要修改的行或增加行。'))
        self._refresh(next_index, preserve_selection=False); self._notify()

    def undo_assignment(self):
        self._finish_editor()
        if not self._undo: return
        before, index = self._undo.pop()
        if before['id'] != self.store.active_id: self._undo.clear(); return
        table = self.active_table; table.clear(); table.update(before)
        self.preview.selected = None; self.message.set('已撤销上次取点；重新选择该位置即可写入。')
        self._refresh(index, preserve_selection=False); self._notify()

    def choose_image(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[('图像', '*.bmp *.png *.jpg *.jpeg *.tif *.tiff')])
        if path: self.load_path(path)

    def use_current_image(self):
        if not self.image_provider: return
        result = self.image_provider()
        try:
            if result and result[0] is not None: self.set_image(result[0], str(result[1] or '当前图像'))
            else: self.message.set('请先在操作与显示页载入图片，或直接选择标定图片。')
        except Exception as exc: self.message.set(str(exc))

    def _start_work(self, kind, callback):
        self._job_id += 1; job_id = self._job_id; table_id = self.store.active_id
        self._busy = True; self._sync_buttons()
        def worker():
            try: self.events.put((job_id, table_id, kind, callback()))
            except Exception as exc: self.events.put((job_id, table_id, 'error', str(exc)))
        threading.Thread(target=worker, daemon=True, name='ols-inline').start()

    def load_path(self, path):
        self._finish_editor(); self.message.set('正在读取标定原图…')
        def load():
            gray = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None: raise ValueError('无法读取图片：'+str(path))
            return gray, str(path)
        self._start_work('loaded', load)

    def set_image(self, gray, source='当前图像'):
        gray = _gray(gray); table = self.active_table; previous_shape = table.get('image_shape')
        has_points = any(str(row.get(k, '')).strip() for row in table['rows'] for k in ('u', 'v'))
        if previous_shape and tuple(previous_shape) != gray.shape and has_points:
            raise ValueError('当前工作台已有其他尺寸图片的点，请新建工作台或先清空图像坐标。')
        self._job_id += 1; self._busy = False; self._undo.clear()
        self.gray = gray; self.source = source; self.markers = []; self.manual = False; self._auto_done = False
        self._size_draft=None
        table['image_shape'] = tuple(map(int, gray.shape))
        self.image_info.set(f'{Path(source).name} · {gray.shape[1]} × {gray.shape[0]}')
        self.preview.set_image(gray); self.preview.arm('pick')
        self.message.set('选择检测方式后，点击“自动提取图像坐标”；再按点号点击候选点。')
        self._refresh(); self._notify(); self._preview_info_changed()
        if self.origin_key=='ring_center':
            self._start_work('origin_fitted',lambda:_fit_frame_center(gray))

    def auto_detect(self):
        self._finish_editor()
        if self.gray is None:
            self.message.set('请先选择标定图片。'); return
        if self._busy: return
        gray, mode = self.gray, self._mode_key(); self.manual = False; self._auto_done = False
        self.preview.arm('pick'); self.markers = []; self.preview.markers = []; self.preview.selected = None; self.preview.draw_overlays()
        self.message.set('正在自动提取图像坐标…')
        self._start_work('detected', lambda: detect_markers(gray, mode))

    def _mode_key(self): return 'centroid' if self.mode.get().startswith('质心') else 'ring'

    def _mode_changed(self, event=None):
        self._job_id += 1; self._busy = False; self.manual = False; self._auto_done = False; self.markers = []
        self.preview.markers = []; self.preview.selected = None; self.preview.arm('pick')
        self.message.set('检测方式已切换，请重新自动提取。'); self._sync_buttons()

    def toggle_manual(self):
        if self.gray is None or self._busy or not self._auto_done: return
        self.manual = not self.manual; self.preview.arm('pick')
        self.message.set('手动补点：点击漏检目标附近，提取成功后直接写入当前行。' if self.manual else '点击候选点直接写入当前行。')
        self._sync_buttons()

    def _cancel_manual(self):
        self.manual = False; self._sync_buttons()

    def _arm_polygon(self):
        if self.manual and self._mode_key() == 'centroid': self.preview.arm('polygon')

    def _pick(self, seed):
        if self.gray is None or self._busy: return
        self._finish_editor(); selected = self.tree.selection()
        if not selected: self.message.set('请先在坐标表中选择写入行，或增加行。'); return
        try:
            mark = min(self.markers, key=lambda m: np.linalg.norm(np.subtract(seed, m.center))) if self.markers else None
            if mark is not None and np.linalg.norm(np.subtract(seed, mark.center)) > max(8/max(self.preview.scale, .01), 4): mark = None
            if mark is None:
                if not self.manual:
                    self.message.set('此处没有候选点；漏检时请开启“手动补点”。'); return
                radius = int(self.radius.get()) if self.radius.get().strip() else None
                mark = extract_marker(self.gray, seed, self._mode_key(), radius)
            self.assign_marker(int(selected[0]), mark, self.gray.shape)
            if not any(np.linalg.norm(np.subtract(mark.center, m.center)) < 1 for m in self.markers):
                self.markers.append(mark); self.preview.markers = self.markers
            self.preview.selected = mark; self.preview.draw_overlays()
        except Exception as exc: self.message.set(str(exc))

    def _polygon_done(self, points):
        if self.gray is None or not self.manual or self._mode_key() != 'centroid': return
        selected = self.tree.selection()
        if not selected: self.message.set('请先选择写入行。'); return
        try:
            mark = marker_from_polygon(points); self.assign_marker(int(selected[0]), mark, self.gray.shape)
            self.preview.selected = mark; self.preview.draw_overlays()
        except Exception as exc: self.message.set(str(exc))

    def _update_assigned(self):
        assigned = []; shape = self.gray.shape if self.gray is not None else self.active_table.get('image_shape')
        for row in self.active_table['rows']:
            try:
                uv = [float(row.get('u', '')), float(row.get('v', ''))]
                if not np.isfinite(uv).all(): continue
                raw = frame_to_image([uv], shape, self.origin_key,self.frame_center)[0]; assigned.append((row['id'], *raw))
            except (ValueError, TypeError): pass
        self.preview.assigned = assigned; self.preview.coordinate_origin = self.origin_key
        self.preview.coordinate_center=self.frame_center
        self.preview.draw_overlays()

    def _sync_buttons(self):
        ready = self.gray is not None and not self._busy
        self.extract_button.configure(state='normal' if ready else 'disabled')
        self.manual_button.configure(state='normal' if ready and self._auto_done else 'disabled', text='结束手动补点' if self.manual else '手动补点（漏检）')
        self.undo_button.configure(state='normal' if self._undo and not self._busy else 'disabled')
        self.current_button.configure(state='normal' if self.image_provider and not self._busy else 'disabled')
        self.mode_combo.configure(state='disabled' if self._busy else 'readonly')
        if self.manual: self.manual_settings.pack(fill='x', before=self.manual_settings.master.winfo_children()[-1])
        else: self.manual_settings.pack_forget()
        self.polygon_button.configure(state='normal' if ready and self.manual and self._mode_key() == 'centroid' else 'disabled')

    def show_preview(self):
        self.preview.pack(fill='both', expand=True)
        self._preview_info_changed()

    def hide_preview(self): self.preview.pack_forget()

    def _preview_info_changed(self):
        if self.on_preview_info: self.on_preview_info(self.image_info.get())

    def _poll(self):
        if self._closed: return
        for _ in range(8):
            try: job, table, kind, result = self.events.get_nowait()
            except queue.Empty: break
            if job != self._job_id or table != self.store.active_id: continue
            self._busy = False
            try:
                if kind == 'loaded': self.set_image(*result)
                elif kind=='origin_fitted':
                    change_table_origin(self.active_table,'ring_center',self.gray.shape,result)
                    self._undo.clear();self._refresh();self._notify()
                    self.message.set(f'圆心已拟合：({result[0]:.3f}, {result[1]:.3f})；坐标以此为原点。')
                elif kind == 'detected':
                    self._auto_done = True
                    self.markers = result; self.preview.markers = result; self.preview.draw_overlays()
                    self.message.set(f'找到 {len(result)} 个候选（疑似 {sum(m.details.get("candidate_quality")=="疑似" for m in result)} 个），表格 {len(self.active_table["rows"])} 行。点击左侧目标直接写入；漏检再手动补点。')
                else:
                    self.origin_var.set(next(k for k,v in ORIGINS.items() if v==self.origin_key))
                    self.message.set(str(result))
            except Exception as exc: self.message.set(str(exc))
            self._sync_buttons()
        self._poll_after = self.after(70, self._poll)

    def _destroyed(self, event):
        if event.widget is not self: return
        self._closed = True; self._job_id += 1
        if self._poll_after:
            try: self.after_cancel(self._poll_after)
            except tk.TclError: pass


class MarkerImageView(ttk.Frame):
    """Original-pixel crop rendering with vector marker overlays and gestures."""
    def __init__(self, parent, pick_callback, polygon_callback):
        super().__init__(parent)
        self.pick_callback = pick_callback; self.polygon_callback = polygon_callback
        self.image = None; self.photo = None; self.scale = 1.; self.x = 0.; self.y = 0.
        self._needs_fit = False
        self.coordinate_origin = 'top_left';self.coordinate_center=None; self.on_cancel = None
        self.mode = 'pick'; self.pan = None; self.vertices = []
        self.markers = []; self.selected = None; self.assigned = []
        self.comparison_mode=False
        self.show_axes=False;self.focus_point=None;self.comparison_pairs=[];self.comparison_current=None
        self.canvas = tk.Canvas(self, bg='#050505', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.info = tk.StringVar(value='点击候选点直接写入 · 滚轮缩放 · 右键拖动')
        self.pointer_text=tk.StringVar(value='u —   v — px')
        self.on_position_click=None
        row = ttk.Frame(self); row.pack(fill='x', pady=5)
        ttk.Label(row, textvariable=self.info).pack(side='left')
        ttk.Button(row, text='适应窗口', command=self.fit).pack(side='right',before=row.winfo_children()[0])
        ttk.Label(row,textvariable=self.pointer_text,foreground='#76B4FF').pack(side='right',padx=10,before=row.winfo_children()[0])
        self.canvas.bind('<Configure>', self._resize)
        self.canvas.bind('<MouseWheel>', lambda e: self.zoom(e, 1.2 if e.delta > 0 else 1 / 1.2))
        self.canvas.bind('<Button-4>', lambda e: self.zoom(e, 1.2))
        self.canvas.bind('<Button-5>', lambda e: self.zoom(e, 1 / 1.2))
        self.canvas.bind('<ButtonPress-3>', self._pan_start)
        self.canvas.bind('<B3-Motion>', self._pan_move)
        self.canvas.bind('<ButtonRelease-3>', lambda e: setattr(self, 'pan', None))
        self.canvas.bind('<Button-1>', self._click)
        self.canvas.bind('<Double-Button-1>', self._double_click)
        self.canvas.bind('<Return>', self._finish_polygon)
        self.canvas.bind('<Escape>', self._cancel)
        self.canvas.bind('<Motion>',self._pointer_motion)
        self.canvas.bind('<Leave>',lambda e:self.pointer_text.set('u —   v — px'))
        self.clear()

    def clear(self, text='选择标定图片，模式选择后自动提取'):
        self.image = None; self.photo = None; self.markers = []; self.selected = None
        self.assigned = []; self.vertices = []; self.pan = None
        self.focus_point=None;self.comparison_pairs=[];self.comparison_current=None
        self.canvas.delete('all')
        self.canvas.create_text(22, 22, text=text, fill='#AAAAAA', anchor='nw')

    def _cancel(self, event=None):
        self.arm('pick')
        if self.on_cancel: self.on_cancel()
        return 'break'

    def set_image(self, gray):
        self.canvas.delete('all')
        self.image = Image.fromarray(gray)
        self.markers = []; self.selected = None; self.vertices = []
        self.focus_point=None;self.comparison_pairs=[];self.comparison_current=None
        self._needs_fit = True
        self.fit()

    def _resize(self, event):
        if self._needs_fit: self.fit()
        else: self.draw()

    def fit(self):
        if self.image is None: return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if width < 50 or height < 50:
            self._needs_fit = True
            return
        self._needs_fit = False
        self.scale = min(width / self.image.width, height / self.image.height)
        self.x = (width - self.image.width * self.scale) / 2
        self.y = (height - self.image.height * self.scale) / 2
        self.draw()

    def arm(self, mode):
        self.mode = mode; self.vertices = []
        self.canvas.focus_set(); self.canvas.configure(cursor='crosshair')
        self.info.set('沿形变外沿逐点圈定，Enter/双击闭合后写入；Esc 取消' if mode == 'polygon' else '点击候选点直接写入 · 滚轮缩放 · 右键拖动')
        self.draw_overlays()
        return 'break'

    def zoom(self, event, ratio):
        if self.image is None: return 'break'
        updated = min(40., max(.02, self.scale * ratio)); ratio = updated / self.scale
        self.x = event.x - (event.x - self.x) * ratio
        self.y = event.y - (event.y - self.y) * ratio
        self.scale = updated; self.draw()
        return 'break'

    def image_point(self, event):
        if self.image is None: return None
        u, v = (event.x - self.x) / self.scale, (event.y - self.y) / self.scale
        if 0 <= u < self.image.width and 0 <= v < self.image.height: return (u, v)
        return None

    def _pointer_motion(self,event):
        point=self.image_point(event)
        if point is None:self.pointer_text.set('u —   v — px');return
        try:uv=image_to_frame([point],(self.image.height,self.image.width),self.coordinate_origin,self.coordinate_center)[0]
        except ValueError:self.pointer_text.set('亮圆环圆心待拟合');return
        self.pointer_text.set(f'u {uv[0]:.3f}   v {uv[1]:.3f} px')

    def _click(self, event):
        self.canvas.focus_set()
        point = self.image_point(event)
        if point is None: return 'break'
        self._pointer_motion(event)
        if self.on_position_click:self.on_position_click(point)
        if self.mode == 'polygon':
            if not self.vertices or np.linalg.norm(np.subtract(point, self.vertices[-1])) * self.scale > 1:
                self.vertices.append(point)
            self.draw_overlays()
        else:
            self.pick_callback(point)
        return 'break'

    def _finish_polygon(self, event=None):
        if self.mode == 'polygon' and len(self.vertices) >= 3:
            points = self.vertices.copy(); self.arm('pick'); self.polygon_callback(points)
        return 'break'

    def _double_click(self, event):
        if self.mode == 'polygon': return self._finish_polygon(event)
        return self._click(event)

    def _pan_start(self, event):
        self.pan = (event.x, event.y)
        return 'break'

    def _pan_move(self, event):
        if self.pan:
            self.x += event.x - self.pan[0]; self.y += event.y - self.pan[1]
            self.pan = (event.x, event.y); self.draw()
        return 'break'

    def _canvas_xy(self, points):
        p = np.asarray(points).reshape(-1, 2)
        return np.c_[self.x + p[:, 0] * self.scale, self.y + p[:, 1] * self.scale].ravel().tolist()

    def draw_overlays(self):
        self.canvas.delete('marker')
        if self.image is None: return
        try:origin, _ = image_frame((self.image.height, self.image.width), self.coordinate_origin,self.coordinate_center)
        except ValueError:origin=None
        if self.show_axes and origin is not None:self._draw_coordinate_axes(origin)
        if origin is not None:x, y = self._canvas_xy([origin])
        if origin is not None and (not self.comparison_mode or self.show_axes):
            self.canvas.create_line(x-7, y, x+7, y, fill='#FFE164', width=1, tags=('marker','coordinate_origin'))
            self.canvas.create_line(x, y-7, x, y+7, fill='#FFE164', width=1, tags=('marker','coordinate_origin'))
            self.canvas.create_text(x+9, y-12 if y > 50 else y+12, text='原点 (0,0)', fill='#FFE164', anchor='w', tags=('marker','coordinate_origin'))
        compared=[]
        if self.comparison_mode:
            for pair in self.comparison_pairs+([self.comparison_current] if self.comparison_current else []):
                compared.extend(pair[k] for k in ('measured','predicted') if pair.get(k) is not None)
        for i, mark in enumerate(self.markers):
            p = mark.contour
            if len(p) > 2: self.canvas.create_line(*self._canvas_xy(np.r_[p, p[:1]]), fill='#38BDD5', width=1, tags='marker')
            x, y = self._canvas_xy([mark.center])
            self.canvas.create_oval(x-5, y-5, x+5, y+5, outline='#38BDD5', width=1, tags='marker')
            if not any(np.linalg.norm(np.subtract(mark.center,p))*self.scale<18 for p in compared):
                self.canvas.create_text(x+8, y-9, text=f'候选{i+1}'+('（疑似）' if mark.details.get('candidate_quality')=='疑似' else ''), fill='#38BDD5', anchor='w', tags='marker')
        for identifier, u, v in self.assigned:
            x, y = self._canvas_xy([(u, v)])
            self.canvas.create_line(x-7, y, x+7, y, fill='#65E98D', width=2, tags='marker')
            self.canvas.create_line(x, y-7, x, y+7, fill='#65E98D', width=2, tags='marker')
            self.canvas.create_text(x+8, y+10, text='点 '+str(identifier), fill='#65E98D', anchor='w', tags='marker')
        if self.selected:
            m = self.selected; p = m.contour
            self.canvas.create_line(*self._canvas_xy(np.r_[p, p[:1]]), fill='#FFE164', width=2, tags='marker')
            x, y = self._canvas_xy([m.center])
            if not self.comparison_mode:
                self.canvas.create_line(x-8, y, x+8, y, fill='#FF625C', width=2, tags='marker')
                self.canvas.create_line(x, y-8, x, y+8, fill='#FF625C', width=2, tags='marker')
        if len(self.vertices) > 1:
            self.canvas.create_line(*self._canvas_xy(self.vertices), fill='#FFE164', width=2, tags='marker')
        for pair in self.comparison_pairs:self._draw_comparison_pair(pair)
        if self.comparison_current:self._draw_comparison_pair(self.comparison_current,current=True)
        if self.focus_point is not None and self.show_axes and origin is not None:self._draw_projections(origin,self.focus_point)

    def _draw_coordinate_axes(self,origin):
        h,w=self.image.height,self.image.width
        _,sign=image_frame((h,w),self.coordinate_origin,self.coordinate_center)
        self.canvas.create_line(*self._canvas_xy([(0,origin[1]),(w-1,origin[1])]),
                                fill='#FFE164',width=1,arrow='last',tags=('marker','coordinate_axis'))
        vertical=[(origin[0],0),(origin[0],h-1)]
        if sign<0:vertical=vertical[::-1]
        self.canvas.create_line(*self._canvas_xy(vertical),fill='#FFE164',width=1,arrow='last',tags=('marker','coordinate_axis'))
        x,y=self._canvas_xy([(w-1,origin[1])])
        self.canvas.create_text(x-12,y+14,text='u / px',fill='#FFE164',anchor='e',tags=('marker','coordinate_axis'))
        x,y=self._canvas_xy([vertical[-1]])
        self.canvas.create_text(x+10,y+14 if sign<0 else y-14,text='v / px',fill='#FFE164',anchor='w',tags=('marker','coordinate_axis'))

    def _draw_projections(self,origin,point):
        p=np.asarray(point,float);uv=image_to_frame([p],(self.image.height,self.image.width),self.coordinate_origin,self.coordinate_center)[0]
        for a,b in [((origin[0],p[1]),p),((p[0],origin[1]),p)]:
            self.canvas.create_line(*self._canvas_xy([a,b]),fill='#83C8FF',width=1,dash=(5,4),tags=('marker','point_projection'))
        x,y=self._canvas_xy([(p[0],origin[1])])
        self.canvas.create_text(x+4,y+14,text=f'u={uv[0]:.3f}',fill='#83C8FF',anchor='w',tags=('marker','point_projection'))
        x,y=self._canvas_xy([(origin[0],p[1])])
        self.canvas.create_text(x+7,y-12,text=f'v={uv[1]:.3f}',fill='#83C8FF',anchor='w',tags=('marker','point_projection'))

    def _draw_comparison_pair(self,pair,current=False):
        actual=pair.get('measured');predicted=pair.get('predicted')
        if actual is not None and predicted is not None:
            self.canvas.create_line(*self._canvas_xy([actual,predicted]),fill='#D6A6EE',dash=(3,3),
                                    width=2 if pair.get('highlight') else 1,tags=('marker','comparison_link'))
        size=8 if current or pair.get('highlight') else 5
        identifier='当前' if current else '#'+str(pair['id'])
        if actual is not None:
            x,y=self._canvas_xy([actual])
            self.canvas.create_line(x-size,y,x+size,y,fill='#65E98D',width=2,tags=('marker','comparison_measured'))
            self.canvas.create_line(x,y-size,x,y+size,fill='#65E98D',width=2,tags=('marker','comparison_measured'))
            self.canvas.create_text(x+10,y-13,text=identifier+' 实测',fill='#65E98D',anchor='w',tags=('marker','comparison_measured'))
        if predicted is not None:
            x,y=self._canvas_xy([predicted])
            self.canvas.create_polygon(x,y-size,x+size,y,x,y+size,x-size,y,outline='#C28AFF',fill='',width=2,
                                       tags=('marker','comparison_predicted'))
            self.canvas.create_text(x+10,y+13,text=identifier+' 反算',fill='#C28AFF',anchor='w',tags=('marker','comparison_predicted'))

    def draw(self):
        if self.image is None: return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self.scale <= 0: return
        u0, v0 = max(0, (0 - self.x) / self.scale), max(0, (0 - self.y) / self.scale)
        u1, v1 = min(self.image.width, (width - self.x) / self.scale), min(self.image.height, (height - self.y) / self.scale)
        self.canvas.delete('bitmap')
        if u1 > u0 and v1 > v0:
            bounds = (math.floor(u0), math.floor(v0), math.ceil(u1), math.ceil(v1))
            tile = self.image.crop(bounds)
            scaled = (max(1, round(tile.width * self.scale)), max(1, round(tile.height * self.scale)))
            tile = tile.resize(scaled, Image.Resampling.NEAREST if self.scale > 2 else Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(tile)
            self.canvas.create_image(self.x + bounds[0]*self.scale, self.y + bounds[1]*self.scale, image=self.photo, anchor='nw', tags='bitmap')
            self.canvas.tag_lower('bitmap')
        self.draw_overlays()



CHECK_MODES={'图像－机台（图像为实测）':'image','机台－图像（机台为实测）':'machine'}


class VerificationPanel(ttk.Frame):
    """Independent measured pairs; compare in the user-selected reference frame."""
    def __init__(self,parent,calibration,preview_parent,on_preview_info=None,image_provider=None):
        super().__init__(parent)
        self.calibration=calibration;self.on_preview_info=on_preview_info;self.image_provider=image_provider
        self.gray=None;self.source='';self.image_id='';self.markers=[];self.records=[];self._image_center=None
        self._job=0;self._closed=False;self._busy=False;self.events=queue.Queue();self._snapshot_key=None
        self._input_after=None;self._updating_inputs=False;self._selected_record_id=None;self._focus_kind='measured'
        self.name=tk.StringVar();self.equation=tk.StringVar();self.image_info=tk.StringVar(value='尚未选择复检图片')
        self.mode=tk.StringVar(value='质心（戳破薄膜）');self.pick_mode=tk.StringVar(value='候选 / 局部精提取')
        self.check_mode=tk.StringVar(value=next(iter(CHECK_MODES)));self.mode_note=tk.StringVar()
        self.uv=[tk.StringVar(),tk.StringVar()];self.xy=[tk.StringVar(),tk.StringVar()]
        self.actual=self.xy  # Independent machine measurement; never filled by image conversion.
        self.predicted=[tk.StringVar(),tk.StringVar()];self.prediction_label=tk.StringVar()
        self.tolerances={'image':tk.StringVar(),'machine':tk.StringVar()};self.tolerance=self.tolerances['image']
        self.tolerance_label=tk.StringVar();self.show_axes=tk.BooleanVar(value=True)
        self.message=tk.StringVar(value='选择复检模式，分别提供图像实测点和独立机台实测坐标。')
        self.result=tk.StringVar();self.statistics=tk.StringVar()
        self.preview=MarkerImageView(preview_parent,self._pick,lambda points:None)
        self.preview.comparison_mode=True
        self.preview.clear('选择复检图片；绿色为图像实测，紫色为机台坐标反算')
        self.preview.info.set('绿色：图像实测 · 紫色：机台反算 · 编号对应记录')
        self.preview.on_position_click=self._position_clicked
        self._build();self._comparison_mode_changed();self.refresh_tables()
        for var in self.uv:var.trace_add('write',lambda *a:self._inputs_changed('measured'))
        for var in self.xy:var.trace_add('write',lambda *a:self._inputs_changed('predicted'))
        for var in self.tolerances.values():var.trace_add('write',lambda *a:self._inputs_changed())
        self._after=self.after(70,self._poll);self.bind('<Destroy>',self._destroyed,add='+')

    @property
    def comparison_space(self):return CHECK_MODES[self.check_mode.get()]

    @property
    def frame_center(self):return self._image_center if self.gray is not None else self.calibration.frame_center

    def _group(self,title):
        group=ttk.LabelFrame(self,text=title,padding=8);group.pack(fill='x',pady=(0,8));return group

    def _pair(self,parent,label,variables,readonly=False):
        row=ttk.Frame(parent);row.pack(fill='x',pady=3)
        options={'textvariable':label} if isinstance(label,tk.Variable) else {'text':label}
        ttk.Label(row,width=17,**options).pack(side='left')
        for var in variables:ttk.Entry(row,textvariable=var,width=11,state='readonly' if readonly else 'normal').pack(side='left',fill='x',expand=True,padx=(3,0))
        return row

    def _build(self):
        group=self._group('复检工作台')
        self.selector=ttk.Combobox(group,textvariable=self.name,state='readonly');self.selector.pack(fill='x')
        self.selector.bind('<<ComboboxSelected>>',self._select_table)
        self.equation_label=ttk.Label(group,textvariable=self.equation,wraplength=370,justify='left',foreground='#76B4FF')
        self.equation_label.pack(fill='x',pady=(5,0))
        group=self._group('复检模式')
        self.check_mode_combo=ttk.Combobox(group,textvariable=self.check_mode,values=list(CHECK_MODES),state='readonly')
        self.check_mode_combo.pack(fill='x');self.check_mode_combo.bind('<<ComboboxSelected>>',self._comparison_mode_changed)
        ttk.Label(group,textvariable=self.mode_note,wraplength=370,foreground='#AAAAAA').pack(fill='x',pady=(4,0))
        group=self._group('复检图像与取点')
        row=ttk.Frame(group);row.pack(fill='x')
        self.open_button=ttk.Button(row,text='选择复检图片',command=self.choose_image)
        self.open_button.pack(side='left',fill='x',expand=True,padx=(0,3))
        ttk.Button(row,text='使用检测主图',command=self.use_current_image).pack(side='left',fill='x',expand=True)
        ttk.Label(group,textvariable=self.image_info,wraplength=365,foreground='#AAAAAA').pack(fill='x',pady=4)
        self.mode_combo=ttk.Combobox(group,textvariable=self.mode,values=('质心（戳破薄膜）','圆环（黑色圆环内孔）'),state='readonly')
        self.mode_combo.pack(fill='x');self.mode_combo.bind('<<ComboboxSelected>>',self._mode_changed)
        self.detect_button=ttk.Button(group,text='自动检测复检特征',command=self.auto_detect);self.detect_button.pack(fill='x',pady=4)
        row=ttk.Frame(group);row.pack(fill='x')
        ttk.Label(row,text='图上取点方式').pack(side='left')
        ttk.Combobox(row,textvariable=self.pick_mode,values=('候选 / 局部精提取','直接像素点'),state='readonly',width=20).pack(side='right')
        TickCheckbutton(group,text='显示图像坐标系及当前点投影',variable=self.show_axes,command=self._refresh_overlays).pack(anchor='w',pady=(6,0))
        group=self._group('实测坐标与比较')
        self.image_row=self._pair(group,'图像实测 u、v / px',self.uv)
        self.machine_row=self._pair(group,'机台实测 X、Y',self.xy)
        self.predicted_row=self._pair(group,self.prediction_label,self.predicted,readonly=True)
        row=ttk.Frame(group);row.pack(fill='x',pady=3)
        ttk.Label(row,textvariable=self.tolerance_label).pack(side='left')
        self.tolerance_entry=ttk.Entry(row,textvariable=self.tolerance,width=12);self.tolerance_entry.pack(side='right')
        row=ttk.Frame(group);row.pack(fill='x',pady=4)
        ttk.Button(row,text='计算并比较',command=self.calculate).pack(side='left',fill='x',expand=True,padx=(0,3))
        ttk.Button(row,text='加入复检记录',command=self.compare).pack(side='left',fill='x',expand=True)
        self.result_label=ttk.Label(group,textvariable=self.result,wraplength=370,foreground='#76B4FF',justify='left');self.result_label.pack(fill='x')
        group=self._group('本次复检记录')
        box=ttk.Frame(group);box.pack(fill='x')
        style=ttk.Style(self);base=tkfont.Font(self,font=style.lookup('.','font') or 'TkDefaultFont').actual()
        compact=(base['family'],max(9,base['size']-1))
        style.configure('Verification.Treeview',font=compact,rowheight=26)
        style.configure('Verification.Treeview.Heading',font=compact)
        self.tree=ttk.Treeview(box,columns=('id','m1','m2','p1','p2','error'),show='headings',height=5,style='Verification.Treeview')
        for key in self.tree['columns']:self.tree.column(key,width=62,minwidth=42,stretch=True,anchor='center')
        self.tree.column('id',width=35,minwidth=30);self.tree.column('error',width=82)
        self.tree.tag_configure('historical',foreground='#888888')
        scroll=ttk.Scrollbar(box,orient='vertical',command=self.tree.yview);scroll.pack(side='right',fill='y')
        self.tree.pack(fill='x');self.tree.configure(yscrollcommand=scroll.set)
        horizontal=ttk.Scrollbar(group,orient='horizontal',command=self.tree.xview);horizontal.pack(fill='x');self.tree.configure(xscrollcommand=horizontal.set)
        for widget in (self.tree,scroll,horizontal):
            for sequence in ('<MouseWheel>','<Button-4>','<Button-5>'):widget.bind(sequence,self._table_wheel)
        self.tree.bind('<<TreeviewSelect>>',self._record_selected)
        row=ttk.Frame(group);row.pack(fill='x',pady=4)
        ttk.Button(row,text='删除所选',command=self.delete_records).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='清空记录',command=self.clear_records).pack(side='left',fill='x',expand=True,padx=3)
        ttk.Button(row,text='导出当前模式',command=self.export_records).pack(side='left',fill='x',expand=True)
        ttk.Label(group,textvariable=self.statistics,wraplength=370,foreground='#76B4FF').pack(fill='x')
        ttk.Label(self,textvariable=self.message,wraplength=375,foreground='#76B4FF',justify='left').pack(fill='x',pady=4)

    def _table_wheel(self,event):
        self.tree.yview_scroll(-3 if getattr(event,'num',None)==4 or getattr(event,'delta',0)>0 else 3,'units');return 'break'

    def _comparison_mode_changed(self,event=None):
        space=self.comparison_space;image=space=='image'
        self.tolerance=self.tolerances[space];self.tolerance_entry.configure(textvariable=self.tolerance)
        self.mode_note.set('以图像实测 u、v 为基准，机台实测 X、Y 反算后比较。' if image else '以机台实测 X、Y 为基准，图像实测 u、v 推算后比较。')
        self.prediction_label.set('机台反算 u、v / px' if image else '图像推算 X、Y')
        self.tolerance_label.set('允许二维偏差 / '+('px' if image else '机台单位'))
        self.image_row.pack_forget();self.machine_row.pack_forget()
        for row in ((self.image_row,self.machine_row) if image else (self.machine_row,self.image_row)):
            row.pack(fill='x',pady=3,before=self.predicted_row)
        titles=('序号','实测u','实测v','反算u','反算v','偏差/px') if image else ('序号','实测X','实测Y','推算X','推算Y','偏差/机台')
        for key,title in zip(self.tree['columns'],titles):self.tree.heading(key,text=title)
        self._selected_record_id=None;self.result.set('');self._refresh_records();self._refresh_current()

    def _inputs_changed(self,kind=None):
        if self._updating_inputs:return
        if kind:self._focus_kind=kind
        self._selected_record_id=None;self.result.set('')
        if self._input_after:self.after_cancel(self._input_after)
        self._input_after=self.after(65,self._refresh_current)

    @staticmethod
    def _numbers(variables,title):
        try:values=np.array([float(v.get()) for v in variables],float)
        except (ValueError,TypeError) as exc:raise ValueError('请填写完整的'+title) from exc
        if not np.isfinite(values).all():raise ValueError(title+'必须为有限数值')
        return values

    @staticmethod
    def _put(variables,values):
        for v,x in zip(variables,values):v.set(f'{float(x):.6f}')

    def _optional(self,variables):
        try:return self._numbers(variables,'坐标')
        except ValueError:return None

    def _snapshot(self):
        snap=self.calibration.export_snapshot()
        if self.gray is not None and tuple(self.gray.shape)!=tuple(snap['image_shape']):
            raise ValueError('复检图片尺寸与该工作台标定尺寸不一致，请使用相同尺寸原图')
        return snap

    def _preview_values(self):
        uv=self._optional(self.uv);xy=self._optional(self.xy);pred_uv=None;pred_xy=None
        try:
            snap=self._snapshot()
            if xy is not None:pred_uv=inverse_transform_points([xy],snap['coefficients'])[0]
            if uv is not None:pred_xy=transform_points([uv],snap['coefficients'])[0]
        except ValueError:pass
        return uv,xy,pred_uv,pred_xy

    def _refresh_current(self):
        self._input_after=None
        uv,xy,pred_uv,pred_xy=self._preview_values()
        target=pred_uv if self.comparison_space=='image' else pred_xy
        if target is not None:self._put(self.predicted,target)
        else:
            for v in self.predicted:v.set('')
        self._refresh_overlays((uv,xy,pred_uv,pred_xy))

    def _compute(self):
        snap=self._snapshot();uv=self._numbers(self.uv,'图像实测 u、v');xy=self._numbers(self.xy,'机台实测 X、Y')
        tolerance=float(self.tolerance.get()) if self.tolerance.get().strip() else None
        result=compare_coordinate_pair([uv],[xy],snap['coefficients'],self.comparison_space,tolerance)
        return snap,uv,xy,tolerance,result

    def _display_result(self,outcome):
        delta=outcome['delta'][0];error=float(outcome['errors'][0]);passed=outcome['passed']
        status='未设容差' if passed is None else '合格' if passed[0] else '超差'
        axes=('u','v') if outcome['space']=='image' else ('X','Y')
        self.result.set(f'Δ{axes[0]} = {delta[0]:.6f}；Δ{axes[1]} = {delta[1]:.6f}\n二维偏差 = {error:.6f} {outcome["unit"]}；{status}')
        self.result_label.configure(foreground='#FFD15C' if passed is None else '#69E58C' if passed[0] else '#FF7979')
        self._put(self.predicted,outcome['predicted'][0]);self._refresh_overlays()
        return status

    def calculate(self):
        self._refresh_current()
        try:
            _,_,_,_,outcome=self._compute();self._display_result(outcome)
            self.message.set('已在'+('图像坐标' if self.comparison_space=='image' else '机台坐标')+'中比较；Δ=推算−实测。')
            return outcome
        except (ValueError,TypeError) as exc:self.message.set(str(exc));return None

    def compare(self):
        try:
            snap,uv,xy,tolerance,outcome=self._compute();status=self._display_result(outcome)
            pred_uv=outcome['predicted_uv'][0];pred_xy=outcome['predicted_xy'][0]
            duv=pred_uv-uv;dxy=pred_xy-xy
            row={'id':max((r['id'] for r in self.records),default=0)+1,'schema_version':2,'space':self.comparison_space,
                 'worktable_id':self.calibration.store.active_id,'worktable':snap['worktable'],'source':self.source,
                 'image_id':self.image_id,'image_shape':list(snap['image_shape']),'origin':snap['origin'],
                 'frame_center':self.frame_center,'equation':repr(snap['coefficients']),
                 'u':float(uv[0]),'v':float(uv[1]),'actual_X':float(xy[0]),'actual_Y':float(xy[1]),
                 'pred_u':float(pred_uv[0]),'pred_v':float(pred_uv[1]),'pred_X':float(pred_xy[0]),'pred_Y':float(pred_xy[1]),
                 'delta_u':float(duv[0]),'delta_v':float(duv[1]),'delta_X':float(dxy[0]),'delta_Y':float(dxy[1]),
                 'error':float(outcome['errors'][0]),'unit':outcome['unit'],'tolerance':tolerance,'status':status}
            self.records.append(row);self._selected_record_id=row['id'];self._refresh_records()
            self.tree.selection_set(str(row['id']));self.tree.see(str(row['id']));self._refresh_overlays()
            self.message.set(f'已加入复检 #{row["id"]}；图上绿色实测点、紫色反算点与记录编号对应。')
            return row
        except (ValueError,TypeError) as exc:self.message.set(str(exc));return None

    def _current_record(self,r):
        table=self.calibration.active_table;fit=table.get('fit')
        return bool(fit and not self.calibration.equation_dirty and r['worktable_id']==table['id'] and
                    r['equation']==repr(fit.coefficients) and r['origin']==table.get('origin'))

    def _visible_records(self):
        return [r for r in self.records if r['space']==self.comparison_space and r['worktable_id']==self.calibration.store.active_id]

    def _refresh_records(self):
        self.tree.delete(*self.tree.get_children())
        image=self.comparison_space=='image'
        fields=('u','v','pred_u','pred_v','error') if image else ('actual_X','actual_Y','pred_X','pred_Y','error')
        for r in self._visible_records():
            current=self._current_record(r)
            self.tree.insert('','end',iid=str(r['id']),values=[str(r['id'])+('' if current else '*')]+[f'{r[k]:.3f}' for k in fields],tags=() if current else ('historical',))
        if self._selected_record_id is not None and str(self._selected_record_id) in self.tree.get_children():self.tree.selection_set(str(self._selected_record_id))
        self._statistics();self._refresh_overlays()

    def _statistics(self):
        rows=[r for r in self._visible_records() if self._current_record(r)]
        if not rows:self.statistics.set(f'当前模式/方程尚无复检记录；全部历史 {len(self.records)} 条。');return
        error=np.array([r['error'] for r in rows]);image=self.comparison_space=='image'
        dx=np.mean([r['delta_u' if image else 'delta_X'] for r in rows]);dy=np.mean([r['delta_v' if image else 'delta_Y'] for r in rows])
        a,b=('u','v') if image else ('X','Y');unit='px' if image else '机台单位'
        self.statistics.set(f'当前模式/方程 {len(rows)} 点：RMS {np.sqrt(np.mean(error**2)):.6f}；最大 {error.max():.6f} {unit}\n平均 Δ{a} {dx:.6f}；Δ{b} {dy:.6f}；* 为历史方程记录。')

    def _record_selected(self,event=None):
        selected=self.tree.selection();self._selected_record_id=int(selected[0]) if selected else None
        self._refresh_overlays()

    def _belongs_to_image(self,r):
        if self.gray is None or not self._current_record(r) or tuple(r.get('image_shape') or ())!=tuple(self.gray.shape):return False
        if r.get('image_id'):return r['image_id']==self.image_id
        return bool(r.get('source') and os.path.normcase(os.path.abspath(r['source']))==os.path.normcase(os.path.abspath(self.source)))

    def _refresh_overlays(self,values=None):
        if self.gray is None:return
        shape=self.gray.shape;self.preview.show_axes=self.show_axes.get();self.preview.coordinate_origin=self.calibration.origin_key
        self.preview.coordinate_center=self.frame_center
        if self.calibration.origin_key=='ring_center' and self.frame_center is None:
            self.preview.comparison_pairs=[];self.preview.comparison_current=None;self.preview.focus_point=None
            self.preview.draw_overlays();return
        pairs=[];selected=None
        for r in self._visible_records():
            if not self._belongs_to_image(r):continue
            actual=frame_to_image([[r['u'],r['v']]],shape,r['origin'],r.get('frame_center',self.frame_center))[0]
            pred=frame_to_image([[r['pred_u'],r['pred_v']]],shape,r['origin'],r.get('frame_center',self.frame_center))[0]
            pair={'id':r['id'],'measured':actual,'predicted':pred,'highlight':r['id']==self._selected_record_id}
            pairs.append(pair)
            if pair['highlight']:selected=pair
        uv,xy,pred_uv,_=values if values is not None else self._preview_values()
        measured=frame_to_image([uv],shape,self.calibration.origin_key,self.frame_center)[0] if uv is not None else None
        predicted=frame_to_image([pred_uv],shape,self.calibration.origin_key,self.frame_center)[0] if pred_uv is not None else None
        current={'measured':measured,'predicted':predicted}
        if selected is not None:
            focus=selected['measured']
        else:focus=predicted if self._focus_kind=='predicted' and predicted is not None else measured if measured is not None else predicted
        if measured is not None and predicted is not None and any(
                np.allclose(measured,pair['measured'],atol=1e-5,rtol=0) and
                np.allclose(predicted,pair['predicted'],atol=1e-5,rtol=0) for pair in pairs):current=None
        self.preview.assigned=[];self.preview.comparison_pairs=pairs;self.preview.comparison_current=current
        self.preview.focus_point=focus;self.preview.draw_overlays()

    def delete_records(self):
        selected={int(i) for i in self.tree.selection()};self.records=[r for r in self.records if r['id'] not in selected]
        self._selected_record_id=None;self._refresh_records()

    def clear_records(self):self.records=[];self._selected_record_id=None;self._refresh_records()

    def export_records(self,path=None):
        rows=self._visible_records()
        if not rows:self.message.set('当前工作台/模式尚无可导出记录。');return
        if path is None:path=filedialog.asksaveasfilename(parent=self,defaultextension='.csv',filetypes=[('CSV','*.csv')],initialfile='verification_'+self.comparison_space+'.csv')
        if not path:return
        image=self.comparison_space=='image'
        headers=['序号','工作台','图片','复检模式']+(['实测u/px','实测v/px','机台实测X','机台实测Y','反算u/px','反算v/px','Δu/px','Δv/px','二维偏差/px'] if image else ['图像实测u/px','图像实测v/px','实测X','实测Y','推算X','推算Y','ΔX','ΔY','二维偏差/机台单位'])+['允许偏差','判定','图像原点','方程']
        keys=('u','v','actual_X','actual_Y','pred_u','pred_v','delta_u','delta_v','error') if image else ('u','v','actual_X','actual_Y','pred_X','pred_Y','delta_X','delta_Y','error')
        try:
            with open(path,'w',newline='',encoding='utf-8-sig') as f:
                writer=csv.writer(f);writer.writerow(headers)
                for r in rows:writer.writerow([r['id'],r['worktable'],r['source'],'图像－机台' if image else '机台－图像']+[f'{r[k]:.3f}' for k in keys]+['' if r['tolerance'] is None else f'{r["tolerance"]:.3f}',r['status'],r['origin'],r['equation']])
        except (OSError,ValueError) as exc:self.message.set('复检记录未能保存：'+str(exc));return
        self.message.set('已导出当前工作台/模式记录：'+str(path))

    def refresh_tables(self):
        table=self.calibration.active_table;fit=table.get('fit')
        self.selector.configure(values=[t['name'] for t in self.calibration.store.tables]);self.name.set(table['name'])
        self.preview.coordinate_origin=table.get('origin','center')
        key=(table['id'],table.get('origin'),tuple(table.get('image_shape') or ()),tuple(np.asarray(fit.coefficients).ravel()) if fit else None,self.calibration.equation_dirty)
        if key!=self._snapshot_key:
            self._snapshot_key=key;self._job+=1;self._busy=False;self._selected_record_id=None
            self._updating_inputs=True
            for v in self.uv+self.xy+self.predicted:v.set('')
            self._updating_inputs=False;self.result.set('');self._controls()
            self.preview.selected=None
        if fit and not self.calibration.equation_dirty:
            origin=next(k for k,v in ORIGINS.items() if v==table.get('origin','top_left'))
            self.equation.set(fit.equations(7)+'\n图像原点：'+origin+'；机台单位随 X、Y。');self.equation_label.configure(foreground='#69E58C')
        else:self.equation.set('该工作台尚未拟合或方程待应用，请先在标定页完成。');self.equation_label.configure(foreground='#FFD15C')
        self._refresh_records();self._refresh_current()
        if table.get('origin')=='ring_center' and self.gray is not None and self._image_center is None and not self._busy:
            gray=self.gray;self._start_work('origin_fitted',lambda:_fit_frame_center(gray))

    def _select_table(self,event=None):
        if self.name.get()!=self.calibration.active_table['name']:
            self.calibration.name.set(self.name.get());self.calibration._select_table()
        self.refresh_tables()

    def _mode_key(self):return 'centroid' if self.mode.get().startswith('质心') else 'ring'

    def _position_clicked(self,raw):
        if self.gray is None:return
        try:uv=image_to_frame([raw],self.gray.shape,self.calibration.origin_key,self.frame_center)[0]
        except ValueError:self.message.set('亮圆环圆心尚未拟合，请稍候。');return
        self.message.set(f'点击位置：u {uv[0]:.3f}，v {uv[1]:.3f} px')

    def _pick(self,raw):
        if self.gray is None or self._busy:return
        if self.pick_mode.get()=='直接像素点':self.preview.selected=None;self._accept_point(raw);return
        mark=min(self.markers,key=lambda m:np.linalg.norm(np.array(raw)-m.center)) if self.markers else None
        if mark is not None and np.linalg.norm(np.array(raw)-mark.center)<=max(8/max(self.preview.scale,.01),4):
            self.preview.selected=mark;self._accept_point(mark.center);return
        gray=self.gray;mode=self._mode_key();self._start_work('picked',lambda:extract_marker(gray,raw,mode))

    def _accept_point(self,raw):
        try:uv=image_to_frame([raw],self.gray.shape,self.calibration.origin_key,self.frame_center)[0]
        except ValueError as exc:self.message.set(str(exc));return
        self._updating_inputs=True
        try:self._put(self.uv,uv)
        finally:self._updating_inputs=False
        self._selected_record_id=None;self._focus_kind='measured';self.result.set('')
        self._refresh_current()

    def _mode_changed(self,event=None):
        self._job+=1;self._busy=False;self.markers=[];self.preview.markers=[];self.preview.selected=None
        self._controls();self._refresh_overlays();self.message.set('检测方式已切换，请重新检测或点选特征。')

    def choose_image(self):
        path=filedialog.askopenfilename(parent=self,filetypes=[('图像','*.bmp *.png *.jpg *.jpeg *.tif *.tiff')])
        if path:self.load_path(path)

    def load_path(self,path):
        def load():
            gray=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_GRAYSCALE)
            if gray is None:raise ValueError('无法读取复检图片')
            return gray,str(path)
        self._start_work('loaded',load)

    def use_current_image(self):
        result=self.image_provider() if self.image_provider else None
        if result and result[0] is not None:self.set_image(result[0],str(result[1] or '检测主图'))
        else:self.message.set('请先选择检测图片或直接选择复检图片。')

    def set_image(self,gray,source='当前图片'):
        self._image_center=None;self.preview.coordinate_center=None
        self.gray=_gray(gray);self.source=source;self.image_id=hashlib.blake2b(np.ascontiguousarray(self.gray).view(np.uint8),digest_size=12).hexdigest()
        self._job+=1;self._busy=False;self.markers=[];self._selected_record_id=None
        self.preview.assigned=[];self.preview.coordinate_origin=self.calibration.origin_key;self.preview.set_image(self.gray)
        self.image_info.set(f'{Path(source).name} · {self.gray.shape[1]} × {self.gray.shape[0]}')
        self._updating_inputs=True
        for v in self.uv+self.xy+self.predicted:v.set('')
        self._updating_inputs=False;self.result.set('');self.message.set('分别提供图像实测点与机台实测坐标，按当前模式比较。')
        self._controls();self._info();self._refresh_current()
        if self.calibration.origin_key=='ring_center':
            gray=self.gray;self._start_work('origin_fitted',lambda:_fit_frame_center(gray))

    def auto_detect(self):
        if self.gray is None:return
        gray=self.gray;mode=self._mode_key();self._start_work('detected',lambda:detect_markers(gray,mode))

    def _start_work(self,kind,callback):
        if self._busy:return
        self._job+=1;job=self._job;table=self.calibration.store.active_id;self._busy=True;self._controls();self.message.set('正在处理复检图像…')
        def work():
            try:self.events.put((job,table,kind,callback()))
            except Exception as exc:self.events.put((job,table,'error',str(exc)))
        threading.Thread(target=work,daemon=True,name='ols-verify').start()

    def _poll(self):
        if self._closed:return
        while True:
            try:job,table,kind,value=self.events.get_nowait()
            except queue.Empty:break
            if job!=self._job or table!=self.calibration.store.active_id:continue
            self._busy=False
            if kind=='loaded':self.set_image(*value)
            elif kind=='origin_fitted':
                self._image_center=list(value);self.preview.coordinate_center=self.frame_center
                self._refresh_current();self.message.set(f'复检圆心已拟合：({value[0]:.3f}, {value[1]:.3f})。')
            elif kind=='detected':
                self.markers=value;self.preview.markers=value
                if len(value)==1:self.preview.selected=value[0];self._accept_point(value[0].center)
                self._refresh_overlays();self.message.set(f'找到 {len(value)} 个候选；绿色为图像实测，紫色为机台反算。')
            elif kind=='picked':self.preview.selected=value;self._accept_point(value.center)
            else:self.message.set(str(value))
            self._controls()
        self._after=self.after(70,self._poll)

    def _controls(self):
        self.open_button.configure(state='disabled' if self._busy else 'normal')
        self.detect_button.configure(state='normal' if self.gray is not None and not self._busy else 'disabled')
        self.mode_combo.configure(state='disabled' if self._busy else 'readonly')

    def _info(self):
        if self.on_preview_info:self.on_preview_info(self.image_info.get())

    def show_preview(self):self.preview.pack(fill='both',expand=True);self._info();self._refresh_current()
    def hide_preview(self):self.preview.pack_forget()

    def save_preferences(self):
        """No source image, measured coordinates, predictions or record history."""
        return {'mode':self.mode.get(),'pick_mode':self.pick_mode.get(),'check_mode':self.check_mode.get(),
                'show_axes':self.show_axes.get(),'tolerances':{k:v.get() for k,v in self.tolerances.items()}}

    def restore_preferences(self,state):
        """Read the same allowlist from new and legacy configuration files."""
        if not isinstance(state,dict):return
        self._updating_inputs=True
        try:
            for name,allowed in [('mode',('质心（戳破薄膜）','圆环（黑色圆环内孔）')),('pick_mode',('候选 / 局部精提取','直接像素点'))]:
                if state.get(name) in allowed:getattr(self,name).set(state[name])
            mode=state.get('check_mode')
            if mode in CHECK_MODES:self.check_mode.set(mode)
            if isinstance(state.get('show_axes'),bool):self.show_axes.set(state['show_axes'])
            values=state.get('tolerances',{})
            if isinstance(values,dict):
                for k,v in values.items():
                    if k in self.tolerances and isinstance(v,str) and len(v)<100:self.tolerances[k].set(v)
            if not values and isinstance(state.get('tolerance'),str) and len(state['tolerance'])<100:
                self.tolerances['machine'].set(state['tolerance'])
        finally:self._updating_inputs=False
        self._comparison_mode_changed();self._refresh_current()

    def _destroyed(self,event):
        if event.widget is self:
            self._closed=True;self._job+=1
            for after in (self._after,self._input_after):
                if after:
                    try:self.after_cancel(after)
                    except tk.TclError:pass


def configure_combobox_selection(root, style):
    """Keep readonly dropdown labels dark after selecting or leaving a field."""
    style.configure('TCombobox',selectbackground='#242424',selectforeground='#EDEDED')
    style.map('TCombobox',
        selectbackground=[('disabled','#202020'),('!disabled','#242424')],
        selectforeground=[('disabled','#777777'),('!disabled','#EDEDED')])
    root.option_add('*TCombobox*exportSelection',False)
    if getattr(root,'_readonly_combobox_selection_binding',False):return
    root._readonly_combobox_selection_binding=True
    def clear(event):
        widget=event.widget
        def finish():
            try:
                if widget.winfo_exists() and widget.instate(('readonly',)):
                    widget.configure(exportselection=False)
                    widget.selection_clear()
            except (tk.TclError,AttributeError):pass
        finish()
        root.after_idle(finish)
    for sequence in ('<<ComboboxSelected>>','<FocusIn>','<FocusOut>'):
        root.bind_class('TCombobox',sequence,clear,add='+')


def standalone_theme(root):
    import tkinter.font as tkfont
    available = set(tkfont.families(root))
    font = next((name for name in ('Microsoft YaHei UI', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei') if name in available), 'TkDefaultFont')
    style = ttk.Style(root); style.theme_use('clam')
    style.configure('.', background='#181818', foreground='#EDEDED', font=(font, 10))
    style.configure('TButton', background='#303030', foreground='#EDEDED', padding=6)
    style.configure('Treeview', background='#151515', fieldbackground='#151515', foreground='#EDEDED', rowheight=28)
    style.configure('Treeview.Heading', background='#303030', foreground='#EDEDED')
    style.map('Treeview', background=[('selected', '#275535')])
    for name in ('TEntry', 'TCombobox'):
        style.configure(name, fieldbackground='#242424', foreground='#EDEDED', insertcolor='#EDEDED')
        style.map(name, fieldbackground=[('readonly', '#242424')], foreground=[('readonly', '#EDEDED')])
    root.option_add('*TCombobox*Listbox.background', '#242424')
    root.option_add('*TCombobox*Listbox.foreground', '#EDEDED')
    configure_combobox_selection(root,style)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?'); parser.add_argument('--state', type=Path)
    args = parser.parse_args(); cv2.setNumThreads(2)
    root = tk.Tk(); root.title('V4.3.0 坐标回归'); root.geometry('1420x900'); root.minsize(1060, 700)
    standalone_theme(root)
    paned = ttk.Panedwindow(root, orient='horizontal'); paned.pack(fill='both', expand=True)
    left = ttk.Frame(paned, padding=10); right = ttk.Frame(paned, width=455, padding=8); right.pack_propagate(False)
    paned.add(left, weight=5); paned.add(right, weight=0)
    canvas = tk.Canvas(right, highlightthickness=0, bg='#181818')
    scroll = ttk.Scrollbar(right, orient='vertical', command=canvas.yview)
    scroll.pack(side='right', fill='y'); canvas.pack(fill='both', expand=True); canvas.configure(yscrollcommand=scroll.set)
    tab_container=ttk.Frame(canvas)
    notebook=WorktableNotebook(tab_container);notebook.pack(fill='both',expand=True)
    calibration_page=ttk.Frame(notebook);verification_page=ttk.Frame(notebook)
    notebook.add(calibration_page,text='标定工作台');notebook.add(verification_page,text='复检')
    panel=CalibrationPanel(calibration_page,preview_parent=left,state_path=args.state)
    panel.pack(fill='both',expand=True)
    verification=VerificationPanel(verification_page,panel,left)
    verification.pack(fill='both',expand=True)
    panel.on_change=verification.refresh_tables
    window=canvas.create_window((0,0),window=tab_container,anchor='nw')
    tab_container.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')),add='+')
    canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width))
    def switch(event=None):
        panel.hide_preview();verification.hide_preview();verification.refresh_tables()
        (verification if notebook.select()==str(verification_page) else panel).show_preview()
        canvas.yview_moveto(0)
    notebook.bind('<<NotebookTabChanged>>',switch,add='+');panel.show_preview()
    def wheel(event):
        widget=event.widget
        while widget is not None and widget is not right:widget=getattr(widget,'master',None)
        if widget is right:
            canvas.yview_scroll(-3 if getattr(event,'num',None)==4 or getattr(event,'delta',0)>0 else 3,'units')
            return 'break'
    root.bind_all('<MouseWheel>',wheel,add='+');root.bind_all('<Button-4>',wheel,add='+');root.bind_all('<Button-5>',wheel,add='+')
    def place_sash():
        if paned.winfo_width() < 900: root.after(50, place_sash); return
        paned.sashpos(0, max(400, paned.winfo_width()-455))
    root.after_idle(place_sash)
    if args.image: root.after(150, lambda: panel.load_path(args.image))
    root.mainloop()


if __name__ == '__main__': main()

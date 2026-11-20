#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single main ROI desktop UI with integrated TCP; no warmup or load-time scan.

Place beside roi_polar_detector_v4_3_0.py and OLS_v4_3_0.py. Run: python roi_polar_ui_v4_3_0.py
Dependencies: numpy opencv-python-headless scipy shapely pillow
Default is white; black/white both output one main inner closed loop.
Worktables are persisted separately in the user profile. OLS_v4_3_0.py provides calibration.
Review uses three steps. Saving supports selectable image/Cartesian/polar
outputs, CSV or tab-separated TXT. Cartesian output follows one of three
worktable origins; polar output uses 720 rays from the bright-ring center.
The labelled directions default to image right / counterclockwise, independently
of regression distance units. Machine-X angle directions are also selectable.
Display settings apply live. Optional bright-ring fitting is cached per image.
UI preferences are saved to ui_settings.json on normal exit and restored on startup.
Only images and coordinate files are saved; TCP returns a selected file name.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import shutil
import dataclasses
import math
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from OLS_v4_3_0 import CalibrationPanel, VerificationPanel, WorktableNotebook, TickCheckbutton, FoldSection, configure_combobox_selection

from roi_polar_detector_v4_3_0 import (DEFAULT_COLORS, DetectorConfig, ExportConfig, load_gray, render_image,
                                  export_analysis, make_export_stem, select_review_arc, preview_review, apply_review, confirm_review,
                                  coordinate_frame, coordinate_path, fit_bright_ring, ensure_bright_ring)
from roi_polar_detector_v4_3_0 import polar_ray_table, polar_ray_at_angle, ensure_wafer_outline, machine_affine
# Socket/task services are embedded here. OLS_v4_3_0.py owns the integrated
# calibration page. Each request snapshots the current controls and origin.
import copy
import csv
import datetime as dt
import io
import ntpath
import os
import re
import socket
import time
from concurrent.futures import Future
from roi_polar_detector_v4_3_0 import analyze

BG="#101010"; PANEL="#181818"; ENTRY="#242424"; FG="#EDEDED"; MUTED="#AAAAAA"; BLUE="#76B4FF"
FILE_TYPES=[("图像文件","*.bmp *.png *.jpg *.jpeg *.tif *.tiff"),("所有文件","*.*")]
POLARITIES={"黑色 ROI":"black","白色 ROI":"white"}
REGIONS={"自动主体闭环":"auto","全图搜索":"full","手动圆形区域":"circle","手动 ROI":"polygon"}
POINT_SELECTION={'1 像素':(1.,None),'自适应机台':(1.,.5)}
POLAR_REFERENCES={'图片水平向右':'image','机台 X 正向':'machine'}

ADVANCED_HELP={
    'path':'按当前笔宽识别高密度转向和反复折返，再约束连续选路。\n\n0 关闭，1 为默认；调大更偏向连贯路线，但可能弱化小细节。只改变异常局部，不把整圈拟合成圆。结果整圈使用主体闭环颜色，可按需人工复核走势。',
    'main_fraction':'作用：把候选面积与当前最大的、有笔迹证据的中心闭环比较，排除小支环和局部缺陷。\n\n默认 60%，通过主体筛选后，同等笔迹证据优先选内侧闭环。\n调大：更偏向大闭环，可能排除内缩很深的主体。\n调小：可接受更小的主体，也更容易接受内部小环。\n\n需要只取局部小环时，可缩小手动圆形搜索区域。',
    'gap':'作用：修补笔迹断口。自动先补小断口；主体仍不闭合时，再沿邻近笔迹走势尝试有限长度补线，并标记插值点。\n\n调大：可连接更大缺口，也可能把相邻笔迹连接起来。\n调小：减少串线，但可能漏掉有断口的 ROI。\n\n建议：先用自动；0 关闭修补。自动补线的单段上限为短边 5.5%；手动模式的单段上限为半径的 2 倍。实际仅在笔迹证据改善时接受，补线结果需复核。',
    'min_ratio':'作用：按搜索区域面积的一定百分比，过滤过小的闭环杂点。\n\n调大：排除更多小闭环，也可能漏掉较小的真实 ROI。\n调小：可保留更小的 ROI，但可能增加杂点。\n\n此项仅在最小闭环面积 / px² 为自动时生效。',
    'min_px':'作用：直接指定闭环所需的最小包围面积，单位是像素平方。\n\n调大：过滤小闭环。\n调小：允许更小的 ROI。\n\n自动：根据搜索区域面积和上一项百分比计算。\n手动：使用本项数值，覆盖百分比计算。',
    'width':'作用：限制笔迹中位宽度与闭环面积平方根的比值，过滤过宽的材料色带。\n\n例如 15 表示该比值最多为 15%。\n调大：允许更粗的笔迹，也更容易接受宽色带。\n调小：筛选更严格，但可能漏掉粗笔迹。',
    'step':'控制检测与人工复核的内部基础采样间距，默认 1.0 原图像素。\n\n保存选项中的“像素选点”会单独精简最终坐标，建议保留这里的默认值。调大内部间距也会降低细小走势的表达，不能代替带最大偏差约束的导出选点。',
    'illumination':'作用：抑制大范围、缓慢变化的明暗背景。\n\n开启：在缩略图上估计缓变背景，适合光照不均；会改变用于筛选的灰度。\n关闭：保留当前图的灰度关系，适合光照较均匀的图片。\n\n开启后，自动灰度范围会重新学习并注明亮度校正后；手动范围需按校正后的图像调整。',
    'refine':'作用：沿笔迹法线分析灰度截面，把曲线调整到笔迹中间。\n\n开启：通常有助于减少曲线偏向笔迹某一侧，建议保留。\n关闭：省去中线精修，结果可能更靠近边缘。\n\n只在选中的主体曲线上采样，不扫描整幅原图；不改变 CSV 坐标原点。',
    'contrast':'作用：要求笔迹与附近背景有足够的局部灰度差，用于排除纹理和材料色带。\n\n调大：减少弱纹理误检，也可能漏掉浅笔迹。\n调小：能保留更浅的笔迹，但可能增加误检。\n\n建议：先用自动，按当前图的背景噪声和灰度变化估计。'
}


IMAGE_SUFFIXES={".bmp",".png",".jpg",".jpeg",".tif",".tiff"}


def socket_ng(reason):
    reason=str(reason).replace("\r"," ").replace("\n"," ").replace(",","；").replace("，","；")
    return "NG,"+(reason.strip() or "未知错误")


def parse_socket_request(message):
    raw=str(message).replace("\x00","").strip()
    if not raw or len(raw)>32768:
        raise ValueError("指令为空或超过 32768 字符")
    # Outside quotes only, normalize the two legacy delimiter alternatives.
    normalized=[];quoted=False
    for char in raw:
        if char=='"':quoted=not quoted
        normalized.append(',' if not quoted and char in '，|' else char)
    if quoted:
        raise ValueError("指令引号未闭合")
    rows=list(csv.reader(io.StringIO(''.join(normalized)),skipinitialspace=True,strict=True))
    if len(rows)!=1:
        raise ValueError("一条指令不能包含多行")
    fields=[field.strip().strip("'") for field in rows[0]]
    if len(fields)==5:
        plate,model,source_dir,source_name,save_dir=fields;reserved=""
    elif len(fields)==6:
        plate,model,reserved,source_dir,source_name,save_dir=fields
    else:
        raise ValueError("应为5或6字段：片号,型号,[预留编号],原路径,图片或文件夹名,保存目录")
    for title,value in (("片号",plate),("型号",model),("原路径",source_dir),("名称",source_name),("保存目录",save_dir)):
        if not value:raise ValueError(title+"为空")
    if len(plate)>128 or len(model)>128 or len(reserved)>128:
        raise ValueError("片号、型号或预留编号过长")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]',plate) or plate in (".",".."):
        raise ValueError("片号包含无效文件名字符")
    pathmod=ntpath if os.name=="nt" or re.match(r"^[A-Za-z]:[\\/]",source_dir) or source_dir.startswith("\\\\") else os.path
    original=pathmod.normpath(source_dir);name=pathmod.normpath(source_name)
    if pathmod.isabs(name):source=name
    elif pathmod.normcase(pathmod.basename(original))==pathmod.normcase(name):source=original
    else:source=pathmod.join(original,name)
    return {"plate":plate,"model":model,"reserved":reserved,"source":source,"save_dir":pathmod.normpath(save_dir)}




def _reply_row(fields):
    stream=io.StringIO();csv.writer(stream,lineterminator="").writerow(fields)
    return stream.getvalue()


class TaskRunner:
    """One computation at a time. Worker threads never read Tk variables."""
    def __init__(self,event_sink=None):
        self.events=event_sink if event_sink is not None else queue.Queue()
        self._lock=threading.Lock()
        self._busy=False
        self._review_locked=False
        self._closing=False
        self._defaults=DetectorConfig()
        self._default_error=""
        self._style={}
        self._export_config=ExportConfig()
        cv2.setNumThreads(2)

    @property
    def ready(self):
        return not self._closing

    @property
    def busy(self):
        with self._lock:return self._busy

    def set_review_lock(self,locked):
        with self._lock:self._review_locked=bool(locked)

    def set_defaults(self,config,style=None,export_config=None):
        config.validate()
        export_config=(export_config or ExportConfig()).validate()
        with self._lock:
            self._defaults=config;self._style=copy.deepcopy(style or {});self._export_config=export_config;self._default_error=""

    def set_default_error(self,message):
        with self._lock:self._default_error=str(message)

    def submit(self,request,manual=False,config=None,style=None,save=True,gray=None,export_config=None):
        future=Future()
        with self._lock:
            if self._review_locked:
                future.set_exception(RuntimeError("正在人工复核，请稍后重试"));return future
            if self._closing or self._busy:
                future.set_exception(RuntimeError("软件忙或正在关闭"));return future
            if self._default_error and config is None:
                future.set_exception(ValueError("当前界面参数无效："+self._default_error));return future
            selected=config or self._defaults
            selected_style=copy.deepcopy(self._style if style is None else style)
            selected_export=export_config or self._export_config
            origin="current_parameters"
            self._busy=True
        request=dict(request);request["submission_source"]="manual" if manual else "socket"
        threading.Thread(target=self._work,args=(request,future,selected,selected_style,origin,save,gray,selected_export),daemon=True,name="roi-detector").start()
        return future

    def _work(self,request,future,config,style,origin,save,preloaded_gray=None,export_config=None):
        started=time.perf_counter()
        self.events.put(("task_started",request,origin))
        try:
            export_config=(export_config or ExportConfig()).validate(require_output=save)
            source=Path(request["source"])
            if source.is_file():
                files=[source] if source.suffix.lower() in IMAGE_SUFFIXES else []
            elif source.is_dir():
                files=sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),key=lambda p:p.name.casefold())
            else:raise ValueError("图片或文件夹不存在："+str(source))
            if not files:raise ValueError("没有支持的图片")
            if save:Path(request["save_dir"]).mkdir(parents=True,exist_ok=True)
            replies=[];last_analysis=None;results=[]
            for index,path in enumerate(files):
                if self._closing:raise RuntimeError("任务已取消")
                self.events.put(("item_started",index,len(files),str(path)))
                detect_started=time.perf_counter()
                try:
                    gray=preloaded_gray if len(files)==1 and preloaded_gray is not None else load_gray(path)
                    self.events.put(("item_loaded",str(path),gray))
                    detect_started=time.perf_counter()
                    analysis=analyze(gray,config)
                    analysis.metadata["source"]=str(path)
                    analysis.metadata.update(task={"plate":request.get("plate","manual"),"model":request.get("model",""),
                                                   "reserved":request.get("reserved",""),"parameter_source":origin,
                                                   "submission_source":request.get("submission_source","socket"),"save_dir":request["save_dir"]})
                    now=dt.datetime.now()
                    analysis.metadata['detected_at']=now.isoformat(timespec='microseconds')
                    ensure_bright_ring(analysis,style)
                    stem=make_export_stem(path,now)
                    output=export_analysis(analysis,request["save_dir"],stem,style,export_config) if save else None
                    if analysis.status=="NG":
                        response=socket_ng("；".join(analysis.metadata["warnings"]))
                    else:
                        response=_reply_row([request.get("plate","manual"),now.strftime("%Y-%m-%d %H:%M:%S"),
                                             analysis.status,len(analysis.rois),request["save_dir"],output["primary"] if output else ""])
                    replies.append(response);last_analysis=analysis
                    results.append({"source":str(path),"status":analysis.status,"output":output})
                    self.events.put(("item_result",analysis,output,index+1,len(files),origin))
                except Exception as exc:
                    replies.append(socket_ng(path.name+"："+str(exc)))
                    results.append({"source":str(path),"status":"NG","error":str(exc)})
                    self.events.put(("item_error",str(exc),index+1,len(files),time.perf_counter()-detect_started))
            payload={"reply":"\n".join(replies),"seconds":time.perf_counter()-started,
                     "results":results,"analysis":last_analysis,"parameter_source":origin}
            future.set_result(payload)
        except Exception as exc:
            future.set_exception(exc)
            self.events.put(("task_error",str(exc)))
        finally:
            with self._lock:self._busy=False
            self.events.put(("task_finished",time.perf_counter()-started))

    def close(self):
        with self._lock:self._closing=True


class SocketServer:
    """Newline framing; 50 ms idle framing retained for legacy clients.

    The server does not depend on GUI polling. A response is sent only after
    selected coordinate files and optional image have been written.
    """
    def __init__(self,runner):
        self.runner=runner;self._server=None
        self._stop=threading.Event();self._connections=set();self._lock=threading.Lock()
        self.encoding="gbk";self.address=None

    @property
    def running(self):return self._server is not None

    def start(self,host="127.0.0.1",port=5000,encoding="gbk"):
        if self.running:return self.address
        if encoding.lower() not in ("gbk","utf-8"):raise ValueError("编码必须为 GBK 或 UTF-8")
        if not 0<=int(port)<=65535:raise ValueError("端口范围应为 0～65535")
        server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind((host,int(port)));server.listen(8);server.settimeout(.2)
        except Exception:
            server.close();raise
        self.encoding=encoding.lower();self.address=server.getsockname()
        self._stop=threading.Event();self._server=server
        threading.Thread(target=self._accept,args=(server,self._stop),daemon=True,name="roi-socket-accept").start()
        return self.address

    def _accept(self,server,stop):
        while not stop.is_set():
            try:conn,_=server.accept()
            except socket.timeout:continue
            except OSError:break
            conn.settimeout(.05)
            with self._lock:self._connections.add(conn)
            threading.Thread(target=self._client,args=(conn,stop,self.encoding),daemon=True,name="roi-socket-client").start()

    @staticmethod
    def _send(conn,message,encoding):
        # Blank line ends one response task, including multi-image folders.
        conn.sendall((message+"\n\n").encode(encoding,errors="replace"))

    def _command(self,conn,data,stop,encoding):
        if not data.strip():return
        try:
            request=parse_socket_request(data.decode(encoding,errors="strict"))
            future=self.runner.submit(request)
            while not future.done():
                if stop.wait(.05):return
            payload=future.result()
            self._send(conn,payload["reply"],encoding)
            self.runner.events.put(("socket_reply",payload["reply"],payload["seconds"]))
        except Exception as exc:
            try:self._send(conn,socket_ng(exc),encoding)
            except OSError:pass

    def _client(self,conn,stop,encoding):
        buffer=bytearray()
        try:
            while not stop.is_set():
                try:data=conn.recv(16384)
                except socket.timeout:
                    if buffer:self._command(conn,bytes(buffer),stop,encoding);buffer.clear()
                    continue
                if not data:
                    if buffer:self._command(conn,bytes(buffer),stop,encoding)
                    break
                buffer.extend(data.replace(b"\x00",b"").replace(b"\r",b"\n"))
                if len(buffer)>32768:
                    self._send(conn,socket_ng("指令过长"),encoding);break
                while b"\n" in buffer:
                    position=buffer.index(b"\n");line=bytes(buffer[:position]);del buffer[:position+1]
                    self._command(conn,line,stop,encoding)
        except OSError:pass
        finally:
            with self._lock:self._connections.discard(conn)
            try:conn.close()
            except OSError:pass

    def stop(self):
        self._stop.set();server=self._server;self._server=None
        if server:
            try:server.close()
            except OSError:pass
        with self._lock:connections=list(self._connections)
        for conn in connections:
            try:conn.shutdown(socket.SHUT_RDWR);conn.close()
            except OSError:pass


class ParameterHelp(tk.Canvas):
    """Vector circled exclamation with a hover or click-open explanation."""
    _active=None

    def __init__(self,parent,title,description,font):
        super().__init__(parent,width=20,height=20,bg=PANEL,highlightthickness=0,takefocus=1,cursor='hand2')
        self.title=title;self.description=description;self.font=font
        self.tip=None;self.pinned=False;self._show_after=None
        self.circle=self.create_oval(2,2,18,18,outline=BLUE,width=1.5)
        self.create_text(10,10,text='!',fill=BLUE,font=(font,10,'bold'))
        self.bind('<Enter>',self._enter);self.bind('<Leave>',self._leave)
        self.bind('<Button-1>',self._toggle);self.bind('<Return>',self._toggle);self.bind('<space>',self._toggle)
        self.bind('<Escape>',self.hide);self.bind('<Unmap>',self.hide);self.bind('<Destroy>',self.hide)

    @classmethod
    def close_active(cls):
        if cls._active is not None:cls._active.hide()

    def _cancel_timer(self):
        if self._show_after is not None:
            self.after_cancel(self._show_after);self._show_after=None

    def _enter(self,event=None):
        self.itemconfigure(self.circle,fill='#24323F')
        self._cancel_timer()
        if self.tip is None:self._show_after=self.after(350,self._hover)

    def _hover(self):
        self._show_after=None;self.show(False)

    def _leave(self,event=None):
        self._cancel_timer();self.itemconfigure(self.circle,fill=PANEL)
        if not self.pinned:self.hide()

    def _toggle(self,event=None):
        self.focus_set()
        if self.tip is not None and self.pinned:self.hide()
        else:self.show(True)
        return 'break'

    def show(self,pinned=False):
        self._cancel_timer()
        if not self.winfo_viewable():return
        if ParameterHelp._active is not self:ParameterHelp.close_active()
        self.pinned=pinned
        if self.tip is not None:return
        tip=tk.Toplevel(self);self.tip=tip;ParameterHelp._active=self
        tip.withdraw();tip.overrideredirect(True);tip.transient(self.winfo_toplevel())
        tip.configure(bg=BLUE)
        body=tk.Frame(tip,bg=ENTRY,padx=12,pady=10);body.pack(padx=1,pady=1)
        header=tk.Frame(body,bg=ENTRY);header.pack(fill='x')
        tk.Label(header,text=self.title,bg=ENTRY,fg=FG,font=(self.font,10,'bold'),anchor='w',wraplength=280).pack(side='left',fill='x',expand=True)
        tk.Button(header,text='×',command=self.hide,bg=ENTRY,fg=MUTED,activebackground=PANEL,activeforeground=FG,relief='flat',bd=0,padx=5,pady=0).pack(side='right',padx=(8,0))
        tk.Label(body,text=self.description,bg=ENTRY,fg=FG,font=(self.font,10),wraplength=320,justify='left',anchor='w').pack(fill='x',pady=(8,0))
        tip.bind('<Escape>',self.hide);tip.update_idletasks()
        width,height=tip.winfo_reqwidth(),tip.winfo_reqheight()
        host=self.winfo_toplevel()
        left=max(4,host.winfo_rootx()+8)
        right=min(self.winfo_screenwidth()-4,host.winfo_rootx()+host.winfo_width()-8)
        top=max(4,host.winfo_rooty()+8)
        bottom=min(self.winfo_screenheight()-4,host.winfo_rooty()+host.winfo_height()-8)
        x=self.winfo_rootx()+24;y=self.winfo_rooty()+20
        if x+width>right:x=self.winfo_rootx()-width-6
        x=max(left,min(x,right-width));y=max(top,min(y,bottom-height))
        tip.geometry(f'+{x}+{y}');tip.deiconify();tip.lift()

    def hide(self,event=None):
        self._cancel_timer();self.pinned=False
        if ParameterHelp._active is self:ParameterHelp._active=None
        tip=self.tip;self.tip=None
        if tip is not None:tip.destroy()
        return 'break'


class ImageView(ttk.Frame):
    """Crop to viewport at native resolution; do not allocate huge zoom images."""
    def __init__(self,parent,font):
        super().__init__(parent)
        self.bgr=None;self.zoom=1.;self.x=0.;self.y=0.;self.dw=0;self.dh=0;self.photo=None;self.pan=None
        self._after=None;self.overview=None
        self.circle=None;self.circle_callback=None;self._circle_armed=False;self._circle_start=None;self.scale=1.0
        self.canvas=tk.Canvas(self,bg="#050505",highlightthickness=0,cursor="fleur")
        self.canvas.pack(fill="both",expand=True)
        self.canvas.create_text(22,22,anchor="nw",text="选择图片后显示 ROI 走势",fill=MUTED,font=(font,13))
        self.info=tk.StringVar(value="滚轮按鼠标缩放 · 左键拖动")
        self.pointer_text=tk.StringVar(value='u —   v — px');self.coordinate_origin_provider=lambda:'center'
        self.coordinate_center_provider=lambda:None
        bar=ttk.Frame(self);bar.pack(fill="x",pady=(7,0))
        ttk.Label(bar,textvariable=self.info).pack(side="left")
        ttk.Button(bar,text="适应窗口",command=self.fit).pack(side="right",before=bar.winfo_children()[0])
        ttk.Label(bar,textvariable=self.pointer_text,foreground=BLUE).pack(side='right',padx=10,before=bar.winfo_children()[0])
        self.canvas.bind("<Configure>",self._resize)
        self.canvas.bind("<MouseWheel>",lambda e:self._zoom(e,1.18 if e.delta>0 else 1/1.18))
        self.canvas.bind("<Button-4>",lambda e:self._zoom(e,1.18))
        self.canvas.bind("<Button-5>",lambda e:self._zoom(e,1/1.18))
        self.canvas.bind("<ButtonPress-1>",self._pointer_down)
        self.canvas.bind("<B1-Motion>",self._pointer_drag)
        self.canvas.bind("<ButtonRelease-1>",self._pointer_up)
        self.canvas.bind('<Motion>',self._pointer_motion)
        self.canvas.bind('<Leave>',lambda e:self.pointer_text.set('u —   v — px'))

    def _pointer_motion(self,event):
        raw=self._image_point(event)
        if raw is None:self.pointer_text.set('u —   v — px');return
        try:origin,sign=coordinate_frame(self.bgr.shape[:2],self.coordinate_origin_provider(),self.coordinate_center_provider())
        except ValueError:self.pointer_text.set('亮圆环圆心待拟合');return
        self.pointer_text.set(f'u {raw[0]-origin[0]:.3f}   v {(raw[1]-origin[1])*sign:.3f} px')

    def set_image(self,bgr,reset=True):
        self.bgr=bgr;self.overview=None
        if reset:self.zoom=1.;self.dw=0
        self.draw(reset=reset)

    def clear(self,text="正在读取图片…"):
        self.bgr=None;self.overview=None;self.photo=None;self.dw=0;self.dh=0
        self.canvas.delete("all")
        self.canvas.create_text(22,22,anchor="nw",text=text,fill=MUTED)

    def _resize(self,event):
        if self._after:self.after_cancel(self._after)
        self._after=self.after(60,self.draw)

    def fit(self):
        self.zoom=1.;self.draw(reset=True)

    def arm_circle(self,callback):
        if self.bgr is None:return
        self.circle_callback=callback;self._circle_armed=True;self._circle_start=None;self._circle_before=self.circle
        self.canvas.configure(cursor='crosshair')
        self.info.set('按住左键确定圆心，拖动设置半径；松开完成，Esc 取消')
        self.canvas.focus_set();self.canvas.bind('<Escape>',self.cancel_circle)

    def cancel_circle(self,event=None):
        self._circle_armed=False;self._circle_start=None;self.circle=getattr(self,'_circle_before',self.circle);self.canvas.configure(cursor='fleur')
        self.draw(keep=True)

    def set_circle(self,circle):
        self.circle=circle;self._draw_circle_overlay()

    def _draw_circle_overlay(self):
        self.canvas.delete('search_circle')
        if self.circle is None or self.bgr is None:return
        cx,cy,r=self.circle;s=self.scale
        x=self.x+cx*s;y=self.y+cy*s;radius=r*s
        self.canvas.create_oval(x-radius,y-radius,x+radius,y+radius,outline='#FFE164',width=2,dash=(6,4),tags='search_circle')
        self.canvas.create_line(x-5,y,x+5,y,fill='#FFE164',tags='search_circle')
        self.canvas.create_line(x,y-5,x,y+5,fill='#FFE164',tags='search_circle')

    def _image_point(self,event):
        if self.bgr is None:return None
        x=(event.x-self.x)/self.scale;y=(event.y-self.y)/self.scale
        h,w=self.bgr.shape[:2]
        if not (0<=x<w and 0<=y<h):return None
        return x,y

    def _pointer_down(self,event):
        if self._circle_armed:
            point=self._image_point(event)
            if point is not None:self._circle_start=point;self.circle=(*point,0.);self._draw_circle_overlay()
        else:self.pan=(event.x,event.y)

    def _pointer_drag(self,event):
        if self._circle_armed and self._circle_start is not None:
            x=(event.x-self.x)/self.scale;y=(event.y-self.y)/self.scale
            cx,cy=self._circle_start;self.circle=(cx,cy,math.hypot(x-cx,y-cy));self._draw_circle_overlay()
        elif not self._circle_armed:self._drag(event)

    def _pointer_up(self,event):
        self.pan=None
        if self._circle_armed and self._circle_start is not None:
            circle=self.circle;self._circle_armed=False;self._circle_start=None;self.canvas.configure(cursor='fleur')
            if circle[2]>=2 and self.circle_callback:self.circle_callback(circle)
            else:self.circle=getattr(self,'_circle_before',None)
            self.draw(keep=True)

    def _zoom(self,event,factor):
        if self.bgr is None:return "break"
        u=(event.x-self.x)/max(1,self.dw);v=(event.y-self.y)/max(1,self.dh)
        self.zoom=float(np.clip(self.zoom*factor,.2,64.))
        self.draw(anchor=(event.x,event.y,u,v));return "break"

    def _drag(self,event):
        if self.pan is not None and self.bgr is not None:
            oldx,oldy=self.pan;self.pan=(event.x,event.y)
            self.x+=event.x-oldx;self.y+=event.y-oldy;self.draw(keep=True)

    def draw(self,reset=False,anchor=None,keep=False):
        self._after=None
        if self.bgr is None:return
        h,w=self.bgr.shape[:2];cw=max(20,self.canvas.winfo_width());ch=max(20,self.canvas.winfo_height())
        scale=max(.001,min(cw/w,ch/h)*self.zoom);dw=max(1,round(w*scale));dh=max(1,round(h*scale))
        if anchor:
            px,py,u,v=anchor;x=px-u*dw;y=py-v*dh
        elif reset or not self.dw:x=(cw-dw)/2;y=(ch-dh)/2
        elif keep:x=self.x;y=self.y
        else:x=cw/2-(cw/2-self.x)/self.dw*dw;y=ch/2-(ch/2-self.y)/self.dh*dh
        x=(cw-dw)/2 if dw<=cw else min(0,max(cw-dw,x))
        y=(ch-dh)/2 if dh<=ch else min(0,max(ch-dh,y))
        self.x,self.y,self.dw,self.dh=x,y,dw,dh;self.scale=scale
        x0=max(0,math.floor(-x/scale)-2);x1=min(w,math.ceil((cw-x)/scale)+2)
        y0=max(0,math.floor(-y/scale)-2);y1=min(h,math.ceil((ch-y)/scale)+2)
        frame=np.full((ch,cw,3),5,np.uint8)
        if x1>x0 and y1>y0:
            factor=min(1.,2400/max(h,w))
            if scale<factor*.95:
                if self.overview is None:self.overview=cv2.resize(self.bgr,(round(w*factor),round(h*factor)),interpolation=cv2.INTER_AREA)
                sy,sx=self.overview.shape[0]/h,self.overview.shape[1]/w
                a0=max(0,math.floor(x0*sx));a1=min(self.overview.shape[1],math.ceil(x1*sx))
                b0=max(0,math.floor(y0*sy));b1=min(self.overview.shape[0],math.ceil(y1*sy))
                crop=self.overview[b0:b1,a0:a1];xx0,yy0,xx1,yy1=a0/sx,b0/sy,a1/sx,b1/sy
            else:crop=self.bgr[y0:y1,x0:x1];xx0,yy0,xx1,yy1=x0,y0,x1,y1
            tw=max(1,round((xx1-xx0)*scale));th=max(1,round((yy1-yy0)*scale))
            tile=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
            dx=round(x+xx0*scale);dy=round(y+yy0*scale)
            ax=max(0,-dx);ay=max(0,-dy);bx=min(tw,cw-dx);by=min(th,ch-dy)
            if bx>ax and by>ay:frame[dy+ay:dy+by,dx+ax:dx+bx]=tile[ay:by,ax:bx]
        self.photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all");self.canvas.create_image(0,0,anchor="nw",image=self.photo)
        self._draw_circle_overlay()
        self.info.set(f"滚轮缩放 · 左键拖动 · {self.zoom*100:.0f}%（相对适应窗口）")


class InteractiveImageView(ImageView):
    """Image-space gestures survive zoom/pan; they are guidance, not tool paths."""
    def __init__(self,parent,font):
        self.mode='pan';self.interaction_callback=None;self.gesture=[];self.clicks=[]
        self.search_polygon=None;self.tools={}
        super().__init__(parent,font)
        self.canvas.bind('<Escape>',self.cancel_interaction)
        self.canvas.bind('<Return>',self._finish_polygon)
        self.canvas.bind('<Double-Button-1>',self._finish_polygon)
        self.canvas.bind('<ButtonPress-3>',self._right_pan)
        self.canvas.bind('<B3-Motion>',self._right_drag)
        self.canvas.bind('<ButtonRelease-3>',lambda e:setattr(self,'pan',None))

    def arm(self,mode,callback):
        if self.bgr is None:return
        if self._circle_armed:self.cancel_circle()
        self.mode=mode;self.interaction_callback=callback;self.gesture=[];self.clicks=[]
        self.canvas.configure(cursor='crosshair');self.canvas.focus_set()
        self.canvas.bind('<Escape>',self.cancel_interaction)
        self._draw_tools()

    def cancel_interaction(self,event=None):
        self.mode='pan';self.interaction_callback=None;self.gesture=[];self.clicks=[];self.pan=None
        self.canvas.configure(cursor='fleur');self.info.set('滚轮缩放 · 左键拖动')
        if self._circle_armed:super().cancel_circle()
        self._draw_tools()
        return 'break'

    def arm_circle(self,callback):
        self.cancel_interaction();super().arm_circle(callback)

    def set_tools(self,**tools):
        self.tools={key:value for key,value in tools.items() if value is not None}
        self._draw_tools()

    def _canvas_points(self,points):
        p=np.asarray(points,float).reshape(-1,2)
        return np.c_[self.x+p[:,0]*self.scale,self.y+p[:,1]*self.scale].ravel().tolist()

    def _draw_tools(self):
        if not hasattr(self,'canvas'):return
        self.canvas.delete('tools')
        if self.bgr is None:return
        if self.search_polygon is not None:
            p=np.asarray(self.search_polygon,float)
            if len(p)>2:self.canvas.create_line(*self._canvas_points(np.r_[p,p[:1]]),fill='#FFE164',width=2,dash=(6,4),tags='tools')
        for key,color,width in [('selection','#FFAA32',4),('guide','#6DB9FF',2),('preview','#26CFFF',4),('zone','#FFAA32',2)]:
            p=self.tools.get(key)
            if p is not None and len(p)>1:
                if key=='zone':p=np.r_[p,np.asarray(p)[:1]]
                self.canvas.create_line(*self._canvas_points(p),fill=color,width=width,tags='tools')
        for index,point in enumerate(self.tools.get('anchors',[])):
            x,y=self._canvas_points([point]);self.canvas.create_oval(x-5,y-5,x+5,y+5,fill='#FF615E',outline='white',width=1,tags='tools')
            self.canvas.create_text(x+9,y-9,text=str(index+1),fill='#FF615E',anchor='w',tags='tools')
        active=self.clicks if self.mode in ('polygon','endpoints') else self.gesture
        if len(active)>1:self.canvas.create_line(*self._canvas_points(active),fill='#64C5FF',width=2,tags='tools')
        for point in self.clicks:
            x,y=self._canvas_points([point]);self.canvas.create_oval(x-3,y-3,x+3,y+3,fill='#FF615E',tags='tools')

    def _pointer_down(self,event):
        if self._circle_armed or self.mode=='pan':return super()._pointer_down(event)
        point=self._image_point(event)
        if point is None:return 'break'
        if self.mode in ('polygon','endpoints'):
            if not self.clicks or np.linalg.norm(np.asarray(point)-self.clicks[-1])>1/self.scale:self.clicks.append(point)
            self._draw_tools()
            if self.mode=='endpoints' and len(self.clicks)==2:self._complete(self.clicks)
        else:
            self.gesture=[point];self._draw_tools()
        return 'break'

    def _pointer_drag(self,event):
        if self._circle_armed or self.mode=='pan':return super()._pointer_drag(event)
        if self.mode in ('guide','free_polygon','review_zone') and self.gesture:
            point=self._image_point(event)
            if point is not None and np.linalg.norm(np.asarray(point)-self.gesture[-1])*self.scale>=2:
                self.gesture.append(point);self._draw_tools()
        return 'break'

    def _pointer_up(self,event):
        if self._circle_armed or self.mode=='pan':return super()._pointer_up(event)
        if self.mode in ('guide','free_polygon','review_zone') and self.gesture:
            point=self._image_point(event)
            if point is not None:self.gesture.append(point)
            required=2 if self.mode=='guide' else 3
            if len(self.gesture)>=required:self._complete(self.gesture)
        return 'break'

    def _finish_polygon(self,event=None):
        if self.mode=='polygon' and len(self.clicks)>=3:self._complete(self.clicks)
        return 'break'

    def _complete(self,points):
        callback=self.interaction_callback;p=np.asarray(points,float).copy()
        self.mode='pan';self.gesture=[];self.clicks=[];self.interaction_callback=None
        self.canvas.configure(cursor='fleur');self.info.set('滚轮缩放 · 左键拖动');self._draw_tools()
        if callback:callback(p)

    def _right_pan(self,event):self.pan=(event.x,event.y);return 'break'
    def _right_drag(self,event):super()._drag(event);return 'break'

    def draw(self,*args,**kwargs):
        super().draw(*args,**kwargs);self._draw_tools()
        if self.mode!='pan':
            label={'polygon':'逐点圈定，双击或 Enter 闭合','endpoints':'大致点击两个端点','guide':'按住左键粗画参考走向','free_polygon':'按住左键圈定搜索 ROI','review_zone':'按住左键圈选需要复核的一段'}.get(self.mode,'')
            self.info.set(label+' · 滚轮缩放 · 右键拖动 · Esc 取消')


class UISettings:
    """Atomic preference JSON, separate from session data and worktable INI."""
    def __init__(self, path=None):
        explicit = path or os.environ.get('ROI_POLAR_SETTINGS_PATH')
        app_dir = Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
        self.fallback = Path(os.environ.get('LOCALAPPDATA') or Path.home())/'RoiPolar'/'v4_3_0'/'ui_settings.json'
        self.path = Path(explicit) if explicit else app_dir/'ui_settings.json'
        self._explicit = bool(explicit); self.data = {}; self.error = ''; self._corrupt = False
        if not explicit and self.fallback.exists():
            if not self.path.exists() or self.fallback.stat().st_mtime > self.path.stat().st_mtime:
                self.path = self.fallback
        source=self.path
        legacy=next((self.fallback.parent.parent/v/'ui_settings.json' for v in ('v4_2_9','v4_2_8','v4_2_7','v4_2_6','v4_2_5','v4_2_4','v4_2_3','v4_2_2','v4_2_1')
                     if (self.fallback.parent.parent/v/'ui_settings.json').exists()),None)
        if not explicit and not source.exists() and legacy:source=legacy
        if source.exists():
            try:
                value = json.loads(source.read_text(encoding='utf-8-sig'))
                if not isinstance(value,dict): raise ValueError('配置根节点应为对象')
                self.data = value
            except Exception as exc:
                self.error = '界面配置读取失败，已使用默认值：'+str(exc); self._corrupt = True

    def save(self, payload):
        content = json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        path = self.path
        try: self._write(path,content)
        except OSError:
            if self._explicit or path == self.fallback: raise
            self._write(self.fallback,content); self.path = self.fallback
        self.data = payload; self.error = ''; self._corrupt = False

    def _write(self,path,content):
        path.parent.mkdir(parents=True,exist_ok=True)
        if self._corrupt and path.exists():
            backup = path.with_name(path.stem+'.corrupt_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S%f')+path.suffix)
            shutil.copy2(path,backup)
        fd,temp = tempfile.mkstemp(prefix='ui_settings_',suffix='.tmp',dir=path.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp,path)
        finally:
            if os.path.exists(temp): os.unlink(temp)


PERSISTENT_VARIABLES = (
    'output_var','auto_save_var','save_image_var','save_polar_var','save_cartesian_var',
    'coordinate_format_var','use_regression_var','selection_var','polarity_var','region_var',
    'path_regularize_var','path_strength_var','gray_auto','gray_min_var','gray_max_var',
    'step_var','main_fraction_var','min_area_var','width_ratio_var','area_auto','area_px_var',
    'illumination_var','refine_var','contrast_auto','contrast_var','gap_auto','gap_var',
    'host_var','port_var','encoding_var','style_width','show_origin','show_axes','axes_mode_var',
    'show_ring_center','show_ring_roi',
    'ring_reference_var','polar_reference_var','show_polar_ray','show_wafer_outline',
)


class RoiPolarApp:
    def __init__(self,root,auto_listen=True,settings_path=None):
        self.settings=UISettings(settings_path)
        self.root=root;root.title("不规则 ROI 边界提取 V4.3.0")
        root.geometry("1420x900");root.minsize(1060,700);root.configure(bg=BG)
        families=set(tkfont.families(root))
        self.font=next((f for f in ("Microsoft YaHei UI","Noto Sans CJK SC","WenQuanYi Zen Hei") if f in families),"TkDefaultFont")
        self.analysis=None;self.gray=None;self.image_path=None
        self._loading=False;self._learning=False
        self._base_config=DetectorConfig();self._circle=None;self._polygon=None
        self._review_active=False;self._review_busy=False;self._review_token=0
        self._review_step=1
        self._review_selection=None;self._review_guide=None;self._review_preview=None;self._review_undo=[]
        self._review_entry_analysis=None;self._last_output=None;self._socket_controls_locked=False;self._manual_save_preferences=None
        self._learn_token=0;self._learn_after=None
        self._display_after=None;self._ring_token=0;self._ring_gray=None;self._ring_result=None;self._ring_busy=False
        self._polar_job=0;self._polar_key=None;self._polar_pending=None;self._polar_table=None;self._polar_error=None
        self._path_job=0;self._path_key=None;self._preview_path=None;self._path_pending=None;self._path_error=None
        self._defaults_signature=None
        self.events=queue.Queue();self._closing=False;self._close_requested=False;self._poll_after=None
        self.runner=TaskRunner(self.events);self.server=SocketServer(self.runner)
        self.help_icons={}
        self._vars();self._apply_saved_variables();self._theme();self._build();self._restore_ui_state()
        self.style_width.trace_add('write',self._on_display_change)
        root.protocol("WM_DELETE_WINDOW",self.close)
        try:self._sync_defaults()
        except Exception as exc:self.status_var.set(str(exc))
        self.select_tab('操作与显示')
        self._poll_after=root.after(60,self._poll)
        if auto_listen:self.toggle_socket()

    def _apply_saved_variables(self):
        saved=self.settings.data.get('ui',{})
        if not isinstance(saved,dict):return
        saved=dict(saved)
        if saved.get('selection_var')=='机台（0.5）':saved['selection_var']='自适应机台'
        choices={'coordinate_format_var':('CSV','TXT'),'selection_var':tuple(POINT_SELECTION),'polarity_var':tuple(POLARITIES),
                 'region_var':tuple(REGIONS),'encoding_var':('GBK','UTF-8'),
                 'axes_mode_var':('直角坐标系','极坐标系')}
        choices['ring_reference_var']=('图片左上角','图片中心')
        choices['polar_reference_var']=tuple(POLAR_REFERENCES)
        for name in PERSISTENT_VARIABLES:
            if name not in saved:continue
            variable=getattr(self,name);value=saved[name]
            if isinstance(variable,tk.BooleanVar):
                if not isinstance(value,bool):continue
            elif not isinstance(value,str) or len(value)>32768:continue
            if name in choices and value not in choices[name]:continue
            variable.set(value)
        # Older versions used top-left as the implicit circle display default.
        # Start this revision at image center, then retain explicit new choices.
        if self.settings.data.get('version') not in ('4.2.8','4.2.9','4.3.0'):self.ring_reference_var.set('图片中心')
        if not self.use_regression_var.get():self.polar_reference_var.set('图片水平向右')
        colors=saved.get('colors')
        if isinstance(colors,list) and colors and all(isinstance(c,str) and re.fullmatch(r'#[0-9a-fA-F]{6}',c) for c in colors):
            self.colors=colors[:1]
        # Legacy paths, search coordinates and window state are session data.
        # Only the allowlisted parameter preferences above are restored.

    def _restore_ui_state(self):
        data=self.settings.data
        section=data.get('calibration',{})
        if isinstance(section,dict):
            panel=self.calibration_panel
            mode=section.get('mode')
            if mode in ('质心（戳破薄膜）','圆环（黑色圆环内孔）'):panel.mode.set(mode)
            radius=section.get('radius')
            if isinstance(radius,str) and len(radius)<100:panel.radius.set(radius)
            if isinstance(section.get('advance'),bool):panel.advance.set(section['advance'])
        self._on_region_change();self._sync_gray_controls();self._rebuild_color_rows()
        view=data.get('view',{})
        if isinstance(view,dict):
            for attr in ('save_section','display_section','advanced_section','polar_section'):
                desired=view.get(attr);section=getattr(self,attr)
                if isinstance(desired,bool) and desired!=section.open:section.toggle()
            expanded=view.get('equation_section')
            if isinstance(expanded,bool) and expanded!=self.calibration_panel.equation_section.open:self.calibration_panel.equation_section.toggle()
            if view.get('socket_details') in ('rules','time'):self.toggle_details(view['socket_details'])
        self.worktable_notebook.select(self.calibration_body)
        self.verification_panel.restore_preferences(data.get('verification'))
        self._calibration_changed()
        if self.settings.error:self.status_var.set(self.settings.error)

    def _save_ui_settings(self):
        self.calibration_panel._finish_editor()
        # Durable worktable calibration remains in its own INI, never this JSON.
        self.calibration_panel.store.save()
        ui={name:getattr(self,name).get() for name in PERSISTENT_VARIABLES}
        ui['colors']=list(self.colors)
        if self._manual_save_preferences is not None:
            ui['output_var'],ui['auto_save_var']=self._manual_save_preferences
        panel=self.calibration_panel
        payload={'schema_version':3,'version':'4.3.0','ui':ui,
            'view':{'save_section':self.save_section.open,
                    'polar_section':self.polar_section.open,
                    'display_section':self.display_section.open,'advanced_section':self.advanced_section.open,
                    'socket_details':self._detail_open,'equation_section':panel.equation_section.open},
            'calibration':{'mode':panel.mode.get(),'radius':panel.radius.get(),'advance':panel.advance.get()},
            'verification':self.verification_panel.save_preferences()}
        self.settings.save(payload)


    def _vars(self):
        self.input_var=tk.StringVar(value="")
        self.output_var=tk.StringVar(value=str(Path.cwd()/"roi_results"))
        self.status_var=tk.StringVar(value="选择图片，或等待 Socket 任务")
        self.socket_status_var=tk.StringVar(value="Socket：未启动")
        self.image_var=tk.StringVar(value="尚未选择图片")
        self.auto_save_var=tk.BooleanVar(value=False)
        self.save_image_var=tk.BooleanVar(value=True);self.save_polar_var=tk.BooleanVar(value=True)
        self.save_cartesian_var=tk.BooleanVar(value=True);self.coordinate_format_var=tk.StringVar(value='CSV')
        self.preview_title_var=tk.StringVar(value='检测图像')
        self.use_regression_var=tk.BooleanVar(value=False)
        self.selection_var=tk.StringVar(value='1 像素')
        self.selection_hint=tk.StringVar(value='直角坐标：0 点；极坐标：0 点')
        self.worktable_label_var=tk.StringVar(value='工作台1')
        self.polarity_var=tk.StringVar(value="白色 ROI")
        self.region_var=tk.StringVar(value=next(iter(REGIONS)))
        self.circle_var=tk.StringVar(value="尚未画圆")
        self.polygon_var=tk.StringVar(value="尚未圈定手动 ROI")
        self.review_hint=tk.StringVar(value="先圈选复核区或选两个端点，再粗画参考走向。选段临时高亮，采用后恢复主体颜色。")
        self.path_regularize_var=tk.BooleanVar(value=True);self.path_strength_var=tk.StringVar(value="1.0")
        self.gray_auto=tk.BooleanVar(value=True)
        self.gray_min_var=tk.StringVar(value="");self.gray_max_var=tk.StringVar(value="")
        self.gray_info_var=tk.StringVar(value="灰度范围在检测时自动学习，无需等待")
        self.count_var=tk.StringVar(value="1");self.step_var=tk.StringVar(value="1.0")
        self.main_fraction_var=tk.StringVar(value="60")
        self.min_area_var=tk.StringVar(value="0.15");self.width_ratio_var=tk.StringVar(value="15")
        self.area_auto=tk.BooleanVar(value=True);self.area_px_var=tk.StringVar(value="10000")
        self.illumination_var=tk.BooleanVar(value=False);self.refine_var=tk.BooleanVar(value=True)
        self.contrast_auto=tk.BooleanVar(value=True);self.contrast_var=tk.StringVar(value="8")
        self.gap_auto=tk.BooleanVar(value=True);self.gap_var=tk.StringVar(value="2.0")
        self.host_var=tk.StringVar(value="127.0.0.1");self.port_var=tk.StringVar(value="5000");self.encoding_var=tk.StringVar(value="GBK")
        self.task_var=tk.StringVar(value="等待任务");self.detail_var=tk.StringVar(value="尚未检测")
        self.reply_var=tk.StringVar(value="—");self.metrics_time=tk.StringVar(value="—")
        self.metrics_points=tk.StringVar(value="—");self.metrics_gray=tk.StringVar(value="—")
        self.style_width=tk.StringVar(value="3");self.show_origin=tk.BooleanVar(value=True)
        self.origin_coords_var=tk.StringVar(value='—');self.ring_coords_var=tk.StringVar(value='未拟合')
        self.ring_machine_coords_var=tk.StringVar(value='—')
        self.show_axes=tk.BooleanVar(value=False);self.axes_mode_var=tk.StringVar(value='直角坐标系')
        self.show_ring_center=tk.BooleanVar(value=False);self.show_ring_roi=tk.BooleanVar(value=False)
        self.ring_status_var=tk.StringVar(value='')
        self.ring_reference_var=tk.StringVar(value='图片中心')
        self.polar_reference_var=tk.StringVar(value='图片水平向右')
        self.polar_reference_caption_var=tk.StringVar(value='角度基准\n对应机台角：—')
        self.show_polar_ray=tk.BooleanVar(value=False);self.polar_angle_var=tk.StringVar(value='0')
        self.show_wafer_outline=tk.BooleanVar(value=False)
        self.polar_error_var=tk.StringVar(value='')
        self.colors=[DEFAULT_COLORS[0]];self.color_buttons=[]

    def _theme(self):
        style=ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".",background=PANEL,foreground=FG,font=(self.font,10))
        style.configure("TFrame",background=PANEL)
        style.configure('Work.TNotebook',background=PANEL,borderwidth=0)
        style.configure('Work.TNotebook.Tab',background='#242424',foreground=FG,padding=(12,7))
        style.map('Work.TNotebook.Tab',background=[('selected','#295B35'),('active','#333333')],foreground=[('selected',FG),('active',FG)])
        style.configure("RegressionActive.TButton",background="#295B35",foreground=FG,padding=(8,6),borderwidth=0)
        style.map("RegressionActive.TButton",background=[("active","#347C41")])
        style.configure("TLabel",background=PANEL,foreground=FG)
        style.configure("TButton",background="#303030",foreground=FG,padding=(8,6),borderwidth=0)
        style.map("TButton",background=[("active","#414141"),("disabled","#252525")],foreground=[("disabled","#777777")])
        style.configure("Green.TButton",background="#3F8B4C",foreground="white",font=(self.font,11,"bold"),padding=(12,10))
        style.map("Green.TButton",background=[("disabled","#31573A"),("active","#347C41"),("!disabled","#3F8B4C")])
        style.configure("Blue.TButton",background="#285E96",foreground="white",padding=(10,9))
        style.map("Blue.TButton",background=[("disabled","#263746"),("pressed","#245581"),("active","#327BD6"),("!disabled","#285E96")],
            foreground=[("disabled","#79838D"),("!disabled",FG)])
        # Workflow controls use a quiet outline instead of a filled action style.
        style.configure("Auxiliary.TButton",background=PANEL,foreground="#CCD3DB",bordercolor="#55616E",
            lightcolor=PANEL,darkcolor=PANEL,borderwidth=1,relief="solid",padding=(8,6))
        style.map("Auxiliary.TButton",
            background=[("disabled",PANEL),("pressed","#293643"),("active","#232C35")],
            foreground=[("disabled","#626A73"),("active",FG)],
            bordercolor=[("disabled","#343A40"),("focus",BLUE),("active","#7A8A9B")])
        style.configure("TLabelframe",background=PANEL,bordercolor="#3A3A3A",relief="solid")
        style.configure("TLabelframe.Label",background=PANEL,foreground=FG,font=(self.font,10,"bold"))
        for name in ("TEntry","TSpinbox","TCombobox"):
            style.configure(name,fieldbackground=ENTRY,background=ENTRY,foreground=FG,insertcolor=FG,arrowcolor=FG,padding=(5,4))
            style.map(name,fieldbackground=[("disabled","#202020"),("readonly",ENTRY)],foreground=[("disabled","#777777"),("readonly",FG)])
        style.configure("TCheckbutton",background=PANEL,foreground=FG)
        style.map("TCheckbutton",background=[("active",PANEL)])
        style.configure("TScrollbar",background="#333333",troughcolor="#121212",arrowcolor=FG)
        style.configure("TProgressbar",background=BLUE,troughcolor=ENTRY)
        style.configure("Hidden.TNotebook",background=PANEL,borderwidth=0)
        style.layout("Hidden.TNotebook.Tab",[])
        style.configure("Treeview",background="#151515",fieldbackground="#151515",foreground=FG,rowheight=28,borderwidth=0)
        style.configure("Treeview.Heading",background="#303030",foreground=FG)
        style.map("Treeview",background=[("selected","#275535")])
        self.root.option_add("*TCombobox*Listbox.background",ENTRY)
        self.root.option_add("*TCombobox*Listbox.foreground",FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground","#275535")
        configure_combobox_selection(self.root,style)

    def _build(self):
        paned=ttk.Panedwindow(self.root,orient="horizontal");paned.pack(fill="both",expand=True);self.paned=paned
        left=ttk.Frame(paned,padding=(12,12,6,12));right=ttk.Frame(paned,padding=(10,12,12,12),width=455)
        # Prevent stretched table columns and scroll-page geometry from growing
        # the sidebar's requested width and collapsing the image pane.
        right.pack_propagate(False)
        paned.add(left,weight=5);paned.add(right,weight=0)
        def place_initial_sash():
            # Font loading can delay window mapping beyond an arbitrary timer.
            # Setting a sash while its widget is still 1px wide clips it to 0.
            if not paned.winfo_ismapped() or paned.winfo_width()<900:
                self.root.after(40,place_initial_sash)
                return
            paned.sashpos(0,min(max(400,paned.winfo_width()-465),paned.winfo_width()-320))
        self.root.after_idle(place_initial_sash)
        header=ttk.Frame(left);header.pack(fill="x",pady=(0,8))
        ttk.Label(header,textvariable=self.preview_title_var,font=(self.font,14,"bold")).pack(side="left")
        self.regression_button=tk.Button(header,textvariable=self.worktable_label_var,
            command=self.open_worktable,bg=PANEL,fg='#FFD15C',activebackground='#243329',
            activeforeground='#FFD15C',font=(self.font,10),cursor='hand2',relief='flat',borderwidth=0,
            highlightthickness=0,padx=10,pady=4)
        self.regression_button.pack(side='right',anchor='center')
        self.worktable_status_label=self.regression_button
        ttk.Label(left,textvariable=self.image_var,foreground=MUTED).pack(anchor="w",pady=(0,6))
        self.preview_container=ttk.Frame(left);self.preview_container.pack(fill="both",expand=True)
        self.view=InteractiveImageView(self.preview_container,self.font);self.view.pack(fill="both",expand=True)
        self.view.coordinate_origin_provider=self._coordinate_origin
        self.view.coordinate_center_provider=self._ring_center

        footer=ttk.Frame(right,padding=(0,8,0,0));footer.pack(side="bottom",fill="x")
        self.operation_footer=footer
        self.detect_button=ttk.Button(footer,text="开始检测",style="Green.TButton",command=self.start_detection,state="disabled")
        self.detect_button.pack(fill="x")
        actions=ttk.Frame(footer);actions.pack(fill='x',pady=(5,0))
        self.clear_button=ttk.Button(actions,text="清除当前结果",command=self.clear_current_result,state="disabled")
        self.clear_button.pack(side='left',fill='x',expand=True,padx=(0,3))
        self.review_button=ttk.Button(actions,text="开始复核",command=self._toggle_review,state="disabled")
        self.review_button.pack(side='left',fill='x',expand=True,padx=(3,0))
        result=ttk.LabelFrame(footer,text="检测结果",padding=10);result.pack(fill="x",pady=(10,0))
        row=ttk.Frame(result);row.pack(fill="x")
        for label,var in (("检测时间",self.metrics_time),("采样点数",self.metrics_points),("中位灰度",self.metrics_gray)):
            box=ttk.Frame(row);box.pack(side="left",fill="x",expand=True)
            ttk.Label(box,text=label,foreground=MUTED).pack(anchor="w")
            ttk.Label(box,textvariable=var,foreground=FG,font=(self.font,14,"bold")).pack(anchor="w")

        nav=tk.Frame(right,bg=PANEL,height=44);nav.pack(fill="x");nav.pack_propagate(False)
        self.navigation=nav
        self.tabs={};self.pages={};self.scrollers={};self.bodies={}
        for title in ("操作与显示","参数页面","Socket通讯"):
            label=tk.Label(nav,text=title,bg="#151515",fg=MUTED,font=(self.font,10),cursor="hand2",padx=6,pady=10)
            label.pack(side="left",expand=True,fill="both",padx=(0,1));label.bind("<Button-1>",lambda e,t=title:self.select_tab(t))
            self.tabs[title]=label
        self.socket_status_label=tk.Label(right,textvariable=self.socket_status_var,bg=PANEL,fg=MUTED,font=(self.font,10),anchor="w",padx=10,pady=8,wraplength=395)
        self.socket_status_label.pack(fill="x",pady=(5,6))
        self.notebook=ttk.Notebook(right,style="Hidden.TNotebook",takefocus=0);self.notebook.pack(fill="both",expand=True)
        for title in (*self.tabs,"坐标回归"):
            page=ttk.Frame(self.notebook);self.notebook.add(page,text=title);self.pages[title]=page
            canvas=tk.Canvas(page,bg=PANEL,highlightthickness=0,width=420)
            scrollbar=ttk.Scrollbar(page,orient="vertical",command=canvas.yview)
            scrollbar.pack(side="right",fill="y");canvas.pack(side="left",fill="both",expand=True);canvas.configure(yscrollcommand=scrollbar.set)
            body=ttk.Frame(canvas,padding=(1,4,8,12));window=canvas.create_window((0,0),window=body,anchor="nw")
            body.bind("<Configure>",lambda e,c=canvas:c.configure(scrollregion=c.bbox("all")))
            canvas.bind("<Configure>",lambda e,c=canvas,i=window:c.itemconfigure(i,width=max(50,e.width)))
            self.scrollers[title]=canvas;self.bodies[title]=body
        self.root.bind_all("<MouseWheel>",self._sidebar_wheel,add="+")
        self.root.bind_all("<Button-4>",self._sidebar_wheel,add="+");self.root.bind_all("<Button-5>",self._sidebar_wheel,add="+")
        self._build_operation(self.bodies["操作与显示"])
        self._build_parameters(self.bodies["参数页面"])
        self._build_socket(self.bodies["Socket通讯"])
        self.worktable_notebook=WorktableNotebook(self.bodies['坐标回归'])
        self.worktable_notebook.pack(fill='both',expand=True)
        self.calibration_body=ttk.Frame(self.worktable_notebook);self.verification_body=ttk.Frame(self.worktable_notebook)
        self.worktable_notebook.add(self.calibration_body,text='标定工作台')
        self.worktable_notebook.add(self.verification_body,text='复检')
        self.calibration_panel=CalibrationPanel(self.calibration_body,on_change=self._calibration_changed,
            image_provider=lambda:(self.gray,self.image_path),preview_parent=self.preview_container,
            on_preview_info=self._calibration_preview_info)
        self.calibration_panel.pack(fill="both",expand=True)
        self.verification_panel=VerificationPanel(self.verification_body,self.calibration_panel,self.preview_container,
            on_preview_info=self._verification_preview_info,image_provider=lambda:(self.gray,self.image_path))
        self.verification_panel.pack(fill='both',expand=True)
        self.worktable_notebook.bind('<<NotebookTabChanged>>',self._worktable_page_changed,add='+')
        self._calibration_changed()

    def open_worktable(self):
        """Every worktable entry starts with calibration, including this session."""
        self.worktable_notebook.select(self.calibration_body)
        self.select_tab('坐标回归')
        self.scrollers['坐标回归'].yview_moveto(0)

    def select_tab(self,title):
        ParameterHelp.close_active()
        if title=='坐标回归' and getattr(self,'active_tab',None)!=title:
            self.worktable_notebook.select(self.calibration_body)
        self.active_tab=title
        self.notebook.select(self.pages[title])
        self.regression_button.configure(bg='#213326' if title=='坐标回归' else PANEL)
        if title=='坐标回归':
            self.operation_footer.pack_forget();self.view.pack_forget()
            self._show_worktable_preview()
        else:
            self.calibration_panel.hide_preview()
            self.verification_panel.hide_preview()
            self.view.pack(fill='both',expand=True)
            if not self.operation_footer.winfo_manager():self.operation_footer.pack(side='bottom',fill='x',before=self.navigation)
            self.preview_title_var.set('检测图像')
            if self.gray is not None:
                self.image_var.set(f'{self.image_path.name if self.image_path else "当前图片"}  ·  {self.gray.shape[1]} × {self.gray.shape[0]}')
            else:self.image_var.set('尚未选择图片')
            self.redraw()
        for name,label in self.tabs.items():label.configure(bg="#295B35" if name==title else "#151515",fg=FG if name==title else MUTED)

    def _calibration_preview_info(self,text):
        if getattr(self,'active_tab',None)=='坐标回归' and not self._is_verification_page():self.image_var.set(text)

    def _verification_preview_info(self,text):
        if getattr(self,'active_tab',None)=='坐标回归' and self._is_verification_page():self.image_var.set(text)

    def _is_verification_page(self):
        return hasattr(self,'worktable_notebook') and self.worktable_notebook.select()==str(self.verification_body)

    def _show_worktable_preview(self):
        self.calibration_panel.hide_preview();self.verification_panel.hide_preview()
        verify=self._is_verification_page()
        self.preview_title_var.set('复检图像' if verify else '标定图像')
        panel=self.verification_panel if verify else self.calibration_panel
        panel.show_preview();self.image_var.set(panel.image_info.get())

    def _worktable_page_changed(self,event=None):
        if not hasattr(self,'verification_panel'):return
        self.verification_panel.refresh_tables()
        # Let Tk propagate the full content request. Reading an unmapped frame's
        # initial 1px request and fixing the height here clipped the first visit.
        self.scrollers['坐标回归'].yview_moveto(0)
        if getattr(self,'active_tab',None)=='坐标回归':self._show_worktable_preview()

    def _coordinate_origin(self):
        return self.calibration_panel.origin_key if hasattr(self,'calibration_panel') else 'center'

    def _ring_center(self):
        return self._ring_result['center_image_xy_px'] if self._ring_result and self._ring_result.get('status')=='ok' else None

    def _sidebar_wheel(self,event):
        target=self.scrollers[self.active_tab];widget=event.widget
        while widget is not None and widget is not self.pages[self.active_tab]:
            panel=getattr(self,'calibration_panel',None)
            if panel is not None and widget in (panel.tree,panel.table_scrollbar):
                return panel._table_wheel(event)
            widget=getattr(widget,"master",None)
        if widget is None:return
        bounds=target.bbox("all")
        if bounds and bounds[3]>target.winfo_height():
            units=-1 if getattr(event,"num",None)==4 or getattr(event,"delta",0)>0 else 1
            target.yview_scroll(units*3,"units")
        return "break"

    @staticmethod
    def group(parent,title):
        group=ttk.LabelFrame(parent,text=title,padding=10);group.pack(fill="x",pady=(0,12));return group

    def _help_icon(self,parent,key,title):
        icon=ParameterHelp(parent,title,ADVANCED_HELP[key],self.font)
        icon.pack(side='left',padx=(5,0));self.help_icons[key]=icon
        return icon

    def row(self,parent,title,var,width=12,help_key=None):
        row=ttk.Frame(parent);row.pack(fill="x",pady=4)
        ttk.Label(row,text=title).pack(side="left")
        if help_key:self._help_icon(row,help_key,title)
        entry=ttk.Entry(row,textvariable=var,width=width);entry.pack(side="right");return entry

    def combo(self,parent,title,var,choices):
        row=ttk.Frame(parent);row.pack(fill="x",pady=4);ttk.Label(row,text=title).pack(side="left")
        combo=ttk.Combobox(row,textvariable=var,values=list(choices),state="readonly",width=22);combo.pack(side="right");return combo

    def _auto_field(self,parent,title,value,automatic,help_key=None):
        row=ttk.Frame(parent);row.pack(fill="x",pady=5)
        ttk.Label(row,text=title).pack(side="left")
        if help_key:self._help_icon(row,help_key,title)
        entry=ttk.Entry(row,textvariable=value,width=8);entry.pack(side="right")
        def sync():entry.configure(state="disabled" if automatic.get() else "normal")
        TickCheckbutton(row,text="自动",variable=automatic,command=sync).pack(side="right",padx=(4,10))
        automatic.trace_add("write",lambda *a:sync());sync()

    def _build_operation(self,parent):
        group=self.group(parent,"输入与保存")
        row=ttk.Frame(group);row.pack(fill="x",pady=(0,7))
        ttk.Entry(row,textvariable=self.input_var,state="readonly",width=20).pack(side="left",fill="x",expand=True)
        self.open_button=ttk.Button(row,text="选择图片",command=self.choose_image)
        self.open_button.pack(side="right",padx=(6,0))
        row=ttk.Frame(group);row.pack(fill="x",pady=(0,7))
        self.output_entry=ttk.Entry(row,textvariable=self.output_var,width=20);self.output_entry.pack(side="left",fill="x",expand=True)
        self.output_button=ttk.Button(row,text="选择目录",command=self.choose_output);self.output_button.pack(side="right",padx=(6,0))
        self.auto_save_check=TickCheckbutton(group,text="自动保存检测结果",variable=self.auto_save_var)
        self.auto_save_check.pack(anchor="w",pady=(2,5))
        self.save_section=FoldSection(group,'保存选项')
        self.save_section.button.configure(command=self._toggle_save_options)
        body=self.save_section.body
        TickCheckbutton(body,text='结果图',variable=self.save_image_var,command=self._save_options_changed).pack(anchor='w',pady=4)
        row=ttk.Frame(body);row.pack(fill='x',pady=4)
        TickCheckbutton(row,text='极坐标（id、theta、r1…）',variable=self.save_polar_var,command=self._save_options_changed).pack(side='left')
        ttk.Label(row,textvariable=self.ring_coords_var,foreground='#76D6FF').pack(side='right')
        row=ttk.Frame(body);row.pack(fill='x',pady=(0,4))
        ttk.Label(row,text='圆心机台 X、Y',foreground=MUTED).pack(side='left',padx=(18,0))
        ttk.Label(row,textvariable=self.ring_machine_coords_var,foreground='#69E58C').pack(side='right')
        self.polar_section=FoldSection(body,'极坐标查看选项')
        self.polar_controls=self.polar_section.body
        row=ttk.Frame(self.polar_controls);row.pack(fill='x',pady=3)
        ttk.Label(row,text='圆心显示基准').pack(side='left')
        self.ring_reference_combo=ttk.Combobox(row,textvariable=self.ring_reference_var,values=('图片左上角','图片中心'),state='readonly',width=13)
        self.ring_reference_combo.pack(side='right');self.ring_reference_combo.bind('<<ComboboxSelected>>',self._on_display_change)
        row=ttk.Frame(self.polar_controls);row.pack(fill='x',pady=4)
        self.polar_reference_caption=ttk.Label(row,textvariable=self.polar_reference_caption_var,justify='left',wraplength=220)
        self.polar_reference_caption.pack(side='left',fill='x',expand=True)
        self.polar_reference_combo=ttk.Combobox(row,textvariable=self.polar_reference_var,values=list(POLAR_REFERENCES),state='readonly',width=14)
        self.polar_reference_combo.pack(side='right')
        row.bind('<Configure>',lambda event:self.polar_reference_caption.configure(
            wraplength=max(80,event.width-self.polar_reference_combo.winfo_reqwidth()-8)))
        self.polar_reference_combo.configure(width=14)
        self.polar_reference_combo.bind('<<ComboboxSelected>>',self._polar_reference_changed)
        row=ttk.Frame(self.polar_controls);row.pack(fill='x',pady=3)
        TickCheckbutton(row,text='查看射线与交点',variable=self.show_polar_ray,command=self._polar_view_changed).pack(side='left')
        ttk.Label(row,text='θ / °').pack(side='left',padx=(10,3))
        self.polar_angle_entry=ttk.Spinbox(row,from_=0,to=359.5,increment=.5,textvariable=self.polar_angle_var,width=8,command=self._polar_view_changed)
        self.polar_angle_entry.pack(side='right')
        TickCheckbutton(self.polar_controls,text='显示晶圆真实外沿',variable=self.show_wafer_outline,command=self._polar_view_changed).pack(anchor='w',pady=3)
        self.polar_error_label=ttk.Label(self.polar_controls,textvariable=self.polar_error_var,foreground='#FFD15C',wraplength=350,justify='left')
        self.polar_angle_var.trace_add('write',self._polar_angle_edited)
        for text,var in (('直角坐标（id、x、y）',self.save_cartesian_var),):
            TickCheckbutton(body,text=text,variable=var,command=self._save_options_changed).pack(anchor='w',pady=4)
        TickCheckbutton(body,text='使用转换回归方程（当前工作台）',variable=self.use_regression_var,command=self._save_options_changed).pack(anchor='w',pady=4)
        self.coordinate_format_combo=self.combo(body,'坐标文件格式',self.coordinate_format_var,('CSV','TXT'))
        self.coordinate_format_combo.bind('<<ComboboxSelected>>',self._save_options_changed)
        self.selection_combo=self.combo(body,'XY 选点模式',self.selection_var,POINT_SELECTION)
        self.selection_combo.bind('<<ComboboxSelected>>',self._selection_changed)
        ttk.Label(body,textvariable=self.selection_hint,foreground=BLUE,wraplength=350,justify='left').pack(fill='x',pady=(4,0))
        self._save_options_changed()
        self._build_review_controls(parent)

    def _build_parameters(self,parent):
        group=self.group(parent,"检测参数")
        self.polarity_combo=self.combo(group,"灰度特征",self.polarity_var,POLARITIES)
        self.polarity_combo.bind('<<ComboboxSelected>>',self._on_polarity_change)
        row=ttk.Frame(group);row.pack(fill='x',pady=5)
        ttk.Label(row,text='灰度范围').pack(side='left')
        self.gray_auto_check=TickCheckbutton(row,text='自动',variable=self.gray_auto,command=self._gray_auto_changed)
        self.gray_auto_check.pack(side='right')
        row=ttk.Frame(group);row.pack(fill='x',pady=4)
        ttk.Label(row,text='下限').pack(side='left')
        self.gray_min_entry=ttk.Entry(row,textvariable=self.gray_min_var,width=8);self.gray_min_entry.pack(side='left',padx=(8,15))
        ttk.Label(row,text='上限').pack(side='left')
        self.gray_max_entry=ttk.Entry(row,textvariable=self.gray_max_var,width=8);self.gray_max_entry.pack(side='left',padx=(8,0))
        ttk.Label(group,textvariable=self.gray_info_var,foreground=MUTED,wraplength=355).pack(anchor='w',pady=(3,8))
        self.region_combo=self.combo(group,'搜索范围',self.region_var,REGIONS)
        self.region_combo.bind('<<ComboboxSelected>>',self._on_region_change)
        self.circle_controls=ttk.Frame(group)
        ttk.Label(self.circle_controls,textvariable=self.circle_var,foreground=MUTED,wraplength=350).pack(fill='x',pady=(2,5))
        row=ttk.Frame(self.circle_controls);row.pack(fill='x')
        self.draw_circle_button=ttk.Button(row,text='在图上画圆',command=self._draw_circle);self.draw_circle_button.pack(side='left',fill='x',expand=True,padx=(0,5))
        ttk.Button(row,text='清除圆形',command=self._clear_circle).pack(side='left',fill='x',expand=True)
        self.polygon_controls=ttk.Frame(group)
        ttk.Label(self.polygon_controls,textvariable=self.polygon_var,foreground=MUTED,wraplength=350).pack(fill='x',pady=4)
        row=ttk.Frame(self.polygon_controls);row.pack(fill='x')
        self.draw_polygon_button=ttk.Button(row,text='逐点圈定',command=lambda:self._draw_polygon(False));self.draw_polygon_button.pack(side='left',expand=True,fill='x')
        self.draw_free_polygon_button=ttk.Button(row,text='自由圈选',command=lambda:self._draw_polygon(True));self.draw_free_polygon_button.pack(side='left',expand=True,fill='x',padx=3)
        ttk.Button(row,text='清除 ROI',command=self._clear_polygon).pack(side='left',expand=True,fill='x')
        ttk.Label(self.polygon_controls,text='仅限制搜索范围；坐标原点跟随工作台标定页',foreground=MUTED,wraplength=350).pack(fill='x',pady=4)
        self.expected_entry=self.row(group,'输出圈数（固定）',self.count_var)
        self.expected_entry.configure(state='disabled')
        ttk.Label(group,text='只输出主体闭环；交错处优先内侧',foreground=MUTED).pack(anchor='w',pady=(0,6))
        self._sync_gray_controls()

        self.display_section=FoldSection(parent,'显示设置')
        group=self.display_section.body
        self.color_rows=ttk.Frame(group);self.color_rows.pack(fill='x');self._rebuild_color_rows()
        self.row(group,'标注线宽 / px',self.style_width)
        row=ttk.Frame(group);row.pack(fill='x',pady=5)
        TickCheckbutton(row,text='显示坐标原点',variable=self.show_origin,command=self._on_display_change).pack(side='left')
        ttk.Label(row,textvariable=self.origin_coords_var,foreground=MUTED).pack(side='right')
        row=ttk.Frame(group);row.pack(fill='x',pady=5)
        TickCheckbutton(row,text='显示坐标系',variable=self.show_axes,command=self._on_display_change).pack(side='left')
        self.axes_mode_combo=ttk.Combobox(row,textvariable=self.axes_mode_var,values=('直角坐标系','极坐标系'),state='readonly',width=12)
        self.axes_mode_combo.pack(side='right');self.axes_mode_combo.bind('<<ComboboxSelected>>',self._on_display_change)
        row=ttk.Frame(group);row.pack(fill='x',pady=5)
        TickCheckbutton(row,text='显示亮圆环拟合圆心',variable=self.show_ring_center,command=self._on_display_change).pack(side='left')
        ttk.Label(row,textvariable=self.ring_coords_var,foreground='#76D6FF').pack(side='right')
        TickCheckbutton(group,text='显示亮圆环 ROI',variable=self.show_ring_roi,command=self._on_display_change).pack(anchor='w',pady=5)
        row=ttk.Frame(group);row.pack(fill='x',pady=3)
        ttk.Label(row,text='拟合圆心显示基准').pack(side='left')
        combo=ttk.Combobox(row,textvariable=self.ring_reference_var,values=('图片左上角','图片中心'),state='readonly',width=13)
        combo.pack(side='right');combo.bind('<<ComboboxSelected>>',self._on_display_change)
        ttk.Label(group,text='圆心显示可切换基准，单位 px；不改变固定圆心极坐标。',foreground=MUTED,wraplength=350).pack(fill='x',pady=(4,0))
        self.ring_status_label=ttk.Label(group,textvariable=self.ring_status_var,foreground=MUTED,wraplength=350)
        self.advanced_section=FoldSection(parent,'检测高级参数')
        group=self.advanced_section.body
        self.row(group,'主体候选面积保留 / %',self.main_fraction_var,help_key='main_fraction')
        self._auto_field(group,'断点修补半径 / px',self.gap_var,self.gap_auto,'gap')
        self.row(group,'最小闭环面积 / 搜索区 %',self.min_area_var,help_key='min_ratio')
        self._auto_field(group,'最小闭环面积 / px²',self.area_px_var,self.area_auto,'min_px')
        self.row(group,'最大笔宽 / √面积 %',self.width_ratio_var,help_key='width')
        self.row(group,'曲线采样间距 / px',self.step_var,help_key='step')
        row=ttk.Frame(group);row.pack(fill='x',pady=4)
        TickCheckbutton(row,text='缓变亮度校正',variable=self.illumination_var,command=self._schedule_learning).pack(side='left')
        self._help_icon(row,'illumination','缓变亮度校正')
        row=ttk.Frame(group);row.pack(fill='x',pady=4)
        TickCheckbutton(row,text='沿灰度截面校正中线',variable=self.refine_var).pack(side='left')
        self._help_icon(row,'refine','沿灰度截面校正中线')
        self._auto_field(group,'局部灰度对比下限',self.contrast_var,self.contrast_auto,'contrast')
        TickCheckbutton(group,text='抑制复杂折返与跳线',variable=self.path_regularize_var).pack(anchor='w',pady=5)
        self.row(group,'曲线整形强度',self.path_strength_var,help_key='path')

    def _build_socket(self,parent):
        group=self.group(parent,"Socket 通讯")
        self.host_entry=self.row(group,"IP 地址",self.host_var,20)
        self.port_entry=self.row(group,"端口",self.port_var,20)
        self.enc_combo=self.combo(group,"字符编码",self.encoding_var,("GBK","UTF-8"))
        self.socket_button=ttk.Button(group,text="开始监听",style="Blue.TButton",command=self.toggle_socket)
        self.socket_button.pack(fill="x",pady=(10,0))
        group=self.group(parent,"当前任务")
        ttk.Label(group,textvariable=self.task_var,wraplength=375,justify="left").pack(fill="x")
        self.progress=ttk.Progressbar(group,maximum=100);self.progress.pack(fill="x",pady=(10,0))
        ttk.Label(group,textvariable=self.status_var,wraplength=375,foreground=MUTED).pack(fill='x',pady=(7,0))
        ttk.Label(group,text="上次返回",foreground=MUTED).pack(anchor="w",pady=(10,4))
        ttk.Label(group,textvariable=self.reply_var,foreground=BLUE,wraplength=375,justify="left").pack(fill="x")
        bar=ttk.Frame(parent);bar.pack(fill="x",pady=(0,8))
        self.rule_button=ttk.Button(bar,text="▶ 通讯规则",command=lambda:self.toggle_details("rules"));self.rule_button.pack(side="left",padx=(0,6))
        self.time_button=ttk.Button(bar,text="▶ 细节及时间",command=lambda:self.toggle_details("time"));self.time_button.pack(side="left")
        self.rule_frame=ttk.Frame(parent,padding=8);self.time_frame=ttk.Frame(parent,padding=8);self._detail_open=None
        rules=("接收：片号,型号,[预留编号],原路径,图片或文件夹名,保存目录\n\n"
               "返回：片号,时间,OK或REVIEW,圈数,保存目录,结果文件名\n"
               "失败：NG,原因。圈数校验失败会保存诊断结果并返回NG。\n\n"
               "输入支持5/6字段、GBK或UTF-8。建议以换行结帧；兼容50ms空闲结帧。文件夹每图一行，返回末尾空行表示任务结束。\n\n"
               "型号仅作任务信息；全部任务使用当前界面参数。\n\n"
               "仅按保存选项写入图片与坐标，不生成 JSON。末字段优先返回直角坐标文件名，其次极坐标，仅保存图片时返回图片名。\n\n"
               "坐标仅三列；启用回归后文件名带 machine，按提交任务时的工作台方程转换。尺寸不一致返回NG。REVIEW表示有结果但需复核。")
        ttk.Label(self.rule_frame,text=rules,wraplength=375,justify="left",foreground=MUTED).pack(fill="x")
        ttk.Label(self.time_frame,textvariable=self.detail_var,wraplength=375,justify="left",foreground=MUTED).pack(fill="x")

    def toggle_details(self,which):
        self._detail_open=None if self._detail_open==which else which
        self.rule_frame.pack_forget();self.time_frame.pack_forget()
        if self._detail_open=="rules":self.rule_frame.pack(fill="x")
        elif self._detail_open=="time":self.time_frame.pack(fill="x")
        self.rule_button.configure(text=("▼" if self._detail_open=="rules" else "▶")+" 通讯规则")
        self.time_button.configure(text=("▼" if self._detail_open=="time" else "▶")+" 细节及时间")

    def get_config(self):
        mode=REGIONS[self.region_var.get()]
        return dataclasses.replace(self._base_config,
            polarity=POLARITIES[self.polarity_var.get()],region_mode=mode,
            search_circle=self._circle if mode=='circle' else None,
            search_polygon=self._polygon if mode=='polygon' else None,
            regularize_path=self.path_regularize_var.get(),path_smoothing=float(self.path_strength_var.get()),
            gray_min=None if self.gray_auto.get() else int(self.gray_min_var.get()),
            gray_max=None if self.gray_auto.get() else int(self.gray_max_var.get()),
            black_threshold=None,white_threshold=None,
            expected_count=1,main_area_fraction=float(self.main_fraction_var.get())/100,sample_step_px=float(self.step_var.get()),
            gap_px=None if self.gap_auto.get() else float(self.gap_var.get()),
            min_area_ratio=float(self.min_area_var.get())/100,max_width_ratio=float(self.width_ratio_var.get())/100,
            min_area_px=None if self.area_auto.get() else float(self.area_px_var.get()),
            illumination_correction=self.illumination_var.get(),
            refine_profiles=self.refine_var.get(),feature_contrast_gray=None if self.contrast_auto.get() else float(self.contrast_var.get())).validate()

    def get_style(self):
        width=int(self.style_width.get())
        if not 1<=width<=30:raise ValueError('标注线宽应为1～30像素')
        try:angle=float(self.polar_angle_var.get())%360
        except ValueError:angle=0.
        if not math.isfinite(angle):angle=0.
        return {'colors':list(self.colors),'width':width,'show_origin':self.show_origin.get(),
                'coordinate_origin':self._coordinate_origin(),
                'show_axes':self.show_axes.get(),'axes_mode':'polar' if self.axes_mode_var.get()=='极坐标系' else 'cartesian',
                'show_ring_center':self.show_ring_center.get(),'show_ring_roi':self.show_ring_roi.get(),
                'require_ring':self.save_polar_var.get() or self._coordinate_origin()=='ring_center' or
                    (self.show_axes.get() and self.axes_mode_var.get()=='极坐标系'),
                'show_polar_ray':self.show_polar_ray.get(),'polar_angle_deg':angle,'show_wafer_outline':self.show_wafer_outline.get(),
                'polar_machine':self.use_regression_var.get(),
                'polar_angle_reference':POLAR_REFERENCES[self.polar_reference_var.get()] if self.use_regression_var.get() else 'image'}

    def _calibration_changed(self):
        if not hasattr(self,'calibration_panel'):return
        table=self.calibration_panel.active_table
        dirty=self.calibration_panel.equation_dirty
        suffix=' · 方程待应用' if dirty else (' · 手动方程' if table['fit'] and table.get('equation_source')=='manual' else (' · 已拟合' if table['fit'] else ' · 未拟合'))
        self.worktable_label_var.set(table['name']+suffix)
        color='#69E58C' if table['fit'] and not dirty else '#FFD15C'
        self.worktable_status_label.configure(fg=color,activeforeground=color)
        self._defaults_signature=None
        if hasattr(self,'verification_panel'):self.verification_panel.refresh_tables()
        self._save_options_changed()

    def get_export_config(self):
        use=self.use_regression_var.get()
        snapshot=self.calibration_panel.export_snapshot() if use else {}
        tolerance,machine_step=POINT_SELECTION[self.selection_var.get()]
        return ExportConfig(result_image=self.save_image_var.get(),polar=self.save_polar_var.get(),
            cartesian=self.save_cartesian_var.get(),coordinate_format=self.coordinate_format_var.get().lower(),
            origin=self._coordinate_origin(),use_regression=use,
            regression_coefficients=snapshot.get('coefficients'),worktable_name=snapshot.get('worktable',''),
            calibration_shape=snapshot.get('image_shape'),regression_origin=snapshot.get('origin',self._coordinate_origin()),
            regression_center=snapshot.get('ring_center'),
            simplify_px=tolerance,machine_step=machine_step,
            polar_angle_reference=POLAR_REFERENCES[self.polar_reference_var.get()] if use else 'image').validate()

    def _selection_changed(self,event=None):
        if self.selection_var.get()=='自适应机台':self.use_regression_var.set(True)
        self._save_options_changed()

    def _save_options_changed(self,event=None):
        use=self.use_regression_var.get()
        if not use:self.polar_reference_var.set('图片水平向右')
        self._update_corresponding_machine_angle()
        self._update_output_counts()
        if hasattr(self,'runner'):
            try:self._sync_defaults()
            except Exception as exc:self.status_var.set(str(exc))
        if hasattr(self,'view') and getattr(self,'active_tab',None)!='坐标回归':self.redraw()

    def _toggle_save_options(self):
        self.save_section.toggle();self._save_options_changed()

    def _clear_metrics(self):
        for var in (self.metrics_time,self.metrics_points,self.metrics_gray):var.set('—')
        self._update_output_counts()

    def _update_output_counts(self):
        # Count exactly the selected output data. Polar counts sum retained
        # ROI intersections only; the 720 mandatory W references are excluded.
        xy=polar=0
        if self.analysis is not None:
            if self.save_cartesian_var.get():
                path=self._preview_path if self._path_key==self._current_path_key() else None
                xy=path['point_count'] if path else '—'
            if self.save_polar_var.get():
                table=self._polar_table if self._polar_key==self._polar_geometry_key() else None
                polar=table['roi_point_count'] if table else '—'
        self.selection_hint.set(f'直角坐标：{xy} 点；极坐标：{polar} 点')

    def _show_metrics(self):
        if self.analysis is None:self._clear_metrics();return
        roi=self.analysis.rois[0]
        self.metrics_time.set(f"{self.analysis.metadata['detect_seconds']:.3f} 秒")
        path=self._preview_path if self._path_key==self._current_path_key() else None
        self.metrics_points.set(str(path['point_count']) if path else str(len(roi.points)))
        self.metrics_gray.set(f"{roi.info.get('gray_quantiles_10_50_90',[0,0,0])[1]:.1f}")

    def _sync_defaults(self):
        signature=tuple(getattr(self,name).get() for name in PERSISTENT_VARIABLES)+(repr(self._circle),repr(self._polygon),tuple(self.colors),self.polar_angle_var.get())
        if signature==self._defaults_signature:return
        try:
            self.runner.set_defaults(self.get_config(),self.get_style(),self.get_export_config())
            self._defaults_signature=signature
        except Exception as exc:
            self._defaults_signature=None;self.runner.set_default_error(str(exc));raise

    def choose_image(self):
        if self.runner.busy or self._loading or self._review_active or self._review_busy:return
        path=filedialog.askopenfilename(filetypes=FILE_TYPES)
        if path:self.load_path(path)

    def load_path(self,path):
        if self.runner.busy or self._loading or self._review_active or self._review_busy:return
        self._cancel_learning();self._loading=True;self.status_var.set('正在读取图片…');self._update_buttons()
        def work():
            try:self.events.put(('loaded',str(path),load_gray(path)))
            except Exception as exc:self.events.put(('load_error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def choose_output(self):
        if self.server.running:
            self.status_var.set("保存目录由 Socket 指令指定");return
        value=filedialog.askdirectory(initialdir=self.output_var.get())
        if value:self.output_var.set(value)

    def start_detection(self):
        if self.image_path is None or self.runner.busy or self._loading or self._review_active or self._review_busy:return
        try:
            cfg=self.get_config();style=self.get_style();options=self.get_export_config()
            saving=self.server.running or self.auto_save_var.get();options.validate(require_output=saving)
            self.runner.set_defaults(cfg,style,options)
            self.analysis=None;self._clear_metrics()
            if self.gray is not None:self.view.set_image(cv2.cvtColor(self.gray,cv2.COLOR_GRAY2BGR),reset=False)
            request={"source":str(self.image_path),"save_dir":self.output_var.get(),"plate":"manual","model":"","reserved":""}
            future=self.runner.submit(request,manual=True,config=cfg,style=style,save=saving,gray=self.gray,export_config=options)
            if future.done():future.result()
            self.status_var.set("正在提取闭合走势…");self._update_buttons()
        except Exception as exc:messagebox.showerror("参数错误",str(exc))

    def redraw(self):
        if self._closing:return
        try:
            style=self.get_style()
        except (ValueError,tk.TclError):return
        try:
            self._sync_auxiliary_image();self._refresh_coordinate_labels()
            self._update_output_counts()
            if self.gray is None:return
            path=self._selected_preview_path()
            points=(path['image_points'] if path else self.analysis.rois[0].points) if self.analysis is not None else None
            self._ensure_polar_geometry()
            self._update_polar_info()
            if self._polar_table is not None and self._polar_key==self._polar_geometry_key():
                if self._polar_table['space']=='machine':style['polar_linear']=self._polar_table['linear']
                if self.show_polar_ray.get():style['polar_ray']=self._polar_ray()
                if self.show_wafer_outline.get():style['wafer_outline']=ensure_wafer_outline(self.analysis)['points']
            bgr=render_image(self.gray,style,points,self._ring_result)
            self.view.set_image(bgr,reset=False);self._show_search_circle()
            self._ensure_ring_fit()
        except Exception as exc:messagebox.showerror("显示错误",str(exc))

    def _current_path_key(self):
        if self.analysis is None:return None
        try:return (id(self.analysis),id(self.analysis.rois[0].points),repr(self.get_export_config()))
        except (ValueError,KeyError,tk.TclError):return None

    def _selected_preview_path(self):
        key=self._current_path_key()
        if key is None:return None
        if key==self._path_key:return self._preview_path
        if self._path_pending==key or self._path_error==key:return None
        self._path_job+=1;job=self._path_job;analysis=self.analysis;options=self.get_export_config()
        self._path_pending=key
        def worker():
            try:self.events.put(('selection_ready',job,key,analysis,coordinate_path(analysis,options)))
            except Exception as exc:self.events.put(('selection_error',job,key,str(exc)))
        threading.Thread(target=worker,daemon=True,name='roi-select-points').start()
        return None

    def _on_display_change(self,*args):
        if self._closing:return
        try:self.get_style()
        except (ValueError,tk.TclError):return
        if self._display_after:self.root.after_cancel(self._display_after)
        self._display_after=self.root.after(65,self._apply_display_change)

    def _apply_display_change(self):
        self._display_after=None
        if self._closing:return
        try:self._sync_defaults()
        except ValueError:pass
        self.redraw()

    def _needs_ring(self):
        return (self.show_ring_center.get() or self.show_ring_roi.get() or self.save_polar_var.get()
                or self.show_polar_ray.get() or self.show_wafer_outline.get() or self._coordinate_origin()=='ring_center'
                or (self.show_axes.get() and self.axes_mode_var.get()=='极坐标系'))

    def _polar_geometry_key(self):
        if self.analysis is None or self.analysis.gray is not self.gray:return None
        p=self.analysis.rois[0].points
        try:options=self.get_export_config()
        except ValueError:return None
        return (id(self.analysis),id(p),len(p),self._ring_token,options.use_regression,options.regression_origin,
                options.polar_angle_reference,
                tuple(np.asarray(options.regression_coefficients).ravel()) if options.use_regression else ())

    def _ensure_polar_geometry(self):
        key=self._polar_geometry_key()
        if key is None or self.runner.busy or not (self.save_polar_var.get() or self.show_polar_ray.get() or self.show_wafer_outline.get()):return
        if key==self._polar_key or key==self._polar_pending or (self._polar_error and self._polar_error[0]==key):return
        if self._ring_result is None:return
        if self._ring_result.get('status')!='ok':
            self._polar_error=(key,'未找到可信亮圆环圆心，极坐标暂不可用。');self._update_polar_info();return
        self._polar_job+=1;job=self._polar_job;analysis=self.analysis
        options=self.get_export_config()
        analysis.metadata['bright_ring']=self._ring_result
        self._polar_pending=key;self._polar_table=None
        def work():
            try:self.events.put(('polar_ready',job,key,polar_ray_table(analysis,options)))
            except Exception as exc:self.events.put(('polar_error',job,key,str(exc)))
        threading.Thread(target=work,daemon=True,name='polar-ray-table').start()

    def _polar_ray(self):
        try:
            angle=float(self.polar_angle_var.get())
            if not math.isfinite(angle):return None
            return polar_ray_at_angle(self.analysis,angle,self.get_export_config())
        except (ValueError,TypeError):return None

    def _update_polar_info(self):
        self._update_corresponding_machine_angle()
        key=self._polar_geometry_key()
        error=''
        if self._polar_error and self._polar_error[0]==key:
            error=self._polar_error[1]
        elif key is not None and self.show_polar_ray.get():
            try:valid=math.isfinite(float(self.polar_angle_var.get()))
            except ValueError:valid=False
            if not valid:error='查看角度请输入有限数值。'
        self.polar_error_var.set(error)
        if error:self.polar_error_label.pack(fill='x',pady=(3,5))
        else:self.polar_error_label.pack_forget()
        self._update_output_counts()

    def _update_corresponding_machine_angle(self):
        if self.polar_reference_var.get()=='机台 X 正向':
            self.polar_reference_caption_var.set('角度基准')
            return
        text='角度基准\n对应机台角：—'
        try:
            angle=float(self.polar_angle_var.get())
            if not math.isfinite(angle):raise ValueError('无效角度')
            panel=getattr(self,'calibration_panel',None)
            if panel is None:raise ValueError('工作台尚未初始化')
            # Snapshot the active APPLIED equation every time; do not reuse an
            # old table's polar geometry or wait for its 720-ray background job.
            snapshot=panel.export_snapshot()
            if self.gray is not None and tuple(self.gray.shape)!=tuple(snapshot['image_shape']):
                raise ValueError('当前图像尺寸与工作台不一致')
            linear,_=machine_affine(snapshot['image_shape'],snapshot['coefficients'],
                                    snapshot['origin'],snapshot.get('ring_center'))
            theta=math.radians(angle%360)
            delta=linear@np.array([math.cos(theta),-math.sin(theta)])
            if not np.isfinite(delta).all() or not np.any(delta):raise ValueError('无效方向')
            machine_angle=math.degrees(math.atan2(delta[1],delta[0]))%360
            display_angle=round(machine_angle,3)%360
            text=f'角度基准\n对应机台角：{display_angle:.3f}°'
        except (ValueError,TypeError,KeyError,tk.TclError):pass
        self.polar_reference_caption_var.set(text)

    def _polar_view_changed(self,event=None):
        self._update_polar_info();self._on_display_change()

    def _polar_angle_edited(self,*args):
        if hasattr(self,'view') and not self._closing:self._polar_view_changed()

    def _polar_reference_changed(self,event=None):
        if self.polar_reference_var.get()=='机台 X 正向':self.use_regression_var.set(True)
        self._save_options_changed()

    def _sync_auxiliary_image(self):
        if self.gray is not self._ring_gray:
            self._ring_token+=1;self._ring_gray=self.gray;self._ring_result=None
            self._polar_job+=1;self._polar_key=None;self._polar_pending=None;self._polar_table=None;self._polar_error=None
        if self.analysis is not None and self.analysis.gray is self.gray and 'bright_ring' in self.analysis.metadata:
            self._ring_result=self.analysis.metadata['bright_ring']

    def _refresh_coordinate_labels(self):
        if self.gray is None:
            self.origin_coords_var.set('—');self.ring_coords_var.set('未拟合');self.ring_machine_coords_var.set('—');return
        try:
            origin,_=coordinate_frame(self.gray.shape,self._coordinate_origin(),self._ring_center())
            self.origin_coords_var.set(f'({origin[0]:.3f}, {origin[1]:.3f})')
        except ValueError:self.origin_coords_var.set('圆心待拟合')
        fit=self._ring_result
        if fit and fit.get('status')=='ok':
            origin,sign=coordinate_frame(self.gray.shape,'center' if self.ring_reference_var.get()=='图片中心' else 'top_left')
            x,y=np.asarray(fit['center_image_xy_px'])-origin;y*=sign
            self.ring_coords_var.set(f'({x:.3f}, {y:.3f})');self.ring_status_var.set('')
        elif fit:
            self.ring_coords_var.set('未识别');self.ring_status_var.set(fit.get('reason','未找到可信圆环'))
        elif self._needs_ring():
            self.ring_coords_var.set('拟合中…');self.ring_status_var.set('')
        else:self.ring_coords_var.set('未拟合');self.ring_status_var.set('')
        self.ring_machine_coords_var.set('待拟合圆心')
        if fit and fit.get('status')=='ok':
            try:
                snapshot=self.calibration_panel.export_snapshot()
                if tuple(self.gray.shape)!=tuple(snapshot['image_shape']):raise ValueError('图像尺寸不匹配')
                linear,offset=machine_affine(self.gray.shape,snapshot['coefficients'],snapshot['origin'],fit['center_image_xy_px'])
                x,y=np.asarray(fit['center_image_xy_px'])@linear.T+offset
                self.ring_machine_coords_var.set(f'({x:.3f}, {y:.3f})')
            except ValueError:
                self.ring_machine_coords_var.set('方程待应用' if self.calibration_panel.equation_dirty else
                    '图像尺寸不匹配' if self.calibration_panel.active_table.get('fit') else '工作台未拟合')
        if self.ring_status_var.get():self.ring_status_label.pack(fill='x',pady=(4,0))
        else:self.ring_status_label.pack_forget()

    def _ensure_ring_fit(self):
        if self._closing or self.gray is None:return
        self._sync_auxiliary_image()
        if not self._needs_ring():return
        if self._ring_result is not None or self._ring_busy or self.runner.busy:return
        gray=self.gray;token=self._ring_token;self._ring_busy=True
        self._refresh_coordinate_labels()
        def worker():
            try:result=fit_bright_ring(gray)
            except Exception as exc:result={'status':'not_found','reason':'圆环拟合失败：'+str(exc)}
            self.events.put(('ring_fitted',token,gray,result))
        threading.Thread(target=worker,daemon=True,name='bright-ring-fit').start()

    def clear_current_result(self):
        if self.runner.busy or self._loading or self._review_active or self._review_busy:return
        self.analysis=None
        self._clear_metrics()
        self.progress.configure(value=0)
        self.task_var.set('等待任务');self.reply_var.set('—');self.detail_var.set('当前结果已清除')
        self.status_var.set('当前结果已清除，可重新检测')
        if self.view._circle_armed:self.view.cancel_circle()
        self.redraw();self._update_buttons()

    def choose_color(self,index):
        value=colorchooser.askcolor(self.colors[index],parent=self.root)[1]
        if value:
            self.colors[index]=value;self.color_buttons[index].configure(bg=value);self._on_display_change()





    def toggle_socket(self):
        try:
            if self.server.running:self.server.stop();self.socket_status_var.set("Socket：已停止")
            else:
                self._sync_defaults()
                host,port=self.server.start(self.host_var.get().strip() or "127.0.0.1",int(self.port_var.get()),self.encoding_var.get().lower())
                self.socket_status_var.set(f"Socket：监听中 {host}:{port}")
            running=self.server.running
            self.socket_button.configure(text="停止监听" if running else "开始监听")
            self.host_entry.configure(state="disabled" if running else "normal");self.port_entry.configure(state="disabled" if running else "normal")
            self.enc_combo.configure(state="disabled" if running else "readonly")
        except Exception as exc:self.socket_status_var.set("Socket：启动失败 "+str(exc))
        self._sync_save_controls();self._update_socket_indicator()

    def _update_socket_indicator(self):
        running=self.server.running
        self.socket_status_label.configure(bg='#11253A' if running else PANEL,fg=BLUE if running else MUTED)

    def _update_buttons(self):
        busy=self.runner.busy or self._loading or self._review_busy
        editing=self._review_active
        mode=REGIONS[self.region_var.get()]
        region_ready=(mode!='circle' or self._circle is not None) and (mode!='polygon' or self._polygon is not None)
        self.detect_button.configure(state='normal' if self.image_path and not busy and not editing and self.runner.ready and region_ready else 'disabled')
        self.open_button.configure(state='disabled' if busy or editing else 'normal')
        has_result=self.analysis is not None
        self.clear_button.configure(state='normal' if has_result and not busy and not editing else 'disabled')
        for button in (self.draw_circle_button,self.draw_polygon_button,self.draw_free_polygon_button):
            button.configure(state='normal' if self.gray is not None and not busy and not editing else 'disabled')
        self.polarity_combo.configure(state='disabled' if busy or editing else 'readonly')
        self.region_combo.configure(state='disabled' if busy or editing else 'readonly')
        self.review_button.configure(state='normal' if self.analysis is not None and not busy else 'disabled',text='退出复核' if editing else '开始复核')
        for button in self._review_basic_buttons:button.configure(state='normal' if editing and not busy else 'disabled')
        self.review_flip_button.configure(state='normal' if editing and self._review_selection is not None and not busy else 'disabled')
        self.review_guide_button.configure(state='normal' if editing and self._review_selection is not None and not busy else 'disabled')
        self.review_preview_button.configure(state='normal' if editing and self._review_selection is not None and not busy else 'disabled')
        self.review_apply_button.configure(state='normal' if editing and self._review_preview is not None and not busy else 'disabled')
        can_undo=bool(self._review_undo) or self._review_selection is not None or self._review_guide is not None or self._review_preview is not None
        self.review_undo_button.configure(state='normal' if editing and can_undo and not busy else 'disabled')
        self.review_save_button.configure(state='normal' if editing and not busy and self._review_preview is None else 'disabled')
        self.review_previous_button.configure(state='normal' if editing and self._review_step>1 and not busy else 'disabled')
        can_next=(self._review_step==1 and self._review_selection is not None) or (self._review_step==2 and self._review_preview is not None)
        self.review_next_button.configure(state='normal' if editing and can_next and not busy else 'disabled')
        for index,button in enumerate(self.review_step_buttons,start=1):
            allowed=editing and not busy and (index!=2 or self._review_selection is not None)
            button.configure(state='normal' if allowed else 'disabled')

    def _poll(self):
        if self._closing:return
        try:self._sync_defaults()
        except Exception:pass
        for _ in range(64):
            try:event=self.events.get_nowait()
            except queue.Empty:break
            kind=event[0]
            if kind in ('selection_ready','selection_error'):
                if event[1]!=self._path_job:continue
                if event[2]!=self._current_path_key():
                    if self._path_pending==event[2]:self._path_pending=None
                    continue
                self._path_pending=None
                if kind=='selection_ready':
                    self._path_key=event[2];self._preview_path=event[4];self._path_error=None
                    self._show_metrics();self.redraw()
                else:
                    self._path_error=event[2];self._update_output_counts();self.status_var.set(event[3])
                continue
            if kind in ('polar_ready','polar_error'):
                if event[1]!=self._polar_job or event[2]!=self._polar_geometry_key():continue
                self._polar_pending=None
                if kind=='polar_ready':
                    self._polar_key=event[2];self._polar_table=event[3];self._polar_error=None
                else:self._polar_table=None;self._polar_error=(event[2],event[3])
                self._update_polar_info();self.redraw();continue
            if kind=='ring_fitted':
                self._ring_busy=False
                if event[1]==self._ring_token and event[2] is self.gray:
                    self._ring_result=event[3]
                    if self.analysis is not None and self.analysis.gray is self.gray:self.analysis.metadata['bright_ring']=event[3]
                    self.redraw()
                continue
            if kind.startswith('review_'):
                self._review_event(event);continue
            if kind=='loaded':
                _,path,gray=event;self._loading=False;self.image_path=Path(path);self.gray=gray;self.analysis=None
                self._last_output=None;self._reset_review_selection();self.view.cancel_interaction()
                self.input_var.set(path);self.image_var.set(f'{Path(path).name}  ·  {gray.shape[1]} × {gray.shape[0]}')
                self._clear_metrics()
                self.view.set_image(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR));self._show_search_circle()
                self.status_var.set('图片已载入，点击开始检测');self._schedule_learning();self.redraw()
            elif kind in ('load_error','task_error'):
                self._loading=False;self.status_var.set(event[1]);self._clear_metrics()
                if kind=='load_error':messagebox.showerror('读取图片',event[1])
                elif not self.server.running:messagebox.showerror('检测失败',event[1])
            elif kind=='task_started':
                _,request,origin=event
                if self.server.running and request.get('submission_source')=='socket':self.output_var.set(request['save_dir'])
                self._cancel_learning()
                self.task_var.set(f"片号：{request.get('plate','')}\n型号：{request.get('model','')}\n使用当前界面参数")
                self.status_var.set('正在检测…');self.analysis=None;self._clear_metrics()
            elif kind=='item_started':
                _,index,total,path=event;self.progress.configure(value=index/total*100)
                self.status_var.set(f'正在检测 {index+1}/{total}：{Path(path).name}')
                self.view.clear();self.gray=None;self.image_var.set(Path(path).name)
            elif kind=='item_loaded':
                _,path,gray=event;self.gray=gray;self.image_path=Path(path);self.input_var.set(path)
                self.image_var.set(f'{self.image_path.name}  ·  {gray.shape[1]} × {gray.shape[0]}')
                self.view.set_image(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR));self._show_search_circle()
                self.redraw()
            elif kind=='item_result':
                _,analysis,output,index,total,origin=event
                self._last_output=output;self._review_undo=[];self._reset_review_selection()
                self.analysis=analysis;self.gray=analysis.gray
                self.image_path=Path(analysis.metadata.get('source','result'));self.input_var.set(str(self.image_path))
                self.image_var.set(f'{self.image_path.name}  ·  {self.gray.shape[1]} × {self.gray.shape[0]}')
                self._show_metrics()
                seconds=analysis.metadata['detect_seconds']
                self.progress.configure(value=index/total*100)
                warnings=analysis.metadata['warnings'];self.status_var.set('提取完成')
                self.detail_var.set(f"检测：{seconds:.3f} s\n保存：{output['directory'] if output else '未自动保存'}\n"+'\n'.join(warnings))
                if self.gray_auto.get() and analysis.metadata['parameters']['polarity']==POLARITIES[self.polarity_var.get()]:
                    self._set_gray_bounds(analysis.metadata['gray_range'])
                self.redraw()
            elif kind=='item_error':
                self._clear_metrics();self.metrics_time.set(f'{event[4]:.3f} 秒');self.analysis=None
                self.status_var.set(event[1]);self.progress.configure(value=event[2]/event[3]*100)
                if not self.server.running:messagebox.showerror('检测失败',event[1])
            elif kind=='socket_reply':self.reply_var.set(event[1])
            elif kind=='task_finished':self.detail_var.set(self.detail_var.get()+f'\n任务总耗时：{event[1]:.3f} s')
        self._update_buttons()
        self._ensure_ring_fit()
        self._ensure_polar_geometry()
        if self.active_tab=='坐标回归':
            panel=self.verification_panel if self._is_verification_page() else self.calibration_panel
            self.image_var.set(panel.image_info.get())
        if self._close_requested and not self.runner.busy and not self._loading and not self._learning:
            self.close();return
        self._poll_after=self.root.after(80,self._poll)

    def close(self):
        if self._review_active and not self._review_busy and not self._exit_review():return
        ParameterHelp.close_active()
        if self.runner.busy or self._loading or self._learning or self._review_busy:
            self.server.stop();self._close_requested=True
            self.status_var.set('正在完成当前任务，随后关闭窗口');return
        try:self._save_ui_settings()
        except Exception as exc:messagebox.showwarning('界面参数保存失败',str(exc),parent=self.root)
        self._closing=True;self.server.stop();self.runner.close()
        if self._poll_after:self.root.after_cancel(self._poll_after)
        if self._learn_after:self.root.after_cancel(self._learn_after)
        if self._display_after:self.root.after_cancel(self._display_after)
        self._ring_token+=1
        self.root.destroy()

    def _rebuild_color_rows(self):
        for child in self.color_rows.winfo_children():child.destroy()
        self.color_buttons=[]
        for index,color in enumerate(self.colors):
            row=ttk.Frame(self.color_rows);row.pack(fill='x',pady=4)
            ttk.Label(row,text='主体闭环颜色').pack(side='left')
            button=tk.Button(row,text='',width=7,height=1,bg=color,activebackground=color,relief='flat',
                             command=lambda i=index:self.choose_color(i))
            button.pack(side='right',padx=(10,0));self.color_buttons.append(button)

    def _sync_gray_controls(self):
        state='disabled' if self.gray_auto.get() else 'normal'
        self.gray_min_entry.configure(state=state);self.gray_max_entry.configure(state=state)

    def _gray_auto_changed(self):
        self._sync_gray_controls()
        if self.gray_auto.get():self._schedule_learning()
        else:
            self._cancel_learning();self.gray_info_var.set('手动灰度范围，包含上下限')

    def _set_gray_bounds(self,bounds):
        if not self.gray_auto.get():return
        if bounds.get('available'):
            self.gray_min_var.set(str(bounds['min']));self.gray_max_var.set(str(bounds['max']))
            self.gray_info_var.set('已从当前图片学习灰度范围'+('（亮度校正后）' if self.illumination_var.get() else ''))
        else:
            self.gray_min_var.set('');self.gray_max_var.set('');self.gray_info_var.set('未找到稳定笔迹，可改用手动灰度范围')
        self._sync_gray_controls()

    def _cancel_learning(self):
        self._learn_token+=1;self._learning=False
        if self._learn_after:
            self.root.after_cancel(self._learn_after);self._learn_after=None

    def _schedule_learning(self):
        # Learning is part of analyze(); loading/changing parameters never runs
        # another topology scan or disables the detection button.
        self._cancel_learning()
        if self.gray_auto.get():
            self.gray_min_var.set('');self.gray_max_var.set('')
            self.gray_info_var.set('灰度范围在检测时自动学习，无需等待')
        self._sync_gray_controls()

    def _on_polarity_change(self,event=None):
        self.analysis=None;self._clear_metrics()
        if self.gray is not None:self.redraw()
        self._schedule_learning()

    def _show_search_circle(self):
        self.view.set_circle(self._circle if REGIONS[self.region_var.get()]=='circle' else None)
        self.view.search_polygon=self._polygon if REGIONS[self.region_var.get()]=='polygon' else None
        self.view._draw_tools()

    def _on_region_change(self,event=None):
        if REGIONS[self.region_var.get()]=='circle':
            self.circle_controls.pack(fill='x',pady=(2,8),before=self.expected_entry.master)
        else:self.circle_controls.pack_forget()
        if REGIONS[self.region_var.get()]=='polygon':self.polygon_controls.pack(fill='x',pady=(2,8),before=self.expected_entry.master)
        else:self.polygon_controls.pack_forget()
        self.view.cancel_interaction();self._show_search_circle();self._schedule_learning()

    def _draw_circle(self):
        if self.gray is None:return
        self.region_var.set('手动圆形区域');self._on_region_change()
        self.view.arm_circle(self._circle_done)
        self.status_var.set('按住左键确定圆心，拖动设置半径；松开完成')

    def _circle_done(self,circle):
        self._circle=tuple(map(float,circle))
        cx,cy,r=self._circle;self.circle_var.set(f'圆心 ({cx:.1f}, {cy:.1f})  半径 {r:.1f} px')
        self._show_search_circle();self._schedule_learning()
        self.status_var.set('圆形搜索范围已设置；坐标原点跟随工作台标定页')

    def _clear_circle(self):
        self._circle=None;self.circle_var.set('尚未画圆');self.view.set_circle(None)
        self._schedule_learning()


    def _sync_save_controls(self):
        running=self.server.running
        if running and not self._socket_controls_locked:
            self._manual_save_preferences=(self.output_var.get(),self.auto_save_var.get())
        if not running and self._socket_controls_locked and self._manual_save_preferences is not None:
            directory,automatic=self._manual_save_preferences
            self.output_var.set(directory);self.auto_save_var.set(automatic);self._manual_save_preferences=None
        self._socket_controls_locked=running
        if running:self.auto_save_var.set(True)
        self.output_entry.configure(state='disabled' if running else 'normal')
        self.output_button.configure(state='disabled' if running else 'normal')
        self.auto_save_check.set_enabled(not running)

    def _draw_polygon(self,freehand=False):
        if self.gray is None or self._review_active or self.runner.busy:return
        self.region_var.set('手动 ROI');self._on_region_change()
        self.view.arm('free_polygon' if freehand else 'polygon',self._polygon_done)
        self.status_var.set('拖动圈选；松开完成' if freehand else '沿搜索范围点选边界，双击或 Enter 闭合；Esc 取消')

    def _polygon_done(self,points):
        try:
            from shapely.geometry import Polygon
            p=np.asarray(points,float)
            poly=Polygon(p)
            if not poly.is_valid or poly.area<4:raise ValueError('ROI 需要闭合且无自交，请重新圈选')
            # A dense gesture is reduced geometrically; it remains a search mask.
            simplified=poly.simplify(max(.5,1/self.view.scale),preserve_topology=True)
            self._polygon=tuple(map(tuple,np.asarray(simplified.exterior.coords)[:-1]))
            self.polygon_var.set(f'已圈定 {len(self._polygon)} 点搜索 ROI')
            self._show_search_circle();self._schedule_learning();self._sync_defaults();self._update_buttons()
            self.status_var.set('手动搜索 ROI 已设置；坐标原点跟随工作台标定页')
        except Exception as exc:messagebox.showerror('搜索 ROI',str(exc))

    def _clear_polygon(self):
        self._polygon=None;self.polygon_var.set('尚未圈定手动 ROI');self._show_search_circle();self._schedule_learning();self._update_buttons()

    def _build_review_controls(self,parent):
        self.review_group=self.group(parent,'局部复核')
        nav=ttk.Frame(self.review_group);nav.pack(fill='x',pady=(0,8))
        self.review_step_buttons=[];self.review_step_frames={}
        for step,title in ((1,'1 选择线段'),(2,'2 参考走向'),(3,'3 采用与保存')):
            button=ttk.Button(nav,text=title,command=lambda s=step:self._review_set_step(s))
            button.pack(side='left',fill='x',expand=True,padx=2);self.review_step_buttons.append(button)
        ttk.Label(self.review_group,textvariable=self.review_hint,foreground=BLUE,wraplength=350).pack(fill='x',pady=(0,7))
        self.review_step_container=ttk.Frame(self.review_group);self.review_step_container.pack(fill='x')
        for step in (1,2,3):self.review_step_frames[step]=ttk.Frame(self.review_step_container)
        self._review_basic_buttons=[]
        row=ttk.Frame(self.review_step_frames[1]);row.pack(fill='x',pady=3)
        for text,command in [('圈选复核区',self._review_zone),('选两个端点',self._review_endpoints)]:
            b=ttk.Button(row,text=text,command=command);b.pack(side='left',fill='x',expand=True,padx=2);self._review_basic_buttons.append(b)
        self.review_flip_button=ttk.Button(self.review_step_frames[1],text='切换选段',command=self._review_flip);self.review_flip_button.pack(fill='x',padx=2,pady=3)
        row=ttk.Frame(self.review_step_frames[2]);row.pack(fill='x',pady=3)
        self.review_guide_button=ttk.Button(row,text='绘制参考走向',command=self._review_draw_guide);self.review_guide_button.pack(side='left',fill='x',expand=True,padx=2)
        self.review_preview_button=ttk.Button(row,text='生成预览',command=self._review_generate);self.review_preview_button.pack(side='left',fill='x',expand=True,padx=2)
        self.review_apply_button=ttk.Button(self.review_step_frames[3],text='采用此段',command=self._review_apply);self.review_apply_button.pack(fill='x',padx=2,pady=3)
        self.review_save_button=ttk.Button(self.review_step_frames[3],text='保存当前复核结果',style='Blue.TButton',command=self._review_confirm_save)
        self.review_save_button.pack(fill='x',padx=2,pady=3)
        ttk.Separator(self.review_group,orient='horizontal').pack(fill='x',pady=(10,7))
        row=ttk.Frame(self.review_group);row.pack(fill='x',pady=(0,3));row.columnconfigure((0,1),weight=1)
        self.review_previous_button=ttk.Button(row,text='上一步',style='Auxiliary.TButton',command=self._review_previous);self.review_previous_button.grid(row=0,column=0,sticky='ew',padx=2)
        self.review_next_button=ttk.Button(row,text='下一步',style='Auxiliary.TButton',command=lambda:self._review_set_step(self._review_step+1));self.review_next_button.grid(row=0,column=1,sticky='ew',padx=2)
        row=ttk.Frame(self.review_group);row.pack(fill='x',pady=3)
        self.review_undo_button=ttk.Button(row,text='撤销前一步',style='Auxiliary.TButton',command=self._review_undo_one);self.review_undo_button.pack(side='left',fill='x',expand=True,padx=2)
        self.review_exit_button=ttk.Button(row,text='退出复核',style='Auxiliary.TButton',command=self._exit_review)
        self.review_exit_button.pack(side='left',fill='x',expand=True,padx=2);self._review_basic_buttons.append(self.review_exit_button)
        self.review_step_frames[1].pack(fill='x');self.review_step_buttons[0].configure(style='Blue.TButton')
        self.review_group.pack_forget()

    def _review_set_step(self,step):
        if self._review_busy or not 1<=step<=3:return
        if step==2 and self._review_selection is None:
            self.review_hint.set('请先在第一步圈选复核区，或选取两个端点。');return
        self._review_step=step
        for number,frame in self.review_step_frames.items():
            frame.pack_forget()
            self.review_step_buttons[number-1].configure(style='Blue.TButton' if number==step else 'TButton')
        self.review_step_frames[step].pack(fill='x')
        # End steps disable navigation without removing its place in the layout.
        self.review_next_button.grid()
        self.review_previous_button.configure(text='返回选段' if step==3 and self._review_selection is None else '上一步')
        if step==1:self.review_hint.set('圈选复核区或选两个端点，二选一；必要时切换选段，然后点下一步。')
        elif step==2:self.review_hint.set('粗画期望走向，无需精确；生成预览后进入第三步，也可不画参考线直接预览。')
        elif self._review_preview is not None:self.review_hint.set('检查临时预览，采用此段后再保存；如需调整，可返回上一步。')
        else:self.review_hint.set('可保存当前曲线，或点击第一步继续复核其他位置。保存内容遵循上方选项。')
        self._update_buttons()

    def _review_previous(self):
        self._review_set_step(1 if self._review_step==3 and self._review_selection is None else max(1,self._review_step-1))

    def _reset_review_selection(self):
        self._review_selection=None;self._review_guide=None;self._review_preview=None
        if hasattr(self,'view'):self.view.set_tools()

    def _toggle_review(self):
        if self._review_active:self._exit_review();return
        if self.analysis is None or self.runner.busy or self._loading:return
        self._review_active=True;self._review_entry_analysis=self.analysis;self._review_undo=[]
        self._review_token+=1;self.runner.set_review_lock(True);self._reset_review_selection()
        if self.save_section.open:self._toggle_save_options()
        self.review_group.pack(fill='x',pady=(0,12));self.select_tab('操作与显示')
        self._review_set_step(1);self.redraw();self._update_buttons()
        self.status_var.set('正在人工复核；新的 Socket 任务会返回忙状态')

    def _exit_review(self,discard=False):
        if self._review_busy:return False
        if self._review_undo and not discard:
            if not messagebox.askyesno('结束复核','当前修改尚未确认保存。退出将恢复进入复核前的曲线，是否退出？'):return False
        if self._review_undo and self._review_entry_analysis is not None:self.analysis=self._review_entry_analysis
        self._review_active=False;self._review_token+=1;self._review_undo=[]
        self.runner.set_review_lock(False);self.view.cancel_interaction();self._reset_review_selection();self.review_group.pack_forget()
        self._show_metrics();self.redraw();self._update_buttons();return True

    def _review_endpoints(self):
        if not self._review_active or self._review_busy:return
        self._reset_review_selection();self.view.arm('endpoints',self._review_endpoints_done)
        self.review_hint.set('在待替换线段两端附近各点击一次，端点会自动吸附到当前曲线。')

    def _review_endpoints_done(self,points):
        try:
            self._review_selection=select_review_arc(self.analysis,endpoints=points)
            self._review_show_selection();self.review_hint.set('橙色为待替换段。可切换选段，然后粗画参考走向并生成预览。')
        except Exception as exc:messagebox.showerror('选取线段',str(exc))
        self._update_buttons()

    def _review_zone(self):
        if not self._review_active or self._review_busy:return
        self._reset_review_selection();self.view.arm('review_zone',self._review_zone_done)
        self.review_hint.set('用左键圈住需要修改的一段，松开后自动选择该区域内的连续曲线。')

    def _review_zone_done(self,polygon):
        try:
            self._review_selection=select_review_arc(self.analysis,polygon=polygon)
            self._review_show_selection();self.review_hint.set('已选中该区域内的一段；可粗画参考走向，也可直接生成平顺预览。')
        except Exception as exc:messagebox.showerror('圈选复核区域',str(exc))
        self._update_buttons()

    def _review_show_selection(self):
        selected=self._review_selection
        self.view.set_tools(selection=selected['points'] if selected else None,
            anchors=selected['endpoints'] if selected else None,guide=self._review_guide,
            preview=self._review_preview['points'] if self._review_preview else None)

    def _review_flip(self):
        if self._review_selection is None:return
        old=self._review_selection;p=self.analysis.rois[0].points[:-1];n=len(p)
        a,b=old['end_index'],old['start_index'];ids=(a+np.arange((b-a)%n+1))%n
        self._review_selection={'start_index':int(a),'end_index':int(b),'indices':ids,'points':p[ids].copy(),'endpoints':p[[a,b]].copy()}
        self._review_guide=None;self._review_preview=None;self._review_show_selection();self._update_buttons()

    def _review_draw_guide(self):
        if self._review_selection is None:return
        self._review_preview=None;self._review_show_selection();self.view.arm('guide',self._review_guide_done)
        self.review_hint.set('从一端附近向另一端粗画想走的路线，无需压在线上；软件会重新贴合笔迹。')

    def _review_guide_done(self,points):
        self._review_guide=np.asarray(points,float);self._review_preview=None;self._review_show_selection()
        self.review_hint.set('参考走向已记录。点击生成预览；蓝色预览确认合适后采用此段。');self._update_buttons()

    def _review_generate(self):
        if self._review_selection is None or self._review_busy:return
        self._review_busy=True;self._review_token+=1;token=self._review_token
        analysis=self.analysis;selection=self._review_selection;guide=self._review_guide
        self._update_buttons();self.review_hint.set('正在沿参考走向寻找连续笔迹并检查闭环…')
        def worker():
            try:self.events.put(('review_preview',token,preview_review(analysis,selection,guide)))
            except Exception as exc:self.events.put(('review_error',token,str(exc)))
        threading.Thread(target=worker,daemon=True,name='roi-review-preview').start()

    def _review_apply(self):
        if self._review_preview is None or self._review_busy:return
        try:
            candidate=apply_review(self.analysis,self._review_selection,self._review_preview['points'],self._review_preview['info'])
            candidate.metadata['review_history'][-1]['accepted_at']=dt.datetime.now().isoformat(timespec='seconds')
            self._review_undo.append(self.analysis);self._review_undo=self._review_undo[-8:]
            self.analysis=candidate;self._reset_review_selection();self._display_review_analysis()
            self._review_set_step(3)
            self.review_hint.set('已采用，整条曲线恢复主体颜色。可保存当前结果，或点击第一步继续复核其他位置。')
        except Exception as exc:messagebox.showerror('采用复核线段',str(exc))
        self._update_buttons()

    def _review_undo_one(self):
        if self._review_busy:return
        self.view.cancel_interaction()
        if self._review_preview is not None:
            self._review_preview=None;self._review_show_selection();self._review_set_step(2)
            self.review_hint.set('已撤销预览，参考走向保留，可重新生成预览。')
        elif self._review_guide is not None:
            self._review_guide=None;self._review_show_selection();self._review_set_step(2)
            self.review_hint.set('已撤销参考走向，可重新绘制。')
        elif self._review_selection is not None:
            self._reset_review_selection();self._review_set_step(1)
        elif self._review_undo:
            self.analysis=self._review_undo.pop();self._reset_review_selection();self._display_review_analysis();self._review_set_step(1)
            self.review_hint.set('已撤销上一处采用的修改，可重新选段。')
        self._update_buttons()

    def _display_review_analysis(self):
        if self.analysis is None:return
        self._show_metrics()
        self.redraw()

    def _review_confirm_save(self):
        if self.analysis is None or self._review_busy:return
        if self._review_preview is not None:
            messagebox.showinfo('尚有预览','请先采用此段，或重新选段取消该预览，再确认保存。');return
        try:options=self.get_export_config().validate(require_output=True);style=self.get_style()
        except Exception as exc:messagebox.showerror('保存选项',str(exc));return
        candidate=confirm_review(self.analysis)
        candidate.metadata['review_completed_at']=dt.datetime.now().isoformat(timespec='seconds')
        if self._last_output:candidate.metadata['supersedes_result']=self._last_output['primary']
        directory=self.output_var.get()
        stem=make_export_stem(self.image_path or 'result',reviewed=True)
        self._review_busy=True;self._review_token+=1;token=self._review_token
        self._update_buttons();self.review_hint.set('正在按保存选项导出当前复核结果…')
        def worker():
            try:self.events.put(('review_saved',token,candidate,export_analysis(candidate,directory,stem,style,options)))
            except Exception as exc:self.events.put(('review_error',token,str(exc)))
        threading.Thread(target=worker,daemon=True,name='roi-review-save').start()

    def _review_event(self,event):
        kind,token=event[:2]
        if token!=self._review_token:return
        self._review_busy=False
        if kind=='review_preview':
            self._review_preview=event[2];self._review_show_selection()
            self._review_set_step(3)
            weak=self._review_preview['info']['low_evidence_interpolation']
            text='蓝色为贴合附近笔迹后的预览。检查两端连接和整段走势，合适后采用此段。'
            if weak:text+=' 部分位置笔迹不清，请重点复核。'
            self.review_hint.set(text)
        elif kind=='review_saved':
            self.analysis=event[2];self._last_output=event[3];self._review_undo=[];self._review_active=False
            self.runner.set_review_lock(False);self.view.cancel_interaction();self._reset_review_selection();self.review_group.pack_forget()
            self._display_review_analysis();self.status_var.set('复核结果已保存：'+event[3]['primary'])
            self.detail_var.set('复核输出：'+event[3]['directory']+'\n'+'\n'.join(event[3]['files']))
        elif kind=='review_error':
            self.review_hint.set('操作未生效，原曲线保留：'+event[2]);messagebox.showerror('复核操作',event[2])
        self._update_buttons()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",nargs="?")
    parser.add_argument("--no-listen",action="store_true")
    parser.add_argument("--settings",type=Path)
    args=parser.parse_args()
    root=tk.Tk();app=RoiPolarApp(root,auto_listen=not args.no_listen,settings_path=args.settings)
    if args.image:
        root.after(100,lambda:app.load_path(args.image))
    root.mainloop()


if __name__=="__main__":main()

#!/usr/bin/env python3
import cv2
import numpy as np
import potrace
import PySimpleGUI as sg
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def bitmap_to_desmos_beziers(image_path, threshold=128, min_length=0):
    """
    Trace the input bitmap with Potrace and return segments list,
    each as (Bx, By) expressions, normalized to pixel units,
    centered on (0,0), with upward in image matching upward in Desmos.
    Also returns raw curve segments for preview plotting.
    Optionally filter out segments shorter than min_length.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load '{image_path}'")

    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0

    bmp = potrace.Bitmap(gray < threshold)
    paths = bmp.trace()

    # determine scale
    xs, ys = [], []
    for curve in paths:
        start = curve.start_point
        for seg in curve:
            pts = [start]
            if seg.is_corner:
                pts.append(seg.c)
            else:
                pts.extend([seg.c1, seg.c2])
            pts.append(seg.end_point)
            for pt in pts:
                xs.append(pt.x)
                ys.append(pt.y)
            start = seg.end_point
    scale_x = max(xs) / w if w else 1.0
    scale_y = max(ys) / h if h else 1.0
    scale = (scale_x + scale_y) / 2.0 or 1.0

    desmos_segments = []
    preview_curves = []  # list of Nx2 numpy arrays for preview

    for curve in paths:
        start = curve.start_point
        for seg in curve:
            if seg.is_corner:
                p0, p1, p2, p3 = start, start, seg.c, seg.end_point
            else:
                p0, p1, p2, p3 = start, seg.c1, seg.c2, seg.end_point

            def loc(pt):
                x = (pt.x / scale) - cx
                y = cy - (pt.y / scale)
                return x, y

            pts_pixel = [loc(p) for p in (p0, p1, p2, p3)]
            # filter by length
            (x0, y0), _, _, (x3, y3) = pts_pixel
            length = np.hypot(x3 - x0, y3 - y0)
            if length < min_length:
                start = seg.end_point
                continue

            # store preview control pts
            preview_curves.append(np.array(pts_pixel))

            # build Desmos expressions
            Bx = (
                f"(1-t)^3*{pts_pixel[0][0]}"
                f"+3*(1-t)^2*t*{pts_pixel[1][0]}"
                f"+3*(1-t)*t^2*{pts_pixel[2][0]}"
                f"+t^3*{pts_pixel[3][0]}"
            )
            By = (
                f"(1-t)^3*{pts_pixel[0][1]}"
                f"+3*(1-t)^2*t*{pts_pixel[1][1]}"
                f"+3*(1-t)*t^2*{pts_pixel[2][1]}"
                f"+t^3*{pts_pixel[3][1]}"
            )
            domain = r"\left\{0 \le t \le 1\right\}"
            desmos_segments.append(f"({Bx}, {By}) {domain}")
            start = seg.end_point

    return desmos_segments, preview_curves


def copy_to_clipboard(text):
    root = tk.Tk(); root.withdraw()
    root.clipboard_clear(); root.clipboard_append(text)
    root.update(); root.destroy()


def draw_figure(canvas, curves):
    # create matplotlib figure with larger size for clarity
    fig, ax = plt.subplots(figsize=(6,6))
    for pts in curves:
        # cubic Bézier curve sampling
        t = np.linspace(0,1,50)
        x = (1-t)**3*pts[0,0] + 3*(1-t)**2*t*pts[1,0] + 3*(1-t)*t**2*pts[2,0] + t**3*pts[3,0]
        y = (1-t)**3*pts[0,1] + 3*(1-t)**2*t*pts[1,1] + 3*(1-t)*t**2*pts[2,1] + t**3*pts[3,1]
        ax.plot(x, y, linewidth=1)
    ax.set_aspect('equal', 'box')
    # remove y-axis inversion for natural preview orientation
    ax.axis('off')

    # draw
    fig_canvas = FigureCanvasTkAgg(fig, master=canvas)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill='both', expand=1)
    return fig_canvas


def main():
    sg.theme('LightBlue2')
    layout = [
        [sg.Text('Image:'), sg.Input(key='-FILE-'),
         sg.FileBrowse(file_types=(('PNG/JPG','*.png;*.jpg'),))],
        [sg.Text('Threshold:'), sg.Slider(range=(0,255), default_value=128,
                                           orientation='h', key='-THR-')],
        [sg.Text('Min length:'), sg.Slider(range=(0,100), default_value=0,
                                           orientation='h', key='-MIN-')],
        [sg.Button('Preview'), sg.Button('Convert'), sg.Button('Copy'),
         sg.Button('Save'), sg.Button('Exit')],
        [sg.Column([[sg.Canvas(key='-CANVAS-')]], size=(600,600)),
         sg.Multiline(key='-OUT-', size=(60,20), font=('Courier',10))]
    ]
    window = sg.Window('Desmos Bézier Exporter', layout, finalize=True)
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    preview_plot = None

    while True:
        event, vals = window.read()
        if event in (None, 'Exit'):
            break
        if event == 'Preview':
            path = vals['-FILE-']; thr = int(vals['-THR-']); mn = int(vals['-MIN-'])
            if not path:
                sg.popup_error('Select an image first!'); continue
            try:
                segments, curves = bitmap_to_desmos_beziers(path, threshold=thr, min_length=mn)
                # clear old
                if preview_plot:
                    preview_plot.get_tk_widget().forget()
                preview_plot = draw_figure(canvas, curves)
            except Exception as e:
                sg.popup_error(f'Error:\n{e}')
        elif event == 'Convert':
            path = vals['-FILE-']; thr = int(vals['-THR-']); mn = int(vals['-MIN-'])
            if not path:
                sg.popup_error('Select an image!'); continue
            try:
                segments, _ = bitmap_to_desmos_beziers(path, threshold=thr, min_length=mn)
                text = "\n".join(segments)
                window['-OUT-'].update(text)
            except Exception as e:
                sg.popup_error(f'Error:\n{e}')
        elif event == 'Copy':
            txt = vals['-OUT-'];
            if txt.strip(): copy_to_clipboard(txt); sg.popup_ok('Copied!')
            else: sg.popup_error('Nothing to copy.')
        elif event == 'Save':
            txt = vals['-OUT-']
            if not txt.strip(): sg.popup_error('Nothing to save.'); continue
            fp = sg.popup_get_file('Save', save_as=True, no_window=True,
                                   default_extension='.txt', file_types=(('Text','*.txt'),))
            if fp:
                with open(fp,'w') as f: f.write(txt)
                sg.popup_ok(f'Saved to:\n{fp}')
    window.close()

if __name__ == '__main__':
    main()
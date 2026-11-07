#!/usr/bin/env python3
"""Label native screenshots; retain all unmodified originals beside the sheets."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
from photograph_geometric_closure import CHOICES

OUT = ROOT / 'artifacts/geometric-closure-photos-clean'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def marked_image(report, stage, view):
    record = next(s for s in report['stages'] if s['stage'] == stage)
    camera = next(im for im in record['images'] if Path(im['path']).stem == stage+'-'+view)
    picture = Image.open(ROOT / camera['path']).convert('RGB')
    # A projected outline at the TCP plane marks the finite evaluation window.
    # It is an annotation, not a rendered cloth patch or an occlusion test.
    r = report['candidate']
    R = np.asarray(r['candidate']['rotation_matrix'])
    p = np.asarray(r['candidate']['grasp_center_unity_world_m'])
    position = np.asarray(camera['position'])
    forward = np.asarray(camera['target'])-position
    forward /= np.linalg.norm(forward)
    right = np.cross(R[:, 1], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    scale = picture.height/(2*np.tan(np.radians(camera['fov_deg'])/2))
    def project(point):
        delta = point-position
        return (picture.width/2+scale*(delta@right)/(delta@forward),
                picture.height/2-scale*(delta@up)/(delta@forward))
    corners = [project(p+R[:, 0]*x+R[:, 2]*z)
               for x, z in [(-.004, -.004), (.004, -.004), (.004, .004), (-.004, .004)]]
    draw = ImageDraw.Draw(picture)
    # Oblique views show the window as a quadrilateral; near-edge-on views
    # naturally collapse it to a line. Do not inflate its projected height.
    draw.line(corners+[corners[0]], fill='#ff00cc', width=3)
    x, y = project(p)
    draw.line([(x-10, y), (x+10, y)], fill='#ff00cc', width=3)
    draw.line([(x, y-10), (x, y+10)], fill='#ff00cc', width=3)
    return picture


def sheet(kind, columns):
    width, height = 560, 440
    row_header = 64
    header = 110
    canvas = Image.new('RGB', (width*len(columns), header+(height+row_header)*len(CHOICES)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 17)
    draw.text((12, 7), '青：布 ／ 肌色：足 ／ 橙・緑：指 ／ 灰：手のひら', fill='black', font=font)
    draw.text((12, 36), '紫：評価窓8×8mmのTCP面への投影と把持中心（注釈）', fill='black', font=small)
    draw.text((12, 61), '実メッシュの静止配置。閉鎖時の布変形なし・貫通あり。太い表示用の縁は非表示。', fill='black', font=small)
    for col, (_, _, label) in enumerate(columns):
        draw.text((col*width+12, 85), label, fill='black', font=font)
    for row, (number, index) in enumerate(CHOICES):
        report = json.loads((OUT / f'pose-{number:02d}-candidate-{index:03d}/report.json').read_text())
        r = report['candidate']
        inner = '橙' if r['insertion']['inner']['link'].endswith('/right_finger_link') else '緑'
        top = header+row*(height+row_header)
        for col, (stage, view, _) in enumerate(columns):
            draw.text((col*width+12, top+3), f"No.{number} 候補{index} / pitch {report['pitch_deg']:g}°, roll {report['roll_deg']:g}°", fill='black', font=small)
            draw.text((col*width+12, top+29), f"内側の指：{inner} / 閉鎖面積 {r['projection']['area_m2']*1e6:.3f} mm²", fill='black', font=small)
            im = marked_image(report, stage, view).resize((width, height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            canvas.paste(im, (col*width, top+row_header))
    canvas.save(OUT / f'{kind}.jpg', quality=94)


if __name__ == '__main__':
    sheet('stages', [('approach-start', 'overview', '接近前（50mm手前）'),
                     ('open', 'overview', '接近完了（開状態）'),
                     ('closed', 'overview', '閉鎖位置（隙間1mm）')])
    sheet('closed-details', [('closed', 'gap-side', '閉鎖位置：側面'),
                             ('closed', 'gap-opposite', '閉鎖位置：反対側')])
    for number, index in CHOICES:
        report = json.loads((OUT / f'pose-{number:02d}-candidate-{index:03d}/report.json').read_text())
        images = [marked_image(report, stage, 'gap-opposite').resize((560, 440))
                  for stage in ('approach-start', 'open', 'closed')]
        single = Image.new('RGB', (1680, 480), 'white')
        draw = ImageDraw.Draw(single)
        for col, (image, label) in enumerate(zip(images, ['接近前', '接近完了', '閉鎖位置：1mm'])):
            draw.text((col*560+12, 5), f'No.{number} 候補{index} ／ {label}', fill='black', font=ImageFont.truetype(FONT, 22))
            single.paste(image, (col*560, 40))
        single.save(OUT / f'pose-{number:02d}-stages.jpg', quality=95)

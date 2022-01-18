import os
import cv2
import shutil
import numpy as np
from lxml import etree


def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def parse_anno_file(cvat_xml, frame):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    #remove png to name
    image_poly_frame_attr = ".//polygon[@frame='{}']".format(frame)
    image_box_frame_attr = ".//box[@frame='{}']".format(frame)
    width = int(root.find(".//width").text)
    height = int(root.find(".//height").text)
    shapes = []
    for child in root.iterfind(image_poly_frame_attr):
        shapes.append(child)
    for child in root.iterfind(image_box_frame_attr):
        shapes.append(child)
    for image_tag in shapes:
        image = {"width": width, "height": height, 'shapes': []}
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            for key, value in box_tag.items():
                box[key] = value
            box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                box['xtl'], box['ytl'], box['xbr'], box['ybr'])
            image['shapes'].append(box)
        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)
    return anno


def create_mask_file(width, height, background, shapes):
    mask = np.full((height, width), background, dtype=np.uint8)
    for shape in shapes:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points.astype(int)
        mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=1)
        mask = cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return mask


def getMaskForFrame(path, frame):
    anno = parse_anno_file(path, frame)
    shapes = []
    for shape in anno:
        shapes.append(shape['shapes'][0])
    height = anno[0]['height']
    width = anno[0]['width']
    background = np.zeros((height, width), dtype=np.uint8)
    background = create_mask_file(width, height, background, shapes)
    mask = background
    mask = cv2.resize(mask, (640, 350))
    return mask

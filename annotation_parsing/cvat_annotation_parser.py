import os
import cv2
import argparse
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm


def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)

#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         fromfile_prefix_chars='@',
#         description='Convert CVAT XML annotations to contours'
#     )
#     parser.add_argument(
#         '--image-dir', metavar='DIRECTORY', required=True,
#         help='directory with input images'
#     )
#     parser.add_argument(
#         '--cvat-xml', metavar='FILE', required=True,
#         help='input file with CVAT annotation in xml format'
#     )
#     parser.add_argument(
#         '--output-dir', metavar='DIRECTORY', required=True,
#         help='directory for output masks'
#     )
#     parser.add_argument(
#         '--scale-factor', type=float, default=1.0,
#         help='choose scale factor for images'
#     )
#     return parser.parse_args()


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
        image = {}
        image["width"] = width
        image["height"] = height
        image['shapes'] = []
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


def create_mask_file(width, height, bitness, background, shapes, scale_factor):
    mask = np.full((height, width, bitness // 8), background, dtype=np.uint8)
    for shape in shapes:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points * scale_factor
        points = points.astype(int)
        mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=5)
        mask = cv2.fillPoly(mask, [points], color=(0, 0, 255))
    return mask


def getMaskForFrame(path, frame, scale_factor=1):

    mask_bitness = 24
    masks = []

    #img_path = os.path.join(args.image_dir, img)
    anno = parse_anno_file(path, frame)
    shapes = []
    for shape in anno:
        shapes.append(shape['shapes'][0])
    background = []
    height = anno[0]['height']
    width = anno[0]['width']
    background = np.zeros((height, width, 3), np.uint8)
    # print(shapes)
    background = create_mask_file(width,
                                  height,
                                  mask_bitness,
                                  background,
                                  shapes,
                                  scale_factor)
    mask = background
    #print(background)
    # savedir = '\\'.join(path.split("\\")[0:-1]) + "\\" + 'masks'
    # dir_create(savedir)
    # name = path.split("\\")[-1][:-4]
    # loc = savedir+'\\'+name
    # saveloc = loc+(str(frame))+(".png")
    # print(saveloc)
    # cv2.imwrite(saveloc, background)
    # print(mask.shape)
    return mask

#getMaskForFrame('B:\\Desktop\\FER\\DIPL_1\\RaspoznavanjeUzoraka\\haralick-features-detector\\dataset\\video_sekvence\\GT\\gettyimages-583944852-640_adpp.xml', 1)
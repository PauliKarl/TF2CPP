import cv2
import glob
import hashlib
import io
import json
import numpy as np
import os
import PIL.Image
import tensorflow as tf
import logging
from object_detection.utils import label_map_util
from pycocotools import mask as maskUtils
import xml.etree.ElementTree as ET



flags = tf.app.flags
flags.DEFINE_string('root_dir', None, 'Absolute path to images_dir and annotation xml files.')
flags.DEFINE_string('label_map_path', None, 'Path to label map proto.')
flags.DEFINE_string('output_path', None, 'Path to the output tfrecord.')

FLAGS = flags.FLAGS

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(annotation_dict, label_map_dict = None):
    """Convert images and annotations to a tf.Example proto.

    Args:
        annotation_dict:A dictionary containing the following keys:
            ['height', 'width', 'filename', 'sha256_key' 'encoded_png',
            'format', 'xmins', 'xmaxs', 'ymins', 'ymaxs', 'masks',
            class_names'].
        label_map_dict: A dictionary mapping class_names to indices.
    
    Returns:
        example: The coverted tf.Example.
    
    Raises:
        ValueError: if label_map_dict is None or is not containing a class_names.
    """
    if annotation_dict is None:
        return None
    if label_map_dict is None:
        raise ValueError('label_map_dict is None')

    height = annotation_dict.get('height', None)
    width = annotation_dict.get('width', None)
    filename = annotation_dict.get('filename', None)
    sha256_key = annotation_dict.get('sha256_key', None)
    encoded_png = annotation_dict.get('encode_png', None)
    image_format = annotation_dict.get('format', None)

    xmins = annotation_dict.get('xmins', None)
    xmaxs = annotation_dict.get('xmaxs', None)
    ymins = annotation_dict.get('ymins', None)
    ymaxs = annotation_dict.get('ymaxs', None)
    masks = annotation_dict.get('masks', None)
    class_names = annotation_dict.get('class_names', None)

    #print("class_names:", class_names)
    labels = []

    for class_name in class_names:
        label = label_map_dict.get(class_name, 'None')
        #print("label", label)
        if label is None:
            raise ValueError('`label_map_dict` is not containing {}.'.format(class_name))
        labels.append(label)
    
    encoded_masks = []
    for mask in masks:
        pil_image = PIL.Image.fromarray(mask.astype(np.uint8))
        output_io = io.BytesIO()
        pil_image.save(output_io, format='PNG')
        encoded_masks.append(output_io.getvalue())

    feature_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(filename.encode('utf8')),
        'image/key/sha256': bytes_feature(sha256_key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_png),
        'image/format': bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/mask': bytes_list_feature(encoded_masks),
        'image/object/class/label': int64_list_feature(labels)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example

def _get_annotation_dict(img_path, label_path):
    """Get boundingboxes and masks.

    Args:
        img_path: path to image.
        label_path: path to annoataed xml file corresponding to the image. 
    
    Returns:
        annotation_dict:A dictionary contaning the following key:
            ['height', 'width', 'filename', 'sha256_key', 'encode_png', 'format',
            'xmins', 'xmaxs', 'ymins', 'ymaxs', 'masks', 'class_names'].
    """
    img_format = img_path.split('.')[-1]
    if (not os.path.exists(img_path) or not os.path.exists(label_path)):
        return None
    
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encode_png = fid.read()
    key = hashlib.sha256(encode_png).hexdigest()

    tree = ET.parse(label_path)
    root = tree.getroot()
    objects = []

    imgsize=root.find('size')
    width = int(imgsize.find('width').text)
    height = int(imgsize.find('height').text)

    filename = str(root.find('filename').text)

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    masks = []
    class_names = []
    # 这里的classname是类别名称还是类别id有待进一步确认
    boxes = []
    thetaobbes = []
    pointobbes = []
    for single_object in root.findall('object'):
        robndbox = single_object.find('robndbox')
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w  = float(robndbox.find('w').text)
        h  = float(robndbox.find('h').text)
        theta = float(robndbox.find('angle').text)
        thetaobb = [cx,cy,w,h,theta]

        cls_name = str(single_object.find('name').text)

        box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4] * 180.0 / np.pi))
        box = np.reshape(box, [-1, ]).tolist()
        pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

        xmin = min(pointobb[0::2])
        ymin = min(pointobb[1::2])
        xmax = max(pointobb[0::2])
        ymax = max(pointobb[1::2])
        bbox = [xmin, ymin, xmax, ymax]

        thetaobbes.append(thetaobb)
        pointobbes.append(pointobb)
        boxes.append(bbox)

        segm = [pointobb]            
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)

        class_names.append(cls_name)
        xmins.append(max(0, min(1.0, float(xmin) / width)))
        xmaxs.append(max(0, min(1.0, float(xmax) / width)))
        ymins.append(max(0, min(1.0, float(ymin) / height)))
        ymaxs.append(max(0, min(1.0, float(ymax) / height)))
        masks.append(mask)   

    annotation_dict = { 'height': height,
                        'width': width,
                        'filename': filename,
                        'sha256_key': key,
                        'encode_png': encode_png,
                        'format': img_format,
                        'xmins': xmins,
                        'xmaxs': xmaxs,
                        'ymins': ymins,
                        'ymaxs': ymaxs,
                        'masks': masks,
                        'class_names': class_names
    }
    return annotation_dict

def main(_):
    if not os.path.exists(FLAGS.root_dir):
        raise ValueError('`root_dir` is not exist')
    if not os.path.exists(FLAGS.label_map_path):
        raise ValueError('`label_map_path` is not exist')
    
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    root_dir = FLAGS.root_dir
    label_map_path = FLAGS.label_map_path
    output_path = os.path.join(FLAGS.output_path, 'gf3trainval.record')

    img_dir = root_dir+ '/images'
    ann_dir = root_dir + '/labelxmls'

    imgs = list(sorted(os.listdir(img_dir)))
    labels = list(sorted(os.listdir(ann_dir)))

    label_map = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    writer = tf.io.TFRecordWriter(output_path)

    num_annotations_skiped = 0
    for img, label in zip(imgs,labels):
        img_path = img_dir + '/' + img
        label_path = ann_dir + '/' + label
        print(img, label)

        annotation_dict = _get_annotation_dict(img_path, label_path)
        if annotation_dict is None:
            num_annotations_skiped += 1
            continue
        tf_example = create_tf_example(annotation_dict, label_map)
        writer.write(tf_example.SerializeToString())
    
    print('Successfully created TFRecord to {}.'.format(output_path))

if __name__ == '__main__':
    tf.app.run()
    
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import numpy as np
import logging
import time
import base64
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
import cv2
from PIL import Image

bird_dict = {'0': '大紫胸鹦鹉', '1': '葵花鹦鹉', '2': '亚历山大鹦鹉', '3': '啄羊鹦鹉', '4': '棕树凤头鹦鹉', '5': '虎皮鹦鹉', '6': '彼氏鹦鹉', '7': '小葵花凤头鹦鹉', '8': '红肩金刚鹦鹉', '9': '蓝眼凤头鹦鹉', '10': '红尾凤头鹦鹉', '11': '白顶啄羊鹦鹉', '12': '橙冠凤头鹦鹉', '13': '花头鹦鹉', '14': '白凤头鹦鹉', '15': '澳东玫瑰鹦鹉', '16': '红顶鹦鹉', '17': '红腰鹦鹉', '18': '鸡尾鹦鹉', '19': '红翅鹦鹉', '20': '和尚鹦鹉', '21': '长嘴凤头鹦鹉', '22': '澳洲王鹦鹉', '23': '黑脸牡丹鹦鹉', '24': '黑凤头鹦鹉', '25': '红蓝鹦鹉', '26': '红额鹦鹉', '27': '红玫瑰鹦鹉', '28': '马岛鹦鹉', '29': '紫蓝金刚鹦鹉', '30': '戈氏凤头鹦鹉', '31': '红脸鹦鹉', '32': '烟色鹦鹉', '33': '红胁绿鹦鹉', '34': '绯胸鹦鹉', '35': '桃脸牡丹鹦鹉', '36': '小凤头鹦鹉', '37': '黑顶鹦鹉', '38': '红腹金刚鹦鹉', '39': '红冠灰凤头鹦鹉', '40': '粉红凤头鹦鹉', '41': '红腹鹦鹉', '42': '彩冠凤头鹦鹉', '43': '红领绿鹦鹉', '44': '非洲灰鹦鹉', '45': '大绿金刚鹦鹉'}
class DecodeImage(object):
    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb

    def __call__(self, img):
        img = base64.b64decode(img)
        data = np.fromstring(img, dtype='uint8')
        # print(data)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        if self.to_rgb:
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
                img.shape)
            img = img[:, :, ::-1]

        return img


class ResizeImage(object):
    def __init__(self, resize_short=None):
        self.resize_short = resize_short

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        percent = float(self.resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        return cv2.resize(img, (w, h))


class CropImage(object):
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None):
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        return (img.astype('float32') * self.scale - self.mean) / self.std


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return img


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    class Arg():
        def __init__(self):
            self.model_file = "/usr/soft/bird_recognize_server/recognize/bird_m/model"
            self.params_file = "/usr/soft/bird_recognize_server/recognize/bird_m/params"
            self.batch_size = 1
            self.use_fp16 = False
            self.use_gpu = False
            self.ir_optim = True
            self.use_tensorrt = False
            self.gpu_mem = 8000
            self.enable_benchmark = False
            self.model_name = "birdrec"
    #print(Arg().model_file)
    return Arg()


def create_predictor(args):
    config = AnalysisConfig(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()

    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=AnalysisConfig.Precision.Half
            if args.use_fp16 else AnalysisConfig.Precision.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_paddle_predictor(config)

    return predictor


def create_operators():
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = DecodeImage()
    resize_op = ResizeImage(resize_short=256)
    crop_op = CropImage(size=(size, size))
    normalize_op = NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = ToTensor()

    return [decode_op, resize_op, crop_op, normalize_op, totensor_op]


def preprocess(base64_str, ops, width = 224,height = 224):
    
    #data = io.BytesIO(base64.b64decode(base64_str))
    #img = Image.open(data)
    #new_img = img.resize((width, height), Image.BILINEAR)
    #if new_img.mode == 'P':
    #    new_img = new_img.convert("RGB")
    #if new_img.mode == 'RGBA':
    #    new_img = new_img.convert("RGB")
    #data = new_img.tobytes()
    data = base64_str
    for op in ops:
        data = op(data)

    return data


def main(image_file):
    args = parse_args()

    if not args.enable_benchmark:
        assert args.batch_size == 1
        assert args.use_fp16 == False
    else:
        assert args.use_gpu == True
        assert args.model_name is not None
        assert args.use_tensorrt == True
    # HALF precission predict only work when using tensorrt
    if args.use_fp16 == True:
        assert args.use_tensorrt == True

    operators = create_operators()
    predictor = create_predictor(args)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])

    test_num = 500
    test_time = 0.0
    if not args.enable_benchmark:
        inputs = preprocess(image_file, operators)
        inputs = np.expand_dims(
            inputs, axis=0).repeat(
                args.batch_size, axis=0).copy()
        input_tensor.copy_from_cpu(inputs)

        predictor.zero_copy_run()

        output = output_tensor.copy_to_cpu()
        output = output.flatten()
        cls = bird_dict[str(np.argmax(output))]
        score = output[np.argmax(output)]
        # logger.info("class: {0}".format(cls))
        # logger.info("score: {0}".format(score))
        return {"name":cls,"score":str(score)}
    else:
        for i in range(0, test_num + 10):
            inputs = np.random.rand(args.batch_size, 3, 224,
                                    224).astype(np.float32)
            start_time = time.time()
            input_tensor.copy_from_cpu(inputs)

            predictor.zero_copy_run()

            output = output_tensor.copy_to_cpu()
            output = output.flatten()
            if i >= 10:
                test_time += time.time() - start_time
            cls = bird_dict[str(np.argmax(output))]
            score = output[np.argmax(output)]
            # logger.info("class: {0}".format(cls))
            # logger.info("score: {0}".format(score))
            return {"name":cls,"score":str(score)}

        # fp_message = "FP16" if args.use_fp16 else "FP32"
        # logger.info("{0}\t{1}\tbatch size: {2}\ttime(ms): {3}".format(
        #     args.model_name, fp_message, args.batch_size, 1000 * test_time /
        #     test_num))
        

# if __name__ == "__main__":
#     rootdir = "/home/aistudio/work/PaddleClas/dataset/flowers102/"
#     imgli = []
#     suma = 0
#     score = 0
#     with open("//home/aistudio/work/PaddleClas/dataset/flowers102/val_list.txt","r") as f:
#         for line in f.readlines():
#             suma += 1
#             imgdir = rootdir + line.split(" ")[0]
#             imgcls = line.split(" ")[1]
#             preimgcls = main(imgdir)
#             if imgcls.replace("\n","") == str(list(preimgcls)[0]):
#                 score += 1
#             print(str(100 * score/suma) + "%")

    # with open("1.jpg","rb") as f:  # 二进制方式打开图文件
    #     base64_str = base64.b64encode(f.read())
    #     print(main(base64_str))

# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# Author: Lintao
# Created: 2020/03/30
# --------------------------------------------------------
import torch
import onnx


def torch_2_onnx(model, x, input_names, output_names, save_path):
    if save_path is not None:
        torch.onnx.export(model, x, save_path,
                          export_params=True,
                          verbose=True,
                          input_names=input_names,
                          output_names=output_names)

        # Checks
        onnx_model = onnx.load(save_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print("===========================Model Graph===========================")
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model


def test_1():
    from torchvision.models.resnet import resnet18
    model = resnet18(num_classes=5)
    input_names = ['input']
    output_names = ['output']
    batch_size = 1
    model = model.cuda()
    x = torch.ones((batch_size, 3, 256, 256)).float().cuda()

    save_path = 'model.onnx'
    torch_2_onnx(model, x, input_names, output_names, save_path)


def test_2():
    from models.yolo import Model
    device = 'cuda:0'
    model_path = '/home/lintao/jobs/logo/yolov5-4.0/yolov5s.pt'

    # model = torch.load(model_path)['model'].float()
    model = Model('/home/lintao/jobs/logo/yolov5-4.0/models/yolov5s.yaml').eval()
    model.to(device)
    model.model[-1].export = True
    x = torch.ones((1, 3, 640, 640)).float().cuda()
    y = model(x)
    save_path = model_path.replace('.pt', '.onnx')  # filename
    torch_2_onnx(model, x, input_names=['images'], output_names=['classes'], save_path=save_path)


if __name__ == '__main__':
    test_2()

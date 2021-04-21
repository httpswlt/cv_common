# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 4/21/21 4:09 PM
# @Version	1.0
# --------------------------------------------------------
import os

os.sys.path.insert(0, '/home/lintao/jobs/torch2trt')
from torch2trt.torch2trt import *

"""
    You should download the torch2trt code from https://github.com/NVIDIA-AI-IOT/torch2trt.git

"""


def build_tensorRT_model(module,
                         inputs,
                         input_names=None,
                         output_names=None,
                         log_level=trt.Logger.ERROR,
                         max_batch_size=1,
                         fp16_mode=False,
                         max_workspace_size=1 << 25,
                         strict_type_constraints=False,
                         int8_mode=False,
                         calibrator=None,
                         int8_calib_dataset=None,
                         int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
                         int8_calib_batch_size=1,
                         engine_name='./model.engine'):
    inputs_in = inputs

    # copy inputs to avoid modifications to source data
    inputs = [tensor.clone()[0:1] for tensor in inputs]  # only run single entry

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)

    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # run once to get num outputs
    outputs = module(*inputs)
    if not isinstance(outputs, tuple) and not isinstance(outputs, list):
        outputs = (outputs,)

    if input_names is None:
        input_names = default_input_names(len(inputs))
    if output_names is None:
        output_names = default_output_names(len(outputs))

    network = builder.create_network()
    with ConversionContext(network) as ctx:

        ctx.add_inputs(inputs, input_names)

        outputs = module(*inputs)

        if not isinstance(outputs, tuple) and not isinstance(outputs, list):
            outputs = (outputs,)
        ctx.mark_outputs(outputs, output_names)

    builder.max_workspace_size = max_workspace_size
    builder.fp16_mode = fp16_mode
    builder.max_batch_size = max_batch_size
    builder.strict_type_constraints = strict_type_constraints

    if int8_mode:
        builder.int8_mode = True

        if calibrator is None:
            # default to use input tensors for calibration
            if int8_calib_dataset is None:
                int8_calib_dataset = TensorBatchDataset(inputs_in)
            calibrator = DatasetCalibrator(inputs, int8_calib_dataset, batch_size=int8_calib_batch_size,
                                           algorithm=int8_calib_algorithm)

        builder.int8_calibrator = calibrator

    engine = builder.build_cuda_engine(network)

    os.makedirs(os.path.abspath(os.path.dirname(engine_name)), exist_ok=True)
    with open(engine_name, "wb") as f:
        f.write(engine.serialize())
    return input_names, output_names


def load_tensorRT_model(engine_path, input_names, output_names):
    """

    :param engine_path:
    :param input_names:
    :param output_names:
    :return:
    """
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    model = TRTModule(engine, input_names, output_names)
    return model


def test_float32():
    from torchvision.models.resnet import resnet18
    engine_name = 'model_f32.engine'
    input_names = ['input']
    output_names = ['output']
    print('Build Engine Model.')
    batch_size = 1
    x = torch.ones((batch_size, 3, 256, 256)).cuda()
    model = resnet18(num_classes=5).cuda().eval()
    input_names, output_names = build_tensorRT_model(model, [x],
                                                     input_names=input_names, output_names=output_names,
                                                     int8_mode=False, engine_name=engine_name)
    model = load_tensorRT_model(engine_name, input_names, output_names)
    model(x)


def test_int8():
    from torchvision.models.resnet import resnet18
    from calibration import ImageCalibrator

    imgs_dir = '/mnt/data/sample_comparison/sample_comp/train_ori'
    images = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)[:3]]
    calibrator = ImageCalibrator(images, width=256, height=256, channel=3, batch_size=1,
                                 cache_file='./{}.cache'.format('int8'))

    engine_name = 'model_i8.engine'
    input_names = ['input']
    output_names = ['output']
    print('Build Engine Model.')
    batch_size = 1
    x = torch.ones((batch_size, 3, 256, 256)).cuda()
    model = resnet18(num_classes=5).cuda().eval()
    input_names, output_names = build_tensorRT_model(model, [x],
                                                     input_names=input_names, output_names=output_names,
                                                     int8_mode=True, calibrator=calibrator, engine_name=engine_name)
    model = load_tensorRT_model(engine_name, input_names, output_names)
    model(x)


if __name__ == '__main__':
    # test_float32()
    test_int8()

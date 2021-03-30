# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# Author: Lintao
# Created: 2020/03/30
# --------------------------------------------------------
import tensorrt as trt
import torch


def torch_dtype_to_trt(dtype):
    if dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError('%s is not supported by tensorrt' % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


class TrtModel(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        """

        :param engine: engine file or engine object, recommend: use the engine file rather than engine object.
        :param input_names: e.g: ['input']
        :param output_names: e.g: ['output']

        Note: you should comparison the consistency of the result \
                between the engine file and engine object when you use the engine object.
        """
        super(TrtModel, self).__init__()
        if isinstance(engine, str):
            self.engine = self.describe_engine_file(engine)
        else:
            self.engine = engine

        self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names

    @staticmethod
    def describe_engine_file(engine_path):
        """

        :param engine_path:
        :return:
        """
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].data_ptr()

        self.context.execute_async(batch_size, bindings, torch.cuda.current_stream().cuda_stream)

        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs


def onnx2trt(args, calib=None):
    """
        convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
        :param args: e.g:
            args = dict(
                        onnx_file_path='model.onnx',
                        mode='fp32',
                        engine_file_path='test.engine',
                        batch_size=1,
                        channel=3,
                        height=256,
                        width=256,
                        channel_last=False
                        )
        :parm calib:
        :return: trt engine
    """

    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"
    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, logger) as parser:
        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            # b = parser.parse(model.read())
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = args.batch_size

        if args.mode.lower() == 'int8':
            assert builder.platform_has_fast_int8, "not support int8"
            assert calib is not None, "Please add the calibrate datasets."
            builder.int8_mode = True
            builder.int8_calibrator = calib
            print("using calibration----------")
        elif args.mode.lower() == 'fp16':
            assert builder.platform_has_fast_fp16, "not support fp16"
            builder.fp16_mode = True

        print("num layers:", network.num_layers)
        last_layer = network.get_layer(network.num_layers - 1)
        if not last_layer.get_output(0):
            network.mark_output(
                network.get_layer(network.num_layers - 1).get_output(0))
        if args.channel_last:
            network.get_input(0).shape = [args.batch_size, args.height, args.width, args.channel]
        else:
            network.get_input(0).shape = [args.batch_size, args.channel, args.height, args.width]

        print(network.get_input(0).name)
        engine = builder.build_cuda_engine(network)
        with open(args.engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("engine:", engine)
        print("Create Engine Success, Save Path: {}".format(args.engine_file_path))
        return engine


def test_1():
    from argparse import Namespace
    args = dict(
        onnx_file_path='model.onnx',
        mode='fp16',
        engine_file_path='test.engine',
        batch_size=1,
        channel=3,
        height=256,
        width=256,
        channel_last=False
    )
    args = Namespace(**args)
    onnx2trt(args, None)


def test_2():
    engine_file = 'test.engine'
    input_names = ['input']
    output_names = ['output']
    model = TrtModel(engine_file, input_names, output_names)
    x = torch.ones((1, 3, 255, 255)).cuda().float()
    print(model(x))


if __name__ == '__main__':
    # test_1()
    test_2()

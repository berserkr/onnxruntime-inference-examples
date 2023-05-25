import os
from pathlib import Path

from onnxruntime.quantization import create_calibrator, write_calibration_table, CalibrationMethod
from onnxruntime.quantization import CalibrationMethod, QuantType, QuantizationMode, QDQQuantizer
from onnxruntime.quantization import quantize_static, quantize_dynamic
import onnx

from data_reader import YoloV3DataReader, YoloV3VariantDataReader, TOFADataReader
from preprocessing import yolov3_preprocess_func, yolov3_preprocess_func_2, yolov3_variant_preprocess_func, yolov3_variant_preprocess_func_2, tofa_preprocess_func
from evaluate import YoloV3Evaluator, YoloV3VariantEvaluator,YoloV3Variant2Evaluator, YoloV3Variant3Evaluator, TOFAEvaluator


def get_calibration_table(model_path, augmented_model_path, calibration_dataset):

    calibrator = create_calibrator(model_path, None, augmented_model_path=augmented_model_path)

    # DataReader can handle dataset with batch or serial processing depends on its implementation
    # Following examples show two different ways to generate calibration table
    '''
    1. Use serial processing
    
    We can use only one DataReader to do serial processing, however,
    some machines don't have sufficient memory to hold all dataset images and all intermediate output.
    So let multiple DataReader do handle different stride of dataset one by one.
    DataReader will use serial processing when batch_size is 1.
    '''

    total_data_size = len(os.listdir(calibration_dataset))
    start_index = 0
    stride = 2000
    for i in range(0, total_data_size, stride):
        data_reader = YoloV3DataReader(calibration_dataset,
                                       start_index=start_index,
                                       end_index=start_index + stride,
                                       stride=stride,
                                       batch_size=1,
                                       model_path=augmented_model_path)
        calibrator.collect_data(data_reader)
        start_index += stride
    '''
    2. Use batch processing (much faster)
    
    Batch processing requires less memory for intermediate output, therefore let only one DataReader to handle dataset in batch. 
    However, if encountering OOM, we can make multiple DataReader to do the job just like serial processing does. 
    DataReader will use batch processing when batch_size > 1.
    '''

    # data_reader = YoloV3DataReader(calibration_dataset, stride=1000, batch_size=20, model_path=augmented_model_path)
    # calibrator.collect_data(data_reader)

    write_calibration_table(calibrator.compute_range())
    print('calibration table generated and saved.')


def get_prediction_evaluation(model_path, validation_dataset, providers):
    data_reader = YoloV3DataReader(validation_dataset,
                                   stride=1000,
                                   batch_size=1,
                                   model_path=model_path,
                                   is_evaluation=True)
    evaluator = YoloV3Evaluator(model_path, data_reader, providers=providers)

    evaluator.predict()
    result = evaluator.get_result()

    annotations = './annotations/instances_val2017.json'
    evaluator.evaluate(result, annotations)


def get_calibration_table_yolov3_variant(model_path, augmented_model_path, calibration_dataset):

    calibrator = create_calibrator(model_path, [], augmented_model_path=augmented_model_path, calibrate_method=CalibrationMethod.Entropy)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])

    # DataReader can handle dataset with batch or serial processing depends on its implementation
    # Following examples show two different ways to generate calibration table
    '''
    1. Use serial processing
    
    We can use only one data reader to do serial processing, however,
    some machines don't have sufficient memory to hold all dataset images and all intermediate output.
    So let multiple data readers to handle different stride of dataset one by one.
    DataReader will use serial processing when batch_size is 1.
    '''

    width = 608
    height = 608

    total_data_size = len(os.listdir(calibration_dataset))
    start_index = 0
    stride = 20 
    batch_size = 1
    for i in range(0, total_data_size, stride):
        data_reader = YoloV3VariantDataReader(calibration_dataset,
                                              width=width,
                                              height=height,
                                              start_index=start_index,
                                              end_index=start_index + stride,
                                              stride=stride,
                                              batch_size=batch_size,
                                              model_path=augmented_model_path)
        calibrator.collect_data(data_reader)
        start_index += stride
    '''
    2. Use batch processing (much faster)
    
    Batch processing requires less memory for intermediate output, therefore let only one data reader to handle dataset in batch. 
    However, if encountering OOM, we can make multiple data reader to do the job just like serial processing does. 
    DataReader will use batch processing when batch_size > 1.
    '''

    # batch_size = 20
    # stride=1000
    # data_reader = YoloV3VariantDataReader(calibration_dataset,
                                          # width=width,
                                          # height=height,
                                          # stride=stride,
                                          # batch_size=batch_size,
                                          # model_path=augmented_model_path)
    # calibrator.collect_data(data_reader)

    write_calibration_table(calibrator.compute_range())
    print('calibration table generated and saved.')


def get_prediction_evaluation_yolov3_variant(model_path, validation_dataset, providers):
    width = 608 
    height = 608 
    evaluator = YoloV3VariantEvaluator(model_path, None, width=width, height=height, providers=providers)

    total_data_size = len(os.listdir(validation_dataset)) 
    start_index = 0
    stride=1000
    batch_size = 1
    for i in range(0, total_data_size, stride):
        data_reader = YoloV3VariantDataReader(validation_dataset,
                                              width=width,
                                              height=height,
                                              start_index=start_index,
                                              end_index=start_index+stride,
                                              stride=stride,
                                              batch_size=batch_size,
                                              model_path=model_path,
                                              is_evaluation=True)

        evaluator.set_data_reader(data_reader)
        evaluator.predict()
        start_index += stride


    result = evaluator.get_result()
    annotations = './annotations/instances_val2017.json'
    evaluator.evaluate(result, annotations)

def get_op_nodes_not_followed_by_specific_op(model, op1, op2):
    op1_nodes = []
    op2_nodes = []
    selected_op1_nodes = []
    not_selected_op1_nodes = []

    for node in model.graph.node:
        if node.op_type == op1:
            op1_nodes.append(node)
        if node.op_type == op2:
            op2_nodes.append(node)

    for op1_node in op1_nodes:
        for op2_node in op2_nodes:
            if op1_node.output == op2_node.input:
                selected_op1_nodes.append(op1_node.name)
        if op1_node.name not in selected_op1_nodes:
            not_selected_op1_nodes.append(op1_node.name)

    return not_selected_op1_nodes

def get_calibration_table_tofa(model_path, augmented_model_path, calibration_dataset):

    op_types_to_quantize = ['MatMul', 'Add']
    calibrator = create_calibrator(model_path, op_types_to_quantize, augmented_model_path=augmented_model_path, calibrate_method=CalibrationMethod.Entropy)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])

    width = 384
    height = 384

    total_data_size = min(1000,len(os.listdir(calibration_dataset))) # limit it to 1000 samples for now...
    start_index = 0
    stride = 500
    batch_size = 1
    for i in range(0, total_data_size, stride):
        data_reader = TOFADataReader(calibration_dataset,
                                              width=width,
                                              height=height,
                                              start_index=start_index,
                                              end_index=start_index + stride,
                                              stride=stride,
                                              batch_size=batch_size,
                                              model_path=augmented_model_path)
        calibrator.collect_data(data_reader)
        start_index += stride

    range_dict = calibrator.compute_range()
    compute_range=dict()

    #TODO: DO we need this in any other versions... json is not liking the tuple, complaining about a float error...
    """
      File "e2e_user_yolov3_example.py", line 246, in <module>
        get_calibration_table_tofa(model_path, augmented_model_path, calibration_dataset)
    File "e2e_user_yolov3_example.py", line 178, in get_calibration_table_tofa
        write_calibration_table(r)
    File "/home/lbathen/miniconda3/envs/onnx/lib/python3.8/site-packages/onnxruntime/quantization/quant_utils.py", line 409, in write_calibration_table
        file.write(json.dumps(calibration_cache))  # use `json.loads` to do the reverse
    File "/home/lbathen/miniconda3/envs/onnx/lib/python3.8/json/__init__.py", line 231, in dumps
        return _default_encoder.encode(obj)
    File "/home/lbathen/miniconda3/envs/onnx/lib/python3.8/json/encoder.py", line 199, in encode
        chunks = self.iterencode(o, _one_shot=True)
    File "/home/lbathen/miniconda3/envs/onnx/lib/python3.8/json/encoder.py", line 257, in iterencode
        return _iterencode(o, 0)
    File "/home/lbathen/miniconda3/envs/onnx/lib/python3.8/json/encoder.py", line 179, in default
        raise TypeError(f'Object of type {o.__class__.__name__} '
    TypeError: Object of type float32 is not JSON serializable
    """
    for k in range_dict.keys():                                                                                                                                                                                      
        r1 = float(range_dict[k][0])                                                                                                                                                                                 
        r2 = float(range_dict[k][1])                                                                                                                                                                                 
        compute_range[k]=(r1,r2)                                                                                                                                                                                               
        print(compute_range)

    write_calibration_table(compute_range) 
    print('calibration table generated and saved.')

    quant_model_path = model_path.split('.onnx')[0]
    quant_model_path = quant_model_path + '_quant.onnx'

    # Generate QDQ model
    mode = QuantizationMode.QLinearOps
    
    model = onnx.load_model(Path(model_path), False)

    # In TRT, it recommended to add QDQ pair to inputs of Add node followed by ReduceMean node.
    nodes_to_exclude = get_op_nodes_not_followed_by_specific_op(model, "Add", "ReduceMean")
    op_types_to_quantize = ['MatMul', 'Add']
    
    #quantize_static(model_path, quant_model_path, data_reader) # last data reader...
    #print(f'Wrote {quant_model_path}...')
    quantizer = QDQQuantizer(
        model,
        True, #per_channel
        False, #reduce_range
        mode,
        True,  #static
        QuantType.QInt8, #weight_type
        QuantType.QInt8, #activation_type
        compute_range,
        [], #nodes_to_quantize
        nodes_to_exclude,
        op_types_to_quantize,
        {'ActivationSymmetric' : True, 'AddQDQPairToWeight' : True, 'OpTypesToExcludeOutputQuantization': op_types_to_quantize, 'DedicatedQDQPair': True, 'QDQOpTypePerChannelSupportToAxis': {'MatMul': 1} }) #extra_options
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(quant_model_path, False)

    print('ONNX full precision model size (MB):', os.path.getsize(model_path)/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize(quant_model_path)/(1024*1024))

    quant_model_path = model_path.split('.onnx')[0]
    quant_model_path = quant_model_path + '_dyn_quant.onnx'
    quantize_dynamic(model_path, quant_model_path)

    print('ONNX full precision model size (MB):', os.path.getsize(model_path)/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize(quant_model_path)/(1024*1024))

    data_reader = TOFADataReader(calibration_dataset,
                                 width=384,
                                 height=384,
                                 stride=1000,
                                 batch_size=1,
                                 model_path=augmented_model_path)
    
    quant_model_path = model_path.split('.onnx')[0]
    quant_model_path = quant_model_path + '_sta_quant.onnx'
    quantize_static(model_path, quant_model_path, data_reader)

    print('ONNX full precision model size (MB):', os.path.getsize(model_path)/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize(quant_model_path)/(1024*1024))


def get_prediction_evaluation_tofa(model_path, validation_dataset, providers):

    width = 384 
    height = 384 
    evaluator = TOFAEvaluator(model_path, None, width=width, height=height, providers=providers)

    total_data_size = min(1000,len(os.listdir(validation_dataset))) # max of 1000 samples...
    start_index = 0
    stride=500
    batch_size = 1
    for i in range(0, total_data_size, stride):
        data_reader = TOFADataReader(validation_dataset,
                                              width=width,
                                              height=height,
                                              start_index=start_index,
                                              end_index=start_index+stride,
                                              stride=stride,
                                              batch_size=batch_size,
                                              model_path=model_path,
                                              is_evaluation=True)

        evaluator.set_data_reader(data_reader)
        evaluator.predict()
        start_index += stride

    result = evaluator.get_result()
    annotations = './annotations/instances_val2017.json'
    evaluator.evaluate(result, annotations)


if __name__ == '__main__':
    '''
    TensorRT EP INT8 Inference on Yolov3 model.

    The script is using subset of COCO 2017 Train images as calibration and COCO 2017 Val images as evaluation.
    1. Please create workspace folders 'train2017/calib' and 'val2017'.
    2. Download 2017 Val dataset: http://images.cocodataset.org/zips/val2017.zip
    3. Download 2017 Val and Train annotations from http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    4. Run following script to download subset of COCO 2017 Train images and save them to 'train2017/calib':
        python3 coco_filter.py -i annotations/instances_train2017.json -f train2017 -c all 

        (Reference and modify from https://github.com/immersive-limit/coco-manager)
    5. Download Yolov3 model:
        (i) ONNX model zoo yolov3: https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx 
        (ii) yolov3 variants: https://github.com/jkjung-avt/tensorrt_demos.git
    '''

    augmented_model_path = 'augmented_model.onnx'
    calibration_dataset = './train2017/calib'
    validation_dataset = './val2017'
    is_onnx_model_zoo_yolov3 = False 
    is_tofa = True

    # TensorRT EP INT8 settings
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = "calibration.flatbuffers"  # Calibration table name
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # Enable engine caching
    execution_provider = ["TensorrtExecutionProvider"]

    if is_tofa:
        model_path = 'tofa_Min.onnx'
        get_calibration_table_tofa(model_path, augmented_model_path, calibration_dataset)
        get_prediction_evaluation_tofa(model_path, validation_dataset, execution_provider)
    elif is_onnx_model_zoo_yolov3:
        model_path = 'yolov3.onnx'
        get_calibration_table(model_path, augmented_model_path, calibration_dataset)
        get_prediction_evaluation(model_path, validation_dataset, execution_provider)
    else:
        # Yolov3 variants from here
        # https://github.com/jkjung-avt/tensorrt_demos.git
        model_path = 'yolov3-608.onnx'
        get_calibration_table_yolov3_variant(model_path, augmented_model_path, calibration_dataset)
        get_prediction_evaluation_yolov3_variant(model_path, validation_dataset, execution_provider)

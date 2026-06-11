import json
import logging
import math
import os
import site
import stat
import sys
import inspect
from typing import List

if site.USER_SITE not in sys.path:
    sys.path.append(site.USER_SITE)
sys.path.insert(1, os.path.split(os.path.abspath(inspect.getsourcefile(lambda: 0)))[0])

import numpy as np
import pandas as pd

from regression.agency_model import Agency
from predict import Predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def initial_params(data):
    """解析出模型输入变量与对应数值，以字典形式返回
        params:{
            data: 输入的计算参数
        }
        return params:{
            sampling_point_dict: 计算参数字典 key::value
        }
    """
    if len(data) != 5:
        logging.error("参数错误,需要传递5个参数")
        return list()
    total_area = data[0]            # 芯片面积
    heat_consumption = data[1]      # 芯片功率
    rjc = data[2]                   # 芯片到外壳的热阻
    rjb = data[3]                   # 芯片到PCB的热阻
    wind_speed = data[4]            # 风速
    # 校验参数
    side = math.sqrt(float(total_area))
    sampling_point = [side, float(rjc), float(rjb), float(heat_consumption), float(wind_speed)]
    var_names = ['chip_length', 'Rjc', 'Rjb', 'power', 'wind_speed']
    json_params = Predict.get_params(sampling_point)
    logging.info("json_params: %s", json_params)

    sampling_point_dict = dict(zip(var_names, sampling_point))

    return sampling_point_dict


def run_param_calculate(data, paths):
    """
        params:{
            data: 输入的计算参数
            paths: 文件地址字典 包含模型文件和结果文件路径
        }
        return params:{
            sampling_point_dict: 计算参数字典 key::value
        }
    """
    params = initial_params(data)
    run_objective(params, paths)  # calculate objective


def run_objective(x, paths):
    """
        params:{
            x: 预处理后的输入参数
            paths: 文件地址字典 包含模型文件和结果文件路径
        }
        return params:{
            path_datafile: 结果csv文件保存路径
        }
    """
    y = passage_predict(x, paths)
    round_y = np.round(y, 6)  # 四舍五入至六位小数
    path_datafile = save_param_results(round_y, paths)
    result_path = paths.get("results_path")
    param_path = os.path.join(result_path, "ai_parameters.json")
    with os.fdopen(os.open(param_path, os.O_CREAT | os.O_WRONLY, 0o755), "w") as param_w:
        param_w.write(json.dumps(x))
    return path_datafile


def passage_predict(x, paths):
    """
        params:{
            x: 预处理后的输入参数
            paths: 文件地址字典 包含模型文件和结果文件路径

        }
        return params:{
            y: 通过算法生成的计算结果
        }
    """
    path_root = paths.get("models_path")
    files = os.listdir(path_root)
    logging.info(files)
    y = np.empty(shape=0)
    for file in files:
        path_file = os.path.join(path_root, file)
        if os.path.isfile(path_file) and file.endswith('.pth'):
            tmp = objective_agency(x, path_file)
            if file.find('_drop') == -1: # 文件名不包含_drop，不输出压降数据
                tmp[0][-1] = None
            if len(y) == 0:
                y = tmp
                continue
            y = np.vstack((y, tmp))
    return y


def objective_agency(x, path_model):
    """
        params:{
            x: 预处理后的输入参数
            path_model: 单个模型文件地址
        }
        return params:{
            ag.objective(x): 未经过格式化的的计算结果
        }
    """
    ag = Agency(path_model)
    return ag.objective(x) # 模型推理


def save_param_results(y, paths):
    """
        params:{
            y: 通过算法生成的计算结果
            paths: 文件地址字典 包含模型文件和结果文件路径
        }
        return params:{
            path_datafile: 结果csv文件保存路径
        }
    """
    logging.info(y)
    target_names = []
    result_path = paths.get("results_path")
    for tar in Predict.params['objective'].get("target_variable"):
        target_names += tar['name']
    path_datafile = os.path.join(result_path, 'task_results.csv')

    case_names, attach_col = get_passage_case_names(paths)
    first_col = np.append(np.array(['编号']), case_names)
    save_select_csv(first_col, y, path_datafile, target_names, attach_col=attach_col)

    return path_datafile


def save_select_csv(first_col: np.ndarray, y: np.ndarray, path_datafile: str,
                    target_names: List[str], attach_col: dict = None):
    """
        params:{
            first_col: 表格第一列数据
            y: 通过算法生成的计算结果
            path_datafile: 结果csv文件保存路径
            target_names: 目标变量名
            attach_col: 结果csv数据
        }
        return params:{
            ag.objective(x): 未经过格式化的的计算结果
        }
    """
    # 检查模型输出与预期输出数量对应
    if y.shape[-1] != len(target_names):
        raise ValueError(f'【io_module.iros_io】data column number not equal with header number!')
    first_label = first_col[0]
    row_label = first_col[1:]  # 散热器编码
    row_labels = (first_label, row_label)
    save_csv(y, path_datafile, header=target_names, row_labels=row_labels, attach_col=attach_col)


def save_csv(data, csv_path, header=None, row_labels=None, attach_col=None):
    """
        params:{
            data: 通过算法生成的计算结果
            csv_path: 结果csv文件保存路径
            header: 目标变量名
            row_labels: 除开目标变量名的表头
            attach_col: 结果csv数据
        }
        return params: null
    """
    data = np.array(data)
    if not data.shape[0]:
        data = np.array([[''] * len(header)])
    if not header or len(header) != data.shape[1]:
        header = False
        logging.info('【io_module.iros_io】 The format of header is not correct!')

    attach_col = attach_col or {}
    col_loc = attach_col.get('col_loc', -1)
    attach_data = attach_col.get('attach_data', None)
    attach_header = attach_col.get('attach_header', None)
    if col_loc != -1:
        data = np.hstack((data[:, 0:col_loc], attach_data, data[:, col_loc:]))
        if header:
            header = header[:col_loc] + attach_header + header[col_loc:]

    first_label, row_label = None, None
    row_label_flag = False
    if row_labels:
        first_label, row_label = row_labels[0], row_labels[1]
        row_label_flag = True

    csv_dir = os.path.dirname(os.path.realpath(csv_path))
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    df = pd.DataFrame(data, index=row_label)
    df.to_csv(csv_path, index=row_label_flag, index_label=first_label, header=header, encoding='utf-8_sig')
    logging.info('【io_module.iros_io】 Data has been saved in %s', csv_path)


def get_passage_case_names(paths) -> tuple:
    """
        params:{
            paths: 文件地址字典 包含模型文件和结果文件路径
        }
        return params:{
            np.array(sn_numbers): 散热器型号列表
            attach_col: 结果csv数据
        }
    """
    pth_path = paths.get("models_path")
    files = os.listdir(pth_path)
    sn_numbers = []
    for file in files:
        path_file = os.path.join(pth_path, file)
        if os.path.isfile(path_file) and file.endswith('.pth'):
            sn_numbers.append(file.split('@')[0])
    path_heatsink_info = os.path.join(pth_path, 'HeatsinkInfo.json')

    attach_col = {'col_loc': -1, 'attach_data': None, 'attach_header': None}
    if os.path.isfile(path_heatsink_info):
        flags = os.O_RDWR | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(path_heatsink_info, flags, modes), 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
        heatsinks = data.get("heatsinks")
        attach_data = attach_data_generator(heatsinks, sn_numbers)
        attach_header = ['加工类型', '宽度(mm)', '深度(mm)', '高度(mm)', '固定方式', '孔位宽度(mm)', '孔位深度(mm)']
        attach_col = {'col_loc': 0, 'attach_data': attach_data, 'attach_header': attach_header}

    return np.array(sn_numbers), attach_col


def attach_data_generator(heatsinks, sn_numbers):
    """
        params:{
            heatsinks: 散热器信息列表
            sn_numbers: 散热器型号列表
        }
        return params:{
            attach_data: 结果csv数据
        }
    """
    attach_data = []
    hole_stand_value = [0.0, "无"]
    for sn_number in sn_numbers:
        if sn_number not in [item.get("id") for item in heatsinks]:
            attach_data.append(['', '', '', '', '', '', ''])
            continue
        for heatsink in heatsinks:
            if sn_number == heatsink.get("id"):
                fix_method = heatsink.get("fixedMethod")
                tmp = [
                    heatsink.get('processingType', ''),
                    heatsink.get('heatsinkWidth', ''),
                    heatsink.get('heatsinkDepth', ''),
                    heatsink.get('heatsinkHeight', ''),
                    fix_method.get("method", ''),
                    "" if fix_method.get('holeWidth', '') in hole_stand_value else fix_method.get('holeWidth'),
                    "" if fix_method.get('holeDeep', '') in hole_stand_value else fix_method.get('holeDeep'),
                ]
                attach_data.append(tmp)
    return attach_data


def start_ai_selection_infer(models_path,results_path,input_argv):
    path_dict = dict()
    path_dict["models_path"] = models_path
    path_dict["results_path"] = results_path
    try:
        run_param_calculate(input_argv, path_dict)
    except Exception as err:
        logging.error(f'AI预测计算错误:{err}')


if __name__ == '__main__':

    logging.info(sys.argv)
    models_path = sys.argv[1]  # 模型目录
    results_path = sys.argv[2]  # 结果目录
    # 模型输入顺序：芯片面积、功率、Rjc、Rjb、风速
    input_argv = sys.argv[3:]
    path_dict = dict()
    if not models_path:
        sys.exit(1)
    if not results_path:
        sys.exit(1)
    path_dict["models_path"] = models_path
    path_dict["results_path"] = results_path
    try:
        run_param_calculate(input_argv, path_dict)
    except Exception as err:
        logging.error(f'AI预测计算错误:{err}')
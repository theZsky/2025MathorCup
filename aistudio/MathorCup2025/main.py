import sys
# 建议统一使用双反斜杠，避免转义问题
sys.path.append("E:\\competition\\aistudio\\MathorCup2025\\PaddleScience")
sys.path.append("E:\\competition\\aistudio\\MathorCup2025\\3rd_lib")
sys.path.append("E:\\competition\\aistudio\\MathorCup2025\\model")

import argparse
import os
import csv
import pandas as pd
from timeit import default_timer
from typing import List
import numpy as np
import paddle
import yaml
from paddle.optimizer.lr import LRScheduler

from model.src.data import instantiate_datamodule
from model.src.networks import instantiate_network
from model.src.utils.average_meter import AverageMeter
from model.src.utils.dot_dict import DotDict
from model.src.utils.dot_dict import flatten_dict


class StepDecay(LRScheduler):
    def __init__(
            self, learning_rate, step_size, gamma=0.1, last_epoch=-1, verbose=False
    ):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s."
                % type(step_size)
            )
        if gamma >= 1.0:
            raise ValueError("gamma should be < 1.0.")

        self.step_size = step_size
        self.gamma = gamma
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        i = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma ** i)


def instantiate_scheduler(config):
    if config.opt_scheduler == "CosineAnnealingLR":
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            config.lr, T_max=config.opt_scheduler_T_max
        )
    elif config.opt_scheduler == "StepLR":
        scheduler = StepDecay(
            config.lr, step_size=config.opt_step_size, gamma=config.opt_gamma
        )
    else:
        raise ValueError(f"Got {config.opt_scheduler=}")
    return scheduler


# 带有相对/绝对 Lp 损失的损失函数
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # 维度和 Lp 范数类型必须为正
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # 假设均匀网格
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * paddle.norm(
            x.reshape((num_examples, -1)) - y.reshape((num_examples, -1)), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        diff_norms = paddle.norm(x - y, 2)
        y_norms = paddle.norm(y, self.p)

        if self.reduction:
            if self.size_average:
                return paddle.mean(diff_norms / y_norms)
            else:
                return paddle.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def set_seed(seed: int = 0):
    paddle.seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def str2intlist(s: str) -> List[int]:
    return [int(item.strip()) for item in s.split(",")]


def parse_args(yaml="UnetShapeNetCar.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/" + yaml,
        help="配置文件路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练使用的设备（cuda 或 cpu）",
    )
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--batch_size", type=int, default=None, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=None, help="训练轮数")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="恢复训练的检查点文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="输出目录路径",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="日志目录路径",
    )
    parser.add_argument("--logger_types", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0, help="训练随机种子")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument(
        "--sdf_spatial_resolution",
        type=str2intlist,
        default=None,
        help="SDF 空间分辨率。使用逗号分隔值，例如 32,32,32。",
    )

    args, _ = parser.parse_known_args()
    return args


def load_config(config_path):
    def include_constructor(loader, node):
        # 获取当前 YAML 文件的路径
        current_file_path = loader.name
        # 获取包含当前 YAML 文件的文件夹
        base_folder = os.path.dirname(current_file_path)
        # 获取包含的文件路径，相对于当前文件
        included_file = os.path.join(base_folder, loader.construct_scalar(node))

        # 读取并解析包含的文件
        with open(included_file, "r") as file:
            return yaml.load(file, Loader=yaml.Loader)

    # 为 !include 注册自定义构造函数
    yaml.Loader.add_constructor("!include", include_constructor)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # 转换为点字典
    config_flat = flatten_dict(config)
    config_flat = DotDict(config_flat)
    return config_flat


import re


def extract_numbers(s):
    return [int(digit) for digit in re.findall(r'\d+', s)]


def write_to_vtk(out_dict, point_data_pos="press on mesh points", mesh_path=None, track=None):
    import meshio
    p = out_dict["pressure"]
    index = extract_numbers(mesh_path.name)[0]

    if track == "Dataset_1":
        index = str(index).zfill(3)
    elif track == "Track_B":
        index = str(index).zfill(4)

    print(f"网格 {index} 的压力形状 = {p.shape}")

    if point_data_pos == "press on mesh points":
        mesh = meshio.read(mesh_path)
        mesh.point_data["p"] = p.numpy()
        if "pred wss_x" in out_dict:
            wss_x = out_dict["pred wss_x"]
            mesh.point_data["wss_x"] = wss_x.numpy()
    elif point_data_pos == "press on mesh cells":
        points = np.load(mesh_path.parent / f"centroid_{index}.npy")
        npoint = points.shape[0]
        mesh = meshio.Mesh(
            points=points, cells=[("vertex", np.arange(npoint).reshape(npoint, 1))]
        )
        mesh.point_data = {"p": p.numpy()}

    print(f"写入：./output/{mesh_path.parent.name}_{index}.vtk")
    mesh.write(f"./output/{mesh_path.parent.name}_{index}.vtk")


@paddle.no_grad()
def eval(model, datamodule, config, loss_fn=None, track="Dataset_1"):
    test_loader = datamodule.test_dataloader(batch_size=config.eval_batch_size, shuffle=False, num_workers=0)
    data_list = []
    cd_list = []

    for i, data_dict in enumerate(test_loader):
        out_dict = model.eval_dict(data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode)
        if 'l2 eval loss' in out_dict:
            if i == 0:
                data_list.append(['id', 'l2 p'])
            else:
                data_list.append([i, float(out_dict['l2 eval loss'])])

        # 如果需要写入 vtk
        if config.write_to_vtk is True:
            print("datamodule.test_mesh_paths = ", datamodule.test_mesh_paths[i])
            write_to_vtk(out_dict, config.point_data_pos, datamodule.test_mesh_paths[i], track)

        # 提交 *.npy 到排行榜
        if "pressure" in out_dict:
            p = out_dict["pressure"].reshape((-1,)).astype(np.float32)
            test_indice = datamodule.test_indices[i]
            npy_leaderboard = f"./output/{track}/press_{str(test_indice).zfill(3)}.npy"
            print(f"保存 *.npy 文件到 [{track}] 排行榜：", npy_leaderboard)
            np.save(npy_leaderboard, p)
        if "velocity" in out_dict:
            v = out_dict["velocity"].reshape((-1, 3)).astype(np.float32)
            test_indice = datamodule.test_indices[i]
            npy_leaderboard = f"./output/{track}/vel_{str(test_indice).zfill(3)}.npy"
            print(f"保存 *.npy 文件到 [{track}] 排行榜：", npy_leaderboard)
            np.save(npy_leaderboard, v)
        if "cd" in out_dict:
            v = out_dict["cd"].item()
            test_indice = datamodule.test_indices[i]
            cd_list.append([i, v])

        # 写到 csv
        with open(f"./output/{config.project_name}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data_list)

    if "cd" in out_dict:
        titles = ["", "Cd"]
        df = pd.DataFrame(cd_list, columns=titles)
        df.to_csv(f'./output/{track}/Answer.csv', index=False)
    return


def train(config):
    model = instantiate_network(config)
    datamodule = instantiate_datamodule(config)
    train_loader = datamodule.train_dataloader(batch_size=config.batch_size, shuffle=False)
    eval_dict = None

    # 初始化优化器
    scheduler = instantiate_scheduler(config)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=scheduler, weight_decay=1e-4
    )

    # 初始化损失函数
    loss_fn = LpLoss(size_average=True)
    L2 = []

    for ep in range(config.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()

        for i, data_dict in enumerate(train_loader):
            optimizer.clear_grad()
            loss_dict = model.loss_dict(data_dict, loss_fn=loss_fn)
            loss = 0
            for k, v in loss_dict.items():
                loss = loss + v.mean()
            loss.backward()
            optimizer.step()
            train_l2_meter.update(loss.item())

        scheduler.step()
        t2 = default_timer()
        print(
            f"训练轮次 {ep} 耗时 {t2 - t1:.2f} 秒。L2 损失：{train_l2_meter.avg:.4f}"
        )

        L2.append(train_l2_meter.avg)
        if ep % config.eval_interval == 0 or ep == config.num_epochs - 1:
            # 你想在训练过程中评估，可使用 eval(...)
            pass
        # 保存权重
        if ep % config.save_interval == 0 or (ep == config.num_epochs - 1 and ep > 1):
            paddle.save(
                model.state_dict(),
                os.path.join("./output/", f"model-{config.model}-{config.track}-{ep}.pdparams"),
            )


def load_yaml(file_name):
    args = parse_args(file_name)
    config = load_config(args.config)

    # 用命令行参数更新配置
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    # 美观打印配置
    if paddle.distributed.get_rank() == 0:
        print(f"\n--------------- 配置 [{file_name}] 表 ----------------")
        for key, value in config.items():
            print("键：{:<30} 值：{}".format(key, value))
        print("--------------- 配置 yaml 表 ----------------\n")
    return config


def leader_board(config, track):
    os.makedirs(f"./output/{track}/", exist_ok=True)
    model = instantiate_network(config)
    checkpoint = paddle.load(f"./output/model-{config.model}-{config.track}-{config.num_epochs - 1}.pdparams")
    model.load_dict(checkpoint)
    print(f"\n------- 开始在 [{config.track}] 上评估 --------")
    config.n_train = 1
    t1 = default_timer()

    config.mode = "test"
    eval(
        model, instantiate_datamodule(config), config, loss_fn=lambda x, y: 0, track=track
    )
    t2 = default_timer()
    print(f"在 [{track}] 上推理耗时 {t2 - t1:.2f} 秒。")


if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)

    config_p = load_yaml("UnetShapeNetCar.yaml")
    train(config_p)
    leader_board(config_p, "Gen_Answer")

    config_cd = load_yaml("Unet_Cd.yaml")
    train(config_cd)
    leader_board(config_cd, "Gen_Answer")

    os.system(
        f"zip -r  ./output/Gen_Code.zip ./configs/ ./model/ ./README.md ./requirements.txt ./main.py ./main.ipynb")
    os.system(f"cd ./output && zip -r ./Gen_Answer.zip ./Gen_Answer")
    print("结果保存为 ./output/Gen_Code.zip，请上传平台完成提交")

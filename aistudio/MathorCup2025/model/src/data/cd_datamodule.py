import paddle
import numpy as np
import sys

sys.path.append("E:\\competition\\aistudio\\MathorCup2025\\PaddleScience")
sys.path.append("E:\\competition\\aistudio\\MathorCup2025\\3rd_lib")

import warnings
from pathlib import Path
from model.src.data.base_datamodule import BaseDataModule
from model.src.data.velocity_datamodule import read


class CdDataset(paddle.io.Dataset):
    def __init__(self, input_dir, obj_list, cd_list):
        self.cd_list = cd_list
        self.input_dir = input_dir
        self.obj_list = obj_list
        self.len = len(obj_list)

    def __getitem__(self, index):
        cd_label = self.cd_list[index]
        obj_name = self.obj_list[index]
        data_dict = read(self.input_dir / "Feature_File" / f"{obj_name}.obj")
        data_dict["cd"] = cd_label
        return data_dict

    def __len__(self):
        return self.len


class CdDataModule(BaseDataModule):
    def __init__(self, train_data_dir, test_data_dir, n_train, n_test, train_index_list, test_index_list):
        super().__init__()  # 等价于 BaseDataModule.__init__(self)
        # 这里注意：train_index_list、test_index_list 在你的配置里是 csv 里的 obj_name 吗？
        # 你的原代码里写了 train_csv, test_csv，并从中读 object list. 下面跟原始逻辑保持一致
        train_csv = np.loadtxt(train_data_dir + "/Label_File/dataset2_train_label.csv",
                               delimiter=",", dtype=str, encoding='utf-8')
        cd_label_train_list = train_csv[:, 2][1:].astype(np.float32)
        train_obj_list = train_csv[:, 1][1:]

        test_csv = np.loadtxt(test_data_dir + "/Label_File/dataset2_fake_test_label.csv",
                              delimiter=",", dtype=str, encoding='utf-8')
        cd_label_test_list = test_csv[:, 2][1:].astype(np.float32)
        fake_test_obj_list = test_csv[:, 1][1:]

        cd_label_train_list = cd_label_train_list[:n_train]
        cd_label_test_list = cd_label_test_list[:n_test]
        train_obj_list = train_obj_list[:n_train]
        test_obj_list = fake_test_obj_list[:n_test]

        available_train_number = len(train_obj_list)
        available_test_number = len(test_obj_list)
        available_case_number = len(test_obj_list)

        if n_train + n_test < available_case_number:
            warnings.warn(
                f"{available_train_number} traning meshes are available, but n_train= {n_train} are requested."
            )
            warnings.warn(
                f"{available_test_number} testing meshes are available, but n_test= {n_test} are requested."
            )

        # 在基类里我们有 self.train_data, self.test_data
        self.train_data = CdDataset(Path(train_data_dir), train_obj_list, cd_label_train_list)
        self.test_data = CdDataset(Path(test_data_dir), test_obj_list, cd_label_test_list)

        # 记录索引，以便 eval 保存文件名
        self.train_indices = train_obj_list
        self.test_indices = test_obj_list

    def decode(self, x):
        return x

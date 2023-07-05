# coding: utf-8

import os
import re
import sys
import orjson
from collections import OrderedDict

def rebuild_json(json_file):
    UP = {
        '音视频应用偏好': ['音乐类', '视频类', '有声读物类'],
        '出行交通工具偏好': ['地铁', '公交', '驾车'],
        '长途交通工具偏好': ['火车', '飞机', '汽车'],
        '是否有车': ['是', '否']
    }
    CA = {
        '移动状态': ['行走', '跑步', '静止', '汽车', '地铁', '高铁', '飞机', '未知'],
        '姿态识别': ['躺卧', '行走', '未知'],
        '地理围栏': ['家', '公司', '国内', '未知'],
        '户外围栏': ['户外', '室内', '未知']
    }

    with open(json_file, "r", encoding="utf-8") as f:
        data = orjson.loads(f.read())

    ids = data.pop("ids")
    assert set(data.keys()) == set(ids)

    rebuilt_data = OrderedDict()

    for data_id, data_dict in data.items():
        # align UP / CA items
        up_dict, ca_dict = {}, {}
        up_features = data_dict.pop("UP")
        for key in UP.keys():
            up_dict[key] = dict(
                (UP[key][i], up_features[key][i] if up_features != [] else 0.0)
                for i in range(len(UP[key])))
        ca_features = data_dict.pop("CA")
        for key in CA.keys():
            ca_dict[key] = dict(
                (CA[key][i], ca_features[key][i] if ca_features != [] else 0.0)
                for i in range(len(CA[key])))

        # divide single-line KG into different items starting with "subject："
        subjects = data_dict.pop("KG").split("；subject：")
        subjects = [subj if subj.startswith("subject：") else f"subject：{subj}" for subj in subjects]
        # print(len(subjects), orjson.dumps(subjects, option=orjson.OPT_INDENT_2).decode())
        # exit(0)

        rebuilt_item = OrderedDict()
        rebuilt_item["用户话语"] = data_dict.pop("用户话语")
        rebuilt_item["intent"] = data_dict.pop("intent")
        rebuilt_item["slot"] = data_dict.pop("slot")
        rebuilt_item["KG"] = subjects
        rebuilt_item["UP"] = up_dict
        rebuilt_item["CA"] = ca_dict
        assert data_dict == {}
        rebuilt_data[data_id] = rebuilt_item

    return rebuilt_data


if __name__ == '__main__':
    dataset_dir = "data/ProSLU"
    for split in ["train", "dev", "test"]:
        json_file = os.path.join(dataset_dir, f"{split}.json")

        rebuilt_json = rebuild_json(json_file)

        with open(os.path.join(dataset_dir, f"{split}_rebuild.json"), "w", encoding="utf-8") as f:
            f.write(orjson.dumps(rebuilt_json, option=orjson.OPT_INDENT_2).decode())
import csv
import os
from enum import Enum, auto
from tabulate import tabulate
from typing import List, Dict, Any


class Model(Enum):
    INCEPTIONTIME        = auto()
    RESNET               = auto()
    TST                  = auto()
    LSTM                 = auto()
    LOGISTIC_CLASSIFIER  = auto()


class MappingType:
    Direct_Mapping = "Direct mapping"


class LearningParadigm:
    Supervised_Learning = "Supervised"


class ModelArchitecture:
    Sequence_Models    = "Sequence Models"
    Convolutional_Model = "Convolutional"
    Traditional_ML     = "Traditional ML"


class ModelRegistry:
    OUTPUT_FILE = "model_details.csv"

    _registry: Dict[Model, Dict[str, Any]] = {
        Model.INCEPTIONTIME: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture":      ModelArchitecture.Convolutional_Model,
            "mapping":           MappingType.Direct_Mapping,
            "task":              "Classification",
        },
        Model.RESNET: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture":      ModelArchitecture.Convolutional_Model,
            "mapping":           MappingType.Direct_Mapping,
            "task":              "Classification",
        },
        Model.TST: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture":      ModelArchitecture.Sequence_Models,
            "mapping":           MappingType.Direct_Mapping,
            "task":              "Classification",
        },
        Model.LSTM: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture":      ModelArchitecture.Sequence_Models,
            "mapping":           MappingType.Direct_Mapping,
            "task":              "Classification",
        },
        Model.LOGISTIC_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture":      ModelArchitecture.Traditional_ML,
            "mapping":           MappingType.Direct_Mapping,
            "task":              "Classification",
        },
    }

    @classmethod
    def get_info(cls, model_type: Model) -> Dict[str, Any]:
        return cls._registry.get(model_type, {})

    @classmethod
    def all_models(cls) -> List[Model]:
        return list(cls._registry.keys())

    @classmethod
    def register_model(cls, model_type: Model, info: Dict[str, Any]) -> None:
        cls._registry[model_type] = info

    @classmethod
    def is_model_in_csv(cls, model_name: str) -> bool:
        try:
            with open(cls.OUTPUT_FILE, mode="r", newline="") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and row[0] == model_name:
                        return True
            return False
        except FileNotFoundError:
            return False

    @classmethod
    def write_model_to_csv(cls, model_type: Model) -> None:
        properties = cls.get_info(model_type)
        if not properties:
            return
        model_name = model_type.name
        if cls.is_model_in_csv(model_name):
            return
        row_data = [
            model_name,
            properties.get("learning_paradigm", ""),
            properties.get("architecture", ""),
            properties.get("mapping", ""),
            properties.get("task", ""),
        ]
        file_existed_before = os.path.exists(cls.OUTPUT_FILE)
        with open(cls.OUTPUT_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_existed_before or os.path.getsize(cls.OUTPUT_FILE) == 0:
                writer.writerow(["Model", "Learning Paradigm", "Architecture", "Mapping", "Task"])
            writer.writerow(row_data)
        table_data = [
            ["Model", model_name],
            ["Learning Paradigm", properties.get("learning_paradigm", "")],
            ["Architecture",      properties.get("architecture", "")],
            ["Mapping",           properties.get("mapping", "")],
            ["Task",              properties.get("task", "")],
        ]
        print(tabulate(table_data, headers=["Property", "Value"], tablefmt="grid"))

    @classmethod
    def get_enum_by_name(cls, name: str):
        for model_enum in cls._registry.keys():
            if model_enum.name == name:
                return model_enum
        raise ValueError(f"No Model enum found for name: {name}")

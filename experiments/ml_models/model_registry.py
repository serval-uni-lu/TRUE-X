import csv
import os
from enum import Enum, auto
from tabulate import tabulate
from typing import List, Dict, Any

class Model(Enum):
    VI_ENSEMBLE = auto()
    VI_DROPOUT = auto()
    VAE_RUL = auto()
    ENCODER = auto()
    RESNET = auto()
    FCN = auto()
    MCD_CNN = auto()
    TIME_CNN = auto()
    INCEPTIONTIME = auto()
    TST = auto()
    LSTM = auto()
    GRU = auto()
    RF_CLASSIFIER = auto()
    ET_CLASSIFIER = auto()
    RF_REGRESSOR  = auto()
    ET_REGRESSOR  = auto()
    XGB_REGRESSOR = auto()
    LGBM_REGRESSOR = auto()
    XGB_CLASSIFIER = auto()    
    LGBM_CLASSIFIER = auto()
    CNN_LSTM_REGRESSOR = auto()
    ATTENTION_LSTM_REGRESSOR = auto()
    TST_REGRESSOR = auto() 
    TCN_REGRESSOR = auto()
    LSTM_FCN_REGRESSOR = auto()
    TFT_REGRESSOR = auto()
    LSTM_REGRESSOR = auto()
    BI_LSTM_REGRESSOR = auto()
    LINEAR_REGRESSOR = auto()
    LOGISTIC_CLASSIFIER = auto() 
    NAMTS_CLASSIFIER = auto()
    NAMTS_REGRESSOR = auto()
    SOFT_DT_CLASSIFIER = auto()
    SOFT_DT_REGRESSOR = auto()
    ATTNPOOL_CLASSIFIER = auto()
    ATTNPOOL_REGRESSOR = auto()


class MappingType:
    Direct_Mapping = "Direct mapping"
    Indirect_Mapping = "Indirect mapping"

class LearningParadigm:
    Supervised_Learning = "Supervised"
    Transfer_Learning = "Transfer Learning"
    SemiSupervised_Learning = "Semi-Supervised"
    Unsupervised_Domain_Adaptation = "Unsupervised Domain Adaptation"

class ModelArchitecture:
    Sequence_Models = "Sequence Models"
    Convolutional_Model = "Convolutional"
    Probabilistic_Model = "Probabilistic"
    Generative_Model = "Generative Models"
    Traditional_ML = "Traditional ML" 

class ModelRegistry:
    """
    Central registry for storing and retrieving model metadata.
    - learning_paradigm (e.g. Supervised)
    - architecture (e.g. Probabilistic)
    - mapping (e.g. Direct mapping)
    - task (e.g. 'Regression'/'Classification')
    """
    OUTPUT_FILE = "model_details.csv"

    _registry: Dict[Model, Dict[str, Any]] = {
        Model.VI_ENSEMBLE: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Probabilistic_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.VI_DROPOUT: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Probabilistic_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.VAE_RUL: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Generative_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.ENCODER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Convolutional_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.RESNET: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Convolutional_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.FCN: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Convolutional_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.MCD_CNN: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Convolutional_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.TIME_CNN: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Convolutional_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.INCEPTIONTIME: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Convolutional_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.TST: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.LSTM: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.GRU: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.RF_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.ET_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.RF_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.ET_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.XGB_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.LGBM_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.XGB_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.LGBM_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.CNN_LSTM_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.ATTENTION_LSTM_REGRESSOR: {        
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.TST_REGRESSOR: {                  
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.TCN_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Convolutional_Model,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.LSTM_FCN_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,  # hybrid; fits existing buckets
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.TFT_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,  # transformer-based
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.LSTM_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.BI_LSTM_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Sequence_Models,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.LINEAR_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
        },
        Model.LOGISTIC_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": ModelArchitecture.Traditional_ML,
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
        },
        Model.NAMTS_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": "Interpretable Additive (per-channel CNN)",
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
            # optional niceties:
            "interpretable": True,
            "explainability_notes": "Sum of per-channel contributions; exposes channel-level logits."
        },
        Model.NAMTS_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": "Interpretable Additive (per-channel CNN)",
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
            "interpretable": True,
            "explainability_notes": "Additive channel contributions to scalar output."
        },
        Model.SOFT_DT_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": "Differentiable Tree",
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
            "interpretable": True,
            "explainability_notes": "Oblique splits; path probabilities and leaf logits are inspectable."
        },
        Model.SOFT_DT_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": "Differentiable Tree",
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
            "interpretable": True,
            "explainability_notes": "Oblique splits; path probabilities and leaf values are inspectable."
        },
        Model.ATTNPOOL_CLASSIFIER: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": "Attention Pooling (per-channel)",
            "mapping": MappingType.Direct_Mapping,
            "task": "Classification",
            "interpretable": True,
            "explainability_notes": "Time attention weights provide saliency per channel; attention ≠ causality."
        },
        Model.ATTNPOOL_REGRESSOR: {
            "learning_paradigm": LearningParadigm.Supervised_Learning,
            "architecture": "Attention Pooling (per-channel)",
            "mapping": MappingType.Direct_Mapping,
            "task": "Regression",
            "interpretable": True,
            "explainability_notes": "Time attention weights provide saliency per channel; attention ≠ causality."
        },
        

    }

    @classmethod
    def get_info(cls, model_type: Model) -> Dict[str, Any]:
        """Returns metadata for a given model_type."""
        return cls._registry.get(model_type, {})

    @classmethod
    def all_models(cls) -> List[Model]:
        """Returns all registered Model enums."""
        return list(cls._registry.keys())

    @classmethod
    def register_model(cls, model_type: Model, info: Dict[str, Any]) -> None:
        """Register or update a model entry at runtime."""
        cls._registry[model_type] = info

    @classmethod
    def is_model_in_csv(cls, model_name: str) -> bool:
        """Checks if the model is already present in the output CSV file."""
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
        """If the model isn't already in the CSV, append it."""
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
                writer.writerow([
                    "Model", 
                    "Learning Paradigm", 
                    "Architecture",
                    "Mapping",
                    "Task",
                ])
            writer.writerow(row_data)

        # Optional: print a small table to console
        table_data = [
            ["Model", model_name],
            ["Learning Paradigm", properties.get("learning_paradigm", "")],
            ["Architecture", properties.get("architecture", "")],
            ["Mapping", properties.get("mapping", "")],
            ["Task", properties.get("task", "")],
        ]
        print(tabulate(table_data, headers=["Property", "Value"], tablefmt="grid"))

    @classmethod
    def get_enum_by_name(cls, name: str):
        for model_enum in cls._registry.keys():
            if model_enum.name == name:
                return model_enum
        raise ValueError(f"No Model enum found for name: {name}")
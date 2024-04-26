import datasets
import json
from variables import PATH_TO_MEDMCQA_DATASET

"""
References:
    - https://github.com/Iker0610/transformer_scripts/blob/main/src/dataset_loader/conll_dataset_loader.py
    - https://huggingface.co/docs/datasets/about_dataset_load
    - https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/datasetbuilder.png
    - https://huggingface.co/docs/datasets/dataset_script
"""

# Create a DatasetBuilder class. GeneratorBasedBuilder is a subclass of DatasetBuilder that allows defining a _generate_examples method.
class MedMCQA_Dataset(datasets.GeneratorBasedBuilder):

    # Define the dataset.BuilderConfig for the dataset
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="MedMCQA_dataset",
            version=datasets.Version("1.0.0"),
            description="Dataset for MedMCQA",
            data_dir = PATH_TO_MEDMCQA_DATASET
        )
    ]

    # Define the structure of the dataset
    def _info(self):
        return datasets.DatasetInfo(
            description="Dataset for Sequence Classification",
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "opa": datasets.Value("string"),
                    "opb": datasets.Value("string"),
                    "opc": datasets.Value("string"),
                    "opd": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(num_classes=4, names=['a','b','c','d']) # Must be named 'label' for the task
                    # ignoring 'exp', 'subject_name', 'topic_name' and 'choice_type' because are not used for the task
                    # 'id' is also ignored because it is used as the key for each instance, not as a feature for it
                }
            ),
            homepage="https://medmcqa.github.io/"
        )

    def _split_generators(self, dl_manager):
        # Se comprueba que se ha especificado el path a los ficheros:
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")

        # Se obtienen los path:
        data_files = dl_manager.download_and_extract(self.config.data_files)
        print("Data files: ", data_files)

        splits = []
        # Si solo se ha pasado un conjunto se carga como el split del train
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files}))

        # Si se ha pasado un diccionario, se guardan los distintos splits:
        else:
            for split_name, files in data_files.items():
                if isinstance(files, str):
                    files = [files]
                splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))

        return splits

    def _generate_examples(self, files):
        for file in files:
            with open(file, "r", encoding="utf-8") as file:
                instances = [json.loads(row) for row in file]

            for instance in instances:
                yield instance["id"], {
                    "question": instance["question"],
                    "opa": instance["opa"],
                    "opb": instance["opb"],
                    "opc": instance["opc"],
                    "opd": instance["opd"],
                    "label": instance["cop"] - 1 # The labels are 1, 2, 3, 4. We need to convert them to 0, 1, 2, 3
                }        
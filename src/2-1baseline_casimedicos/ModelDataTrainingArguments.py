from dataclasses import dataclass, field
from typing import Optional

'''
This class is used to define the arguments that will be used to train the model.
'''

@dataclass
class ModelDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on
    the command line.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_loader: str = field(
        metadata={"help": "Path to the data loader script."}
    )
    train_file: str = field(
        metadata={"help": "File containing the training data."}
    )
    validation_file: str = field(
        metadata={"help": "File containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "File containing the test data."})

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
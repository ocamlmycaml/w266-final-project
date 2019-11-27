import os
import csv
import enum
from typing import List

import tensorflow as tf
import pandas as pd


TEXT_FIELD_NAME = "text"
DEFAULT_DATA_DIR = 'data/GCDC_rerelease'


class Label(enum.Enum):
    EXPERT_CONCENSUS = "labelA"  # field in the CSV files
    MTURK_CONCENSUS = "labelM"
    
    
class TrainOrTest(enum.Enum):
    TRAIN = 0
    TEST = 1

    
class Source(enum.Enum):
    CLINTON = ('Clinton_train.csv', 'Clinton_test.csv')
    ENRON = ('Enron_train.csv', 'Enron_test.csv')
    YAHOO = ('Yahoo_train.csv', 'Yahoo_test.csv')
    YELP = ('Yelp_train.csv', 'Yelp_test.csv')
    

ALL_SOURCES = [source for source in Source]


def iterate_over_file(sources: List[Source], train_or_test: TrainOrTest, data_dir: str, label: Label):
    for source in sources:
        file_name = source.value[train_or_test.value]  # clever, eh?
        with open(os.path.join(data_dir, file_name)) as infile:
            reader = csv.DictReader(infile)
            for item in reader:
                yield (item[TEXT_FIELD_NAME], item[label.value])

            
def load(
        train_or_test: TrainOrTest,
        sources: List[Source] = ALL_SOURCES,
        data_dir: str = DEFAULT_DATA_DIR,
        label: Label = Label.EXPERT_CONCENSUS
):
    return tf.data.Dataset.from_generator(
        lambda: iterate_over_file(sources, train_or_test, data_dir, label),
        (tf.string, tf.int64),
        (tf.TensorShape([]), tf.TensorShape([]))
    )


def load_pandas(
        train_or_test: TrainOrTest,
        sources: List[Source] = ALL_SOURCES,
        data_dir: str = DEFAULT_DATA_DIR,
        label: Label = Label.EXPERT_CONCENSUS
):
    data = list(iterate_over_file(sources, train_or_test, data_dir, label))
    return pd.DataFrame({
        'text': [text for text, _ in data],
        'label': [l for _, l in data]
    })
import dataclasses
from logging import getLogger

logger = getLogger(__name__)

__all__ = ["DATASET_CSVS"]


@dataclasses.dataclass(frozen=True)
class DatasetCSV:
    train: str
    val: str
    test: str


DATASET_CSVS = {
    # paths from `src` directory
    "Jung": DatasetCSV(
        train="./csv/Jung/train.csv",
        val="./csv/Jung/val.csv",
        test="./csv/Jung/test.csv",
    ),
    "RDD": DatasetCSV(
        train="./csv/RDD/train.csv",
        val="./csv/RDD/val.csv",
        test="./csv/RDD/test.csv",
    ),
    "Kliger": DatasetCSV(
        train="./csv/Kliger/train.csv",
        val="./csv/Kliger/val.csv",
        test="./csv/Kliger/test.csv",
    ),
}
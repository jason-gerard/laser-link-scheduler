from enum import StrEnum, auto
import os

from laser_link_scheduler.constants import SOURCES_ROOT


class FileType(StrEnum):
    CONTACT_PLAN = auto()
    CONTACT_PLAN_SCHEDULED = auto()
    TEG = auto()
    TEG_SCHEDULED = auto()
    REPORT = auto()


def get_experiment_file(experiment_name, file_type: FileType) -> str:
    return os.path.join(
        SOURCES_ROOT, experiment_name, f"{experiment_name}_{file_type}"
    )

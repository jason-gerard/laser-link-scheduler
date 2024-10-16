import os
from enum import Enum

from constants import SOURCES_ROOT


class FileType(Enum):
    CONTACT_PLAN = "contact_plan"
    SCHEDULED = "contact_plan_scheduled"
    TEG = "teg"
    TEG_SCHEDULED = "teg_scheduled"
    REPORT = "report"


def get_experiment_file(experiment_name, file_type: FileType) -> str:
    file_suffix = file_type.value
    return str(os.path.join(SOURCES_ROOT, experiment_name, f"{experiment_name}_{file_suffix}"))

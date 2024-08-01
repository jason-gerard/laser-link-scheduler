import pickle
from tests.testing_utils import EXPERIMENT_NAMES, scheduler_test_driver, get_regression_experiment_file
from utils import FileType

if __name__ == "__main__":
    for experiment_name in EXPERIMENT_NAMES:
        teg, scheduled_teg = scheduler_test_driver(experiment_name)
        
        teg_path = get_regression_experiment_file(experiment_name, FileType.TEG)
        with open(teg_path, "wb") as f:
            pickle.dump(teg, f)

        scheduled_teg_path = get_regression_experiment_file(experiment_name, FileType.TEG_SCHEDULED)
        with open(scheduled_teg_path, "wb") as f:
            pickle.dump(scheduled_teg, f)

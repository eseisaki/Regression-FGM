from tools import *
from dataset import *


def test_update_bar():
    bar_percent = 0
    line_counter = 0
    max_lines = 10
    for line in range(max_lines):
        bar_percent, line_counter = update_bar(bar_percent, line_counter, max_lines)

    assert bar_percent == 100


def test_fixed_dataset():
    create_drift_dataset(5, 1, 10, 0.25, 2, "../io_files/draft")

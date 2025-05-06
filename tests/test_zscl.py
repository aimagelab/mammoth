import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


def test_zscl():
    sys.argv = ['mammoth',
                '--model',
                'zscl',
                '--dataset',
                'seq-cifar100',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--clip_backbone',
                'ViT-B/16',
                '--debug_mode',
                '1']

    try:
        main()
    except AssertionError as e:
        if 'Conceptual Captions dataset not found' in str(e):
            print("Conceptual Captions dataset not found, will skip the test.")
            return

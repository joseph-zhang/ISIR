import sys
import os
import params
if params.PYLIBS_DIR is not None:
    sys.path.insert(1, params.PYLIBS_DIR)

from conf import Conf
from dataFunctions import parse_args

def main(argv):
    training, mode = parse_args(argv, params)
    model = Conf(params=params, mode=mode)
    if training:
        try:
            model.train()
        except ValueError as error:
            print(error, file=sys.stderr)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = params.TEST_GPUS
        model.test()

if __name__ == '__main__':
    main(sys.argv)

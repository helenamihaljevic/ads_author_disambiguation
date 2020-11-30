from glob import glob
from os.path import dirname, abspath, join

DIR_PATH = dirname(abspath(__file__))
PATH_DATA = join(DIR_PATH, 'data')
PATH_MODELS = join(DIR_PATH, 'models')

DIR_LABELED_DATA = join('data', 'labeled_data')
EXCEL_FILES = glob(join(DIR_PATH, DIR_LABELED_DATA, '*.xlsx'))
PATH_MATRICES = join(DIR_PATH, 'matrices')

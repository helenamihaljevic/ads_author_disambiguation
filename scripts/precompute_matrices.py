import sys
import time
from os.path import join, getmtime

sys.path.append('../')
from config import DIR_PATH, EXCEL_FILES
from disambiguator import ALL_FEATURES
from utils.modelling import load_ai_blocks, build_similarity_matrices, load_matrices
from utils.helpers import dump_pickle_file, extract_file_name_from_path

import argparse

parser = argparse.ArgumentParser(description='Precompute all matrices')
parser.add_argument('-a', '--all', type=bool, required=False, default=False,
                    help='update all matrices or only those for which the block files were recently updated')
parser.add_argument('--min_block_size', type=int, required=False, default=2,
                    help='Minimum block size')
parser.add_argument('--max_block_size', type=int, required=False, default=10000,
                    help='Maximum block size. if you want to use all, set to any number > 3000')
args = parser.parse_args()


def fetch_recently_updated_files(baseline_file, files, blocks):
    updated_data = [extract_file_name_from_path(f)[0] for f in files if getmtime(f) > getmtime(baseline_file)]
    updated_data = [f for f in updated_data if f in blocks]
    return updated_data


if __name__ == '__main__':
    start_time = time.time()
    ai_blocks = load_ai_blocks(EXCEL_FILES, args.max_block_size, args.min_block_size)
    if args.all is False: # do not attempt to update all matrices
        try:  # load existing matrices and update only those where the labeled data files have changed in the meantime
            ai_block_to_attrs, ai_block_to_matrix = load_matrices(blocks=ai_blocks)  # load precomputed matrices
            print("1")
            updated_blocks = fetch_recently_updated_files(join(DIR_PATH, 'matrices', 'ai_block_to_matrix.pickle'),
                                                          EXCEL_FILES, ai_blocks)
            print("2")
            print(f"ai blocks that have been updated since the last matrix computation: {updated_blocks}")
            update_ai_block_to_matrix, update_ai_block_to_attrs = build_similarity_matrices(updated_blocks,
                                                                                            ALL_FEATURES,
                                                                                            verbose=True)
            print(update_ai_block_to_matrix.keys())
            ai_block_to_attrs.update(update_ai_block_to_attrs)
            ai_block_to_matrix.update(update_ai_block_to_matrix)

        except:  # Precompute all matrices
            ai_block_to_matrix, ai_block_to_attrs = build_similarity_matrices(ai_blocks, ALL_FEATURES, verbose=True)
    else:
        ai_block_to_matrix, ai_block_to_attrs = build_similarity_matrices(ai_blocks, ALL_FEATURES, verbose=True)

    dump_pickle_file(join(DIR_PATH, 'matrices', 'ai_block_to_matrix.pickle'), ai_block_to_matrix)
    dump_pickle_file(join(DIR_PATH, 'matrices', 'ai_block_to_attrs.pickle'), ai_block_to_attrs)

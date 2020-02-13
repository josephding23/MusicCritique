import numpy as np
import tables
import os

DATA_PATH = 'data'
RESULTS_PATH = 'E:/data'
SCORE_FILE = os.path.join(RESULTS_PATH, 'match_scores.json')
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def load_data_from_npz(filename):
    """Load and return the training data from a npz file (sparse format)."""
    with np.load(filename) as f:
        data = np.zeros(f['shape'], np.bool_)
        data[[x for x in f['nonzero']]] = True
    return data


def msd_id_to_h5(msd_id):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join(RESULTS_PATH, 'lmd_matched_h5',
                        msd_id_to_dirs(msd_id) + '.h5')



if __name__ == '__main__':
    msd_id = 'TRAAAGR128F425B14B'
    with tables.open_file(msd_id_to_h5(msd_id)) as h5:
        print( 'ID: {}'.format(msd_id))
        print('"{}" by {} on "{}"'.format(
            h5.root.metadata.songs.cols.title[0].decode('UTF-8'),
            h5.root.metadata.songs.cols.artist_name[0].decode('UTF-8'),
            h5.root.metadata.songs.cols.release[0].decode('UTF-8')))
        print(h5.root.metadata.songs.cols.genre[0].decode('UTF-8'))
        print('Top 5 artist terms:', ', '.join([ term.decode('UTF-8') for term in list(h5.root.metadata.artist_terms)[:5]]))

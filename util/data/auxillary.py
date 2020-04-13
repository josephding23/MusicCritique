from pymongo import MongoClient
import math
import pypianoroll
import os
import pretty_midi
import matplotlib.pyplot as plt


def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi


def get_genre_collection():
    client = MongoClient(connect=False)
    return client.free_midi.genres


def get_jazz_collection():
    client = MongoClient(connect=False)
    return client.jazz_midi.midi


def find_music_with_multiple_genres():
    root_dir = 'E:/merged_midi/'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'GenresNum': {'$exists': False}}):
        performer = midi['Performer']
        name = midi['Name']
        genres = []
        total_genres_num = midi_collection.count({'Name': name, 'Performer': performer})
        for md in midi_collection.find({'Name': name, 'Performer': performer}):
            genres.append(md['Genre'])
        print(total_genres_num, genres)
        midi_collection.update_one({'_id': midi['_id']},
                                   {'$set': {'GenresNum': total_genres_num, 'TotalGenres': genres}})
        print('Progress: {:.2%}\n'.format(midi_collection.count({'GenresNum': {'$exists': True}}) / midi_collection.count()))


def print_all_genres_num():
    genres_collection = get_genre_collection()
    midi_collection = get_midi_collection()

    for genre in genres_collection.find():
        whole_num = 0
        for midi in midi_collection.find({'Genre': genre['Name']}):
           whole_num += midi['PiecesNum']
        genres_collection.update_one(
            {'_id': genre['_id']},
            {'$set': {'PiecesNum': whole_num}}
        )
        print(genre['Name'], whole_num)


def add_midi_valid_pieces_num():
    midi_collection = get_midi_collection()
    for midi in midi_collection.find():
        pieces = midi['PiecesNum']
        valid_pieces = int(math.modf(pieces)[1] + 1) if math.modf(pieces)[0] >= 0.9 else int(math.modf(pieces)[1])
        midi_collection.update_one({'_id': midi['_id']}, {'$set': {'ValidPiecesNum': valid_pieces}})


def add_genre_valid_pieces_num():
    midi_collection = get_midi_collection()
    genre_collection = get_genre_collection()
    for genre in genre_collection.find():
        valid_pieces_num = 0
        for midi in midi_collection.find({'Genre': genre['Name']}):
            valid_pieces_num += midi['ValidPiecesNum']
        genre_collection.update_many({'_id': genre['_id']}, {'$set': {'ValidPiecesNum': valid_pieces_num}})
        print(f"{genre['Name']} finished")


def get_original_tempo(md5):
    midi = get_midi_collection().find_one({'md5': md5})
    print(midi['Info']['tempo'][0])
    return midi['Info']['tempo'][0]


def set_paragraph_num_info():
    midi_collection = get_midi_collection()
    root_dir = 'E:/free_midi_library/merged_midi/'
    for midi in midi_collection.find({'PiecesNum': {'$exists': False}}, no_cursor_timeout = True):
        path = root_dir + midi['Genre'] + '/' + midi['md5'] + '.mid'
        pm = pretty_midi.PrettyMIDI(path)
        length = pm.get_end_time()

        piece_num = length / 8
        print(piece_num)

        midi_collection.update_one({'_id': midi['_id']}, {'$set': {'PiecesNum': piece_num}})
        print('Progress: {:.2%}\n'.format(midi_collection.count({'PiecesNum': {'$exists': True}}) / midi_collection.count()))


def get_nonempty_tracks_num():
    midi_collection = get_midi_collection()
    tracks_num_list = [0, 0, 0, 0, 0, 0]
    for midi in midi_collection.find():
        track_num = midi['NotEmptyTracksNum']
        tracks_num_list[track_num] += 1
    plt.bar([0, 1, 2, 3, 4, 5], tracks_num_list)
    plt.xlabel('Instrument tracks num')
    plt.ylabel('MIDI num')
    plt.show()

    print(tracks_num_list[5] / midi_collection.count())


def label_all_numpy_existed():
    root_dir = 'e:/midi_matrix/rock'
    for file in os.listdir(root_dir):
        md5 = file[:-4]
        get_midi_collection().update_one({'md5': md5, 'Genre': 'rock'}, {'$set': {'MultiInstrNpyGenerated': True}})


def find_data_with_no_empty_tracks():
    root_dir = 'E:/merged_midi/'
    total = 0
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'NotEmptyTracksNum': {'$exists': False}}):
        instr_tracks = {
            'Drums': None,
            'Piano': None,
            'Guitar': None,
            'Bass': None,
            'Strings': None
        }
        num = 0
        try:
            path = root_dir + midi['Genre'] + '/' + midi['md5'] + '.mid'
            mult = pypianoroll.parse(path)
            for track in mult.tracks:
                num += 1
            midi_collection.update_one(
                {'_id': midi['_id']},
                {'$set': {'NotEmptyTracksNum': num}}
            )
            print('Progress: {:.2%}\n'.format(midi_collection.count({'NotEmptyTracksNum': {'$exists': True}}) / midi_collection.count()))
        except:
            total += 1
            # midi_collection.delete_one({'_id': midi['_id']})
    print(total)


def get_total_piece_num():
    total_pieces = 0
    for genre in get_genre_collection().find():
        total_pieces += genre['PiecesNum']
    print(total_pieces)


def get_genre_files_num():
    genre_collection = get_genre_collection()
    for genre in genre_collection.find():
        files_num = get_midi_collection().count({'Genre': genre['Name']})
        genre_collection.update_one(
            {'Name': genre['Name']},
            {'$set': {'FilesNum': files_num}}
        )


def fix_jazz_pieces_num():
    genre_collection = get_genre_collection()
    midi_collection = get_midi_collection()
    jazz_collection = get_jazz_collection()

    old_pieces = 0
    for midi in midi_collection.find({'Genre': 'jazz'}):
        old_pieces += midi['ValidPiecesNum']
    old_train = int(old_pieces * 0.9)
    old_test = old_pieces - old_train

    old_files = get_midi_collection().count({'Genre': 'jazz'})

    new_pieces = 0
    for midi in jazz_collection.find({}):
        new_pieces += midi['ValidPiecesNum']
    new_train = int(new_pieces * 0.9)
    new_test = new_pieces - new_train

    new_files = get_jazz_collection().count()

    genre_collection.update_one(
        {'Name': 'jazz'},
        {'$set': {
            'OriginalFilesNum': old_files,
            'OriginalValidPiecesNum': old_pieces,
            'OriginalTrainPieces': old_train,
            'OriginalTestPieces': old_test,

            'NewFilesNum': new_files,
            'NewValidPiecesNum': new_pieces,
            'NewTrainPieces': new_train,
            'NewTestPieces': new_test,

            'FilesNum': old_files + new_files,
            'ValidPiecesNum': old_pieces + new_pieces,
            'TrainPieces': old_train + new_train,
            'TestPieces': old_test + new_test
        }}
    )


if __name__ == '__main__':
    fix_jazz_pieces_num()

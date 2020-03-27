from pymongo import MongoClient
import pypianoroll
import os

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def get_genre_collection():
    client = MongoClient(connect=False)
    return client.free_midi.genres



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


def add_more_valid_pieces_num():
    for genre in get_genre_collection().find():
        valid_pieces_num = genre['PiecesNum'] - get_midi_collection().count({'Genre': genre['Name']})
        get_genre_collection().update_many({'_id': genre['_id']}, {'$set': {'ValidPiecesNum': valid_pieces_num}})


def get_original_tempo(md5):
    midi = get_midi_collection().find_one({'md5': md5})
    print(midi['Info']['tempo'][0])
    return midi['Info']['tempo'][0]


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


def get_music_with_no_empty_tracks():
    midi_collection = get_midi_collection()
    print(midi_collection.count({'Genre': 'rock', 'NotEmptyTracksNum': {'$gte': 4}}), midi_collection.count())




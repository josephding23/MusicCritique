from pymongo import MongoClient
import shutil
import hashlib
from tqdm import tqdm
import os

def get_performer_collection():
    client = MongoClient(connect=False)
    return client.free_midi.performers

def get_genre_collection():
    client = MongoClient(connect=False)
    return client.free_midi.genres

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def name_to_md5():
    m = hashlib.md5()

    root_dir = 'E:/free_MIDI/'
    new_root_dir = 'E:/MIDI_Files/'
    midi_collection = get_midi_collection()
    '''
    for genre in os.listdir(root_dir):
        genre_dir = os.path.join(root_dir, genre)
        genre_songs = os.listdir(genre_dir)
        print(len(genre_songs), midi_collection.count({'Genre': genre}))
    '''
    pbar = tqdm(total=midi_collection.count())
    pbar.update(midi_collection.count({'Copied': True}))
    for song in midi_collection.find({'Copied': False}):
        genre = song['Genre']
        title = song['Name']
        performer = song['Performer']
        src_path = root_dir + genre + '/' + title + ' - ' + performer + '.mid'
        # print(os.path.exists(src_path))
        genre_path  =  new_root_dir + genre
        if not os.path.exists(genre_path):
            os.mkdir(genre_path)
        m.update(bytes(title + ' - ' + performer, 'utf-8'))
        md5Value = m.hexdigest()
        dst_path = genre_path + '/' + md5Value + '.mid'
        # print(dst_path)
        shutil.copyfile(src_path, dst_path)
        midi_collection.update_one(
            {'_id': song['_id']},
            {'$set': {
                'Copied': True,
                'md5': md5Value
            }})
        pbar.update(1)

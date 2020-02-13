from pymongo import MongoClient
import os
from hashlib import md5


def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

if __name__ == '__main__':
    root_dir = 'E:/free_MIDI'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'Downloaded': True}):
        genre = midi['Genre']
        name = midi['Name']
        performer = midi['Performer']

        path = root_dir + '/' + genre + '/' + name + ' - ' + performer + '.mid'
        h1 = md5()
        h1.update(bytes(path, 'utf-8'))

        print(h1.hexdigest())
        if not os.path.exists(path):
            print(path)

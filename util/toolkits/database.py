from pymongo import MongoClient

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi


def get_classical_collection():
    client = MongoClient(connect=False)
    return client.classical_midi.midi


def get_jazz_collection():
    client = MongoClient(connect=False)
    return client.jazz_midi.midi


def get_jazzkar_collection():
    client = MongoClient(connect=False)
    return client.jazz_midikar.midi


def get_genre_collection():
    client = MongoClient(connect=False)
    return client.free_midi.genres


def get_md5_of(performer, song, genre):
    if genre != 'classical':
        midi_collection = get_midi_collection()
        try:
            md5 = midi_collection.find_one({'Performer': performer, 'Name': song, 'Genre': genre})['md5']
            return md5
        except Exception:
            raise Exception('No midi Found.')
    else:
        midi_collection = get_classical_collection()
        try:
            md5 = midi_collection.find_one({'Composer': performer, 'Name': song})['md5']
            return md5
        except Exception:
            raise Exception('No midi Found.')
from util.toolkits.data_convert import *
from util.toolkits.database import *

def evaluate_density_of_file(path, valid_pieces_num):
    npy_file = np.load(path)
    midi_data = npy_file['arr_0']
    notes_in_valid_paragraphs = 0
    for i in range(midi_data.shape[0]):
        data = midi_data[i, :]
        if data[0] < midi_data:
            notes_in_valid_paragraphs += 1
    print(notes_in_valid_paragraphs, midi_data.shape[0])
    density = notes_in_valid_paragraphs / (valid_pieces_num * (64 * 84))
    return density


def evaluate_all_free_midi_density():
    midi_collection = get_midi_collection()
    root_dir = 'E:/midi_matrix/one_instr'
    for midi in midi_collection.find({'NotesDensity': {'$exists': False}}):
        path = root_dir + '/' + midi['Genre'] + '/' + midi['md5'] + '.npz'
        pieces_num = midi['PiecesNum']
        density = evaluate_density_of_file(path, pieces_num)
        # midi_collection.update_one({'_id': midi['_id']}, {'$set': {'NotesDensity': density}})
        '''
        print('Progress: {:.2%}\n'.format(
            midi_collection.count({'NotesDensity': {'$exists': True}}) / midi_collection.count()))
            '''


def evaluate_all_other_midi_density():
    midi_collection = get_jazzkar_collection()
    root_dir = 'E:/jazz_midkar/npy_files'
    for midi in midi_collection.find({'NotesDensity': {'$exists': False}}):
        path = root_dir + '/' + midi['md5'] + '.npz'
        pieces_num = midi['PiecesNum']
        density = evaluate_density_of_file(path, pieces_num)
        midi_collection.update_one({'_id': midi['_id']}, {'$set': {'NotesDensity': density}})
        print('Progress: {:.2%}\n'.format(
            midi_collection.count({'NotesDensity': {'$exists': True}}) / midi_collection.count()))


def plot_density():
    density_info = [0, 0, 0, 0, 0, 0, 0, 0]
    y_info = ['0~5', '5~10', '10~15', '15~20', '20~25', '25~30', '30~35', '35~40']
    for collection in [get_midi_collection(), get_classical_collection(), get_jazz_collection(), get_jazzkar_collection()]:
        for midi in collection.find():
            density = midi['NotesDensity']
            density_info[int(density * 100) // 5] += 1

    plt.barh(range(len(density_info)), density_info)
    plt.xlabel('Music files num')
    plt.ylabel('Notes density (%)')
    plt.yticks([i for i in range(8)], y_info)
    plt.show()


if __name__ == '__main__':
    plot_density()

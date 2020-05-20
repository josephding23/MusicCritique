from util.toolkits.database import *
from util.toolkits.data_convert import *
import music21


def get_chord(note_nums):
    notes = []
    for note_num in note_nums:
        name = pretty_midi.note_number_to_name(note_num)
        note = music21.note.pitch.Pitch(name)
        notes.append(note)
    # print(notes)
    chord = music21.chord.Chord(notes)
    # chord.duration = music21.duration.Duration(length)
    return chord


def evaluate_midi_chord(song='21 Guns', performer='Green Day', genre='rock'):
    root_dir = 'E:/free_midi_library/transposed_midi'

    try:
        md5 = get_md5_of(performer, song, genre)
        original_path = root_dir + '/' + genre + '/' + md5 + '.mid'
        print(original_path)
    except Exception as e:
        print(e)
        return

    key_mode = get_midi_collection().find_one({'md5': md5})['KeySignature']['Mode']
    if key_mode == 'major':
        current_key = music21.key.Key('C')
    else:
        current_key = music21.key.Key('a')
    ori_data = generate_data_from_midi(original_path)
    data_piece = ori_data[0, :, :]
    # plot_data(data=ori_data[0, :, :])

    notes = []
    for part in range(ori_data.shape[0]):
        print(f'Part {part}:')
        data_piece = ori_data[part, :, :]
        for time in range(64):
            if time == 0:
                old_notes = []
            else:
                old_notes = notes
            notes = []
            for note in range(84):
                if data_piece[time][note] != 0:
                    notes.append(note + 24)

            if len(notes) == 0:
                notes = old_notes
            elif notes == old_notes:
                continue
            else:
                try:
                    chord = get_chord(notes)
                    print(current_key.getScaleDegreeAndAccidentalFromPitch(chord.root()), chord.commonName)
                except Exception as e:
                    print(e.__traceback__)
                    return
        print()


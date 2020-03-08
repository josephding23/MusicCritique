import guitarpro

def test_module():
    song = guitarpro.parse('./data/Deacon Blues.gp5')

    for track in song.tracks:
        for measure in track.measures:
            for voice in measure.voices:
                print(voice)

if __name__ == '__main__':
    test_module()
Python编曲实践（四）：向MIDI文件中添加鼓组音轨 
=========================
## 前言
在前面三篇文章中，我介绍了如何通过Python的Mido库添加旋律、和弦和模拟滑音、颤音。然而，鼓的作用也是不可忽视的，它就像骨骼一样支撑起整个音乐，而编辑鼓点的样式也为想象力的发挥提供了无限空间，是十分愉悦的事情，本篇文章我就介绍如何向MIDI文件中添加鼓的音轨。
## 专属频道
同其他乐器不同，MIDI格式为鼓组提供了专属的频道，默认在10号频道，并在这个频道的40多个音符对应处替换上了不同种类的鼓点样式，为了更方便调用，我做成了一个函数，以供参考：
```python
def get_drum_dict():
    drum_dict = {
        'acoustic_bass': 35,
        'bass1': 36,
        'side_stick': 37,
        'acoustic_snare': 38,
        'hand_clap': 39,
        'electric_snare': 40,
        'low_floor_tom': 41,
        'closed_hi-hat': 42,
        'high_floor_tom': 43,
        'pedal_hi-hat': 44,
        'low_tom': 45,
        'open_hi-hat': 46,
        'low-mid_tom': 47,
        'hi-mid_tom': 48,
        'crash_cymbal1': 49,
        'high_tom': 50,
        'ride_cymbal1': 51,
        'chinese_cymbal': 52,
        'ride_bell': 53,
        'tambourine': 54,
        'splash_cymbal': 55,
        'cowbell': 56,
        'crash_cymbal2': 57,
        'vibraslap': 58,
        'ride_cymbal2': 59,
        'hi_bongo': 60,
        'low_bongo': 61,
        'mute_hi_bongo': 62,
        'open_hi_bongo': 63,
        'low_conga': 64,
        'high_timbale': 65,
        'low_timbale': 66,
        'high_agogo': 67,
        'low_agogo': 68,
        'cabasa': 69,
        'maracas': 70,
        'short_whistle': 71,
        'long_whistle': 72,
        'short_guiro': 73,
        'long_guiro': 74,
        'claves': 75,
        'hi_wood_block': 76,
        'low_wood_block': 77,
        'mute_cuica': 78,
        'open_cuica': 79,
        'mute_triangle': 80,
        'open_triangle': 81
    }
    return drum_dict
```
可见从35到81全是对应的鼓的样式，对普通编曲而言肯定是够了。

## 编程实现
有了专属的频道，那么我们就能够很轻松地来创作自己的鼓组音轨了，为了方便我写了一个添加鼓点的函数，供大家参考：
```python
def add_drum(name, time, track, delay=0, velocity=1):
    bpm = get_bpm(track)
    meta_time = 60 * 60 * 10 / bpm
    drum_dict = get_drum_dict()
    try:
        note = drum_dict[name]
    except:
        print(traceback.format_exc())
        return
    track.append(Message('note_on', note=note, velocity=round(64 * velocity), time=delay, channel=9))
    track.append(
        Message('note_off', note=note, velocity=round(64 * velocity), time=round(meta_time * time), channel=9))
```
为了让自己的鼓点更加丰富一点，我添加了两个鼓的音轨，一个用于踩镲（Hi-Hat），主要是用于基本节奏的把握，另一个是军鼓（Snare）、嗵鼓（Tom）和强音钹（Crash Cymbal），用于使节奏更加Funky和Groovy，并在其他乐器空白间隙加入过门，使得音乐更加丰满。
- 首先是镲轨，我编写得十分简单：
```python
def hi_hat(track):
    for i in range(8):
        add_drum('open_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
        add_drum('open_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
    for i in range(16):
        add_drum('open_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.25, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.25, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.25, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.25, track, velocity=0.6)
        add_drum('open_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
```
- 然后是玩花儿的轨，我用了两个函数来实现：
```python
def tom_and_snare_pt1(track):
    for i in range(7):
        add_drum('acoustic_snare', 0.5, track, velocity=0.8)
        add_drum('low_tom', 0.5, track, velocity=0.6)
        add_drum('low-mid_tom', 0.5, track, velocity=0.6)
        add_drum('acoustic_snare', 0.5, track, velocity=0.6)
        add_drum('low_tom', 0.25, track, velocity=0.8)
        add_drum('acoustic_snare', 0.5, track, velocity=0.8)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)
    add_drum('acoustic_snare', 0.5, track, velocity=0.8)
    add_drum('low_tom', 0.25, track, velocity=0.6)
    add_drum('low-mid_tom', 0.25, track, velocity=0.6)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('acoustic_snare', 0.125, track, velocity=0.8)
    add_drum('acoustic_snare', 0.125, track, velocity=0.8)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)


def tom_and_snare_pt2(track, num):
    for i in range(3): #6
        add_drum('acoustic_snare', 0.25, track, velocity=0.8)
        add_drum('acoustic_snare', 0.25, track, velocity=0.8)
        add_drum('low_tom', 0.5, track, velocity=0.6)
        add_drum('high_tom', 0.25, track, velocity=0.6)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)

        add_drum('acoustic_snare', 0.5, track, velocity=0.6)
        add_drum('low_tom', 0.25, track, velocity=0.8)
        add_drum('acoustic_snare', 0.5, track, velocity=0.9)
        add_drum('low-mid_tom', 0.125, track, velocity=0.6)
        add_drum('hi-mid_tom', 0.125, track, velocity=0.6)

        add_drum('acoustic_snare', 0.5, track, velocity=0.8)
        add_drum('low_tom', 0.25, track, velocity=0.6)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)
        add_drum('high_tom', 0.25, track, velocity=0.6)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)

        add_drum('acoustic_snare', 0.125, track, velocity=0.8)
        add_drum('acoustic_snare', 0.125, track, velocity=0.8)
        add_drum('acoustic_snare', 0.125, track, velocity=0.8)
        add_drum('acoustic_snare', 0.125, track, velocity=0.8)
        add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
        add_drum('high_tom', 0.25, track, velocity=0.6)
        add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)

    add_drum('acoustic_snare', 0.5, track, velocity=0.8)
    add_drum('low_tom', 0.25, track, velocity=0.6)
    add_drum('low-mid_tom', 0.25, track, velocity=0.6)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('acoustic_snare', 0.125, track, velocity=0.8)
    add_drum('acoustic_snare', 0.125, track, velocity=0.8)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)

    add_drum('high_tom', 0.125, track, velocity=0.9)
    add_drum('high_tom', 0.125, track, velocity=0.9)
    add_drum('hi-mid_tom', 0.125, track, velocity=0.9)
    add_drum('hi-mid_tom', 0.125, track, velocity=0.9)
    add_drum('low-mid_tom', 0.125, track, velocity=0.9)
    add_drum('low-mid_tom', 0.125, track, velocity=0.9)
    add_drum('low_tom', 0.125, track, velocity=0.9)
    add_drum('low_tom', 0.125, track, velocity=0.9)

    add_drum('high_floor_tom', 0.25, track, velocity=1.1)
    add_drum('high_floor_tom', 0.25, track, velocity=1.1)
    add_drum('low_floor_tom', 0.25, track, velocity=1.2)
    add_drum('low_floor_tom', 0.25, track, velocity=1.2)

    if num == 1:
        add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
        add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
        add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
        add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
    if num == 2:
        add_drum('low_floor_tom', 0.25, track, velocity=1.3)
        add_drum('low_floor_tom', 0.25, track, velocity=1.3)
        add_drum('chinese_cymbal', 1, track, velocity=1.9)
```
鼓组音轨的MIDI文件可参考 [drum.mid](https://github.com/Truedick23/MusicCritique/blob/master/music/midi/write/drum.mid)
完整项目可参考 [Github - Truedick23/MusicCritique
](https://github.com/Truedick23/MusicCritique)
## 参考资料
- [MIDI Tutorial](http://www.music-software-development.com/midi-tutorial.html)
- [GM 1 Sound Set ](https://www.midi.org/specifications/item/gm-level-1-sound-set)

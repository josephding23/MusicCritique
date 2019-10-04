# Python编曲实践（三）：如何模拟“弯音轮”实现滑音和颤音效果 
===============
## 前言
弯音轮，是在MIDI键盘或专业电子琴一旁安装的一个装置（如下图）。
![弯音轮](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZS5taWRpZmFuLmNvbS9kYXRhL2F0dGFjaC9hbGJ1bS8yMDEyLzEwMDkvMzBfMV8xMzQ5Nzk5Njk2LmpwZw?x-oss-process=image/format,png)
通过前后拨动滚轮，可以实现弯音和颤音的效果。这对于追求特殊电音效果的作曲者来说是必不可少的，而这两个技巧也是吉他等乐器演奏时十分常用的技巧，故在编程中学会更加自然和协调的模拟弯音和颤音效果，是模拟吉他等乐器时必不可少的。
## 再谈Message
我们第一篇文章已经简单讲过Message类是MIDI编曲中最为重要的概念，地位和作用相当于人体的细胞。
再次参考 [Mido官方文档中的Message Type章节](https://mido.readthedocs.io/en/latest/message_types.html) ，我们可以看到在所有的Message种类中有pitchwheel，它便是用于模拟弯音轮效果的一种消息类别，也是实现滑音和颤音的关键。
Pitchwheel类Message的基本格式如下：
```python
Message('pitchwheel', pitch, time, channel)
```
其中time和channel的意义同之前相同，而pitch参数是一个区间为-8192到8192的整数，用于表示音高“弯曲”的程度，取正数时趋向于高音，取负数时趋向于低音。pitch取3000的时候效果是“弯曲”一个半音。
若要实现完整的滑音过程，我们还需要Aftertouch这个类型的Message类型，其基本格式如下：
```python
Message('aftertouch', time, channel, ...)
```
这种Message是用于在音符按下且未结束的时候改变某些属性，比如音量和频道等，在此我们仅仅用它来维持我们的音高。

## 编程实现
我们的目标是通过Pitchwheel这一种Message类型实现两种效果——滑音和颤音，故我们对这两种效果分别编码，将相关的代码添加到改名后的  [实践（一）](https://blog.csdn.net/TruedickDing/article/details/101780003)的play_note函数——add_note函数中：
```python
def add_note(note, length, track, base_num=0, delay=0, velocity=1.0, channel=0, pitch_type=0, tremble_setting=None, bend_setting=None):
    bpm = get_bpm(track)
    meta_time = 60 * 60 * 10 / bpm
    major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
    base_note = 60
    if pitch_type == 0: # No Pitch Wheel Message
        track.append(Message('note_on', note=base_note + base_num*12 + sum(major_notes[0:note]), velocity=round(64*velocity), time=round(delay*meta_time), channel=channel))
        track.append(Message('note_off', note=base_note + base_num*12 + sum(major_notes[0:note]), velocity=round(64*velocity), time=round(meta_time*length), channel=channel))
    elif pitch_type == 1: # Tremble
        try:
            pitch = tremble_setting['pitch']
            wheel_times = tremble_setting['wheel_times']
            track.append(Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity),
                                 time=round(delay * meta_time), channel=channel))
            for i in range(wheel_times):
                track.append(Message('pitchwheel', pitch=pitch, time=round(meta_time * length / (2 * wheel_times)),
                                     channel=channel))
                track.append(Message('pitchwheel', pitch=0, time=0, channel=channel))
                track.append(Message('pitchwheel', pitch=-pitch, time=round(meta_time * length / (2 * wheel_times)),
                                     channel=channel))
            track.append(Message('pitchwheel', pitch=0, time=0, channel=channel))
            track.append(Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity), time=0, channel=channel))
        except:
            print(traceback.format_exc())
    elif pitch_type == 2: # Bend
        try:
            pitch = bend_setting['pitch']
            PASDA = bend_setting['PASDA'] # Prepare-Attack-Sustain-Decay-Aftermath (Taken the notion of ADSR)
            prepare_rate = PASDA[0] / sum(PASDA)
            attack_rate = PASDA[1] / sum(PASDA)
            sustain_rate = PASDA[2] / sum(PASDA)
            decay_rate = PASDA[3] / sum(PASDA)
            aftermath_rate = PASDA[4] / sum(PASDA)
            track.append(Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity), time=round(delay * meta_time), channel=channel))
            track.append(Message('aftertouch', time=round(meta_time * length * prepare_rate), channel=channel))
            track.append(Message('pitchwheel', pitch=pitch, time=round(meta_time * length * attack_rate), channel=channel))
            track.append(Message('aftertouch', time=round(meta_time * length * sustain_rate), channel=channel))
            track.append(Message('pitchwheel', pitch=0, time=round(meta_time * length * decay_rate), channel=channel))
            track.append(Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity), time=round(meta_time * length * aftermath_rate), channel=channel))
        except:
            print(traceback.format_exc())
```
根据pitch_type的值，我们将函数分为三部分：
-  pitch_type为0，代表没有附加效果，同之前的play_note效果一样。
- pitch_type为1，代表添加颤音效果，即吉他中的揉弦。产生这一效果的两个参数pitch和wheel_time通过tremble_setting传入，分别表示颤音的幅度和颤音的次数。根据我的实践来看，一个全音符跟随3至4次颤音是比较自然的；而pitch的赋值也应适中，在1000左右比较合适，太小则看不出效果，太大则会跳动到另一个音符，很不自然。
- pitch_type为2，代表滑音效果，即吉他中的推弦。由于这一效果的变化十分多样，故我参考电子合成音乐中的 [ADSR(Attack Decay Sustain Release) ](https://www.wikiaudio.org/adsr-envelope/)属性，自己设计了一个 PASDA(Prepare - Attack - Sustain - Decay - Aftermath) 属性，即 ***初始音 - 向目标音行进过程中 - 滑到目标音后保持 - 向初始音行进过程中 - 初始音***，这样就可以比较好地表示滑音的属性了，可以参考下图来进行理解：
![pasda](https://img-blog.csdnimg.cn/2019100319552849.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RydWVkaWNrRGluZw==,size_16,color_FFFFFF,t_70)
根据PASDA不同阶段所占比例的大小，我们就能很好地构建出心怡的滑音效果。
之后我们就可以对原始的音乐进行改进：
```python
def verse(track):
    add_note(1, 0.5, track)       # 小
    add_note(1, 0.5, track, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.1, 0.3, 2, 0.3, 0]})       # 时
    add_note(1, 1.5, track, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})       # 候
    add_note(7, 0.25, track, -1)  # 妈
    add_note(6, 0.25, track, -1)  # 妈

    add_note(5, 0.5, track, -1, channel=1)  # 对
    add_note(2, 0.5, track, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.1, 0.8, 2, 0, 0]})      # 我
    add_note(3, 2, track, channel=1, pitch_type=1, tremble_setting={'pitch': 640, 'wheel_times': 8})        # 讲

    add_note(3, 0.5, track)           # 大
    add_note(3, 0.5, track, pitch_type=2, bend_setting={'pitch': 3000, 'PASDA': [0.1, 0.8, 2, 0.3, 0]})
    add_note(3, 1.5, track, pitch_type=1, tremble_setting={'pitch': 400, 'wheel_times': 6})           # 海
    add_note(2, 0.25, track)          # 就
    add_note(1, 0.25, track)          # 是

    add_note(6, 0.5, track, -1, channel=1)  # 我
    add_note(1, 0.5, track, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.2, 0.8, 2, 0, 0]})      # 故
    add_note(2, 2, track, channel=1, pitch_type=1, tremble_setting={'pitch': 600, 'wheel_times': 8})        # 乡

    add_note(7, 0.5, track, -1)  # 海
    add_note(1, 0.5, track)
    add_note(7, 1.5, track, -1, tremble_setting={'pitch': 500, 'wheel_times': 6})  # 边
    add_note(6, 0.25, track, -1)
    add_note(5, 0.25, track, -1)

    add_note(5, 0.5, track, -1, channel=1)  # 出
    add_note(1, 0.5, track, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.2, 1.5, 3, 0, 0]})
    add_note(2, 2, track, channel=1, pitch_type=1, tremble_setting={'pitch': 400, 'wheel_times': 8})        # 生

    add_note(3, 1.5, track, pitch_type=2, bend_setting={'pitch': 3000, 'PASDA': [0, 0.3, 3, 0, 0]})       # 海
    add_note(3, 0.5, track)       # 里
    add_note(1, 0.5, track, channel=1)       # 成
    add_note(6, 0.5, track, -1, channel=1)

    add_note(1, 3, track, channel=1, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})         # 长

```
完整代码见 [Github](https://github.com/Truedick23/MusicCritique/blob/master/write/mother_ocean.py)

## 参考资料
- [MIDI Tutorial](http://www.music-software-development.com/midi-tutorial.html)
- [ADSR - Wikipedia](https://en.wikipedia.org/wiki/ADSR)

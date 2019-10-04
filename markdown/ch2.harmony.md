Python编曲实践（二）：和弦的实现和进行 
=================
## 前言
[上一篇文章](https://blog.csdn.net/TruedickDing/article/details/101780003) 中我简单介绍了如何使用Mido这个库在Python中实现MIDI编程，分享了我的经验和心得，但是单音轨的纯音符堆砌听起来太单薄了，故本篇文章介绍如何轻松添加和弦音轨，使得乐曲更加饱满、丰富。
## 和弦的背景知识
和弦，简单来讲就是三个及以上的不同音高的音符的组合。这些音符可以同时演奏，形成混响的效果，也可以按照一定的组合方式交替演奏。最常见的和弦有两种：三和弦和七和弦，前者由三个音符组成，又分为四种；后者由四个音符组成，又分为8种，具体分类见下面的表格：
名称 | 音程
------|------
大三和弦|大三度、纯五度
小三和弦|小三度、纯五度
增三和弦|大三度、增五度
减三和弦|小三度、减五度

名称|音程
--|--
大七和弦|大三度、完全五度、大七度
小七和弦|小三度、完全五度、小七度
属七和弦（大小七和弦）|大三度、完全五度、小七度
小大七和弦|小三度、完全五度、大七度
半减七和弦|小三度、减五度、减七度
减七和弦|小三度、减五度、减七度
增七和弦|大三度、增五度、小七度
增大七和弦|大三度、增五度、大七度
下面这个图是 [维基百科](https://zh.wikipedia.org/zh-hans/%E5%8D%81%E4%BA%8C%E5%B9%B3%E5%9D%87%E5%BE%8B) 的十二平均律表格，显示了不同音程名称与间隔半音数的对应关系：
![十二平均律](https://img-blog.csdnimg.cn/20191003155458203.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RydWVkaWNrRGluZw==,size_16,color_FFFFFF,t_70)

结合十二平均律的音程名称和间隔半音数的对照关系，我们就可以动手写和弦了.

## 编程实现
首先编写一个方程用于得到不同和弦对应的音程关系：
```python
def get_chord_arrangement(name):
    chord_dict = {
        'maj3': [0, 4, 7, 0],  # 大三和弦 根音-大三度-纯五度
        'min3': [0, 3, 7, 0],  # 小三和弦 根音-小三度-纯五度
        'aug3': [0, 4, 8, 0],  # 增三和弦 根音-大三度-增五度
        'dim3': [0, 3, 6, 0],  # 减三和弦 根音-小三度-减五度

        'M7': [0, 4, 7, 11],  # 大七和弦 根音-大三度-纯五度-大七度
        'Mm7': [0, 4, 7, 10],  # 属七和弦 根音-大三度-纯五度-小七度
        'm7': [0, 3, 7, 10],  # 小七和弦 根音-小三度-纯五度-小七度
        'mM7': [0, 3, 7, 11],  # 小大七和弦 根音-小三度-纯五度-大七度
        'aug7': [0, 4, 8, 10],  # 增七和弦 根音-大三度-增五度-小七度
        'augM7': [0, 4, 8, 11],  # 增大七和弦 根音-大三度-增五度-小七度
        'm7b5': [0, 3, 6, 10],  # 半减七和弦 根音-小三度-减五度-减七度
        'dim7': [0, 3, 6, 9]  # 减减七和弦 根音-小三度-减五度-减七度
    }

    chord = [0, 0, 0, 0]
    try:
        chord = chord_dict[name]
    except:
        print(traceback.format_exc())
    return chord
```
返回值是一个长度为4的一维数组，每一个值表示这个音符与根音相差的半音数。
然后编写一个用于将和弦添加到音轨中的函数add_chord：
```python
def add_chord(root, name, format, length, track, root_base=0, channel=3):
    bpm = get_bpm(track)
    major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    notes_dict = {}
    for i, note in enumerate(notes):
        notes_dict[note] = 60 + sum(major_notes[0:i+1])
    root_note = notes_dict[root] + root_base*12
    chord = get_chord_arrangement(name)
    meta_time = 60 * 60 * 10 / bpm
    time = round(length / len(format) * meta_time)

    for dis in format:
        note = root_note + chord[dis]
        track.append(Message('note_on', note=note, velocity=56, time=0, channel=channel))
        track.append(Message('note_off', note=note, velocity=56, time=time, channel=channel))
```
在参数中：
- root是根音的名字，这里我们考虑的音符是在notes数组中给出，暂时不考虑升降音为根音的情况；root_base是根音所在的音域；
- format是用于演奏和弦的方式，用一维数组表示，数组表示和弦中第几个音符，比如 [0, 1,  2, 3, 2, 1] ；
- length是整个和弦演奏的时长，用占全音符长度的数量来表示
通过这个函数中的循环，我们可以按照format指示的方式将和弦添加到MIDI音轨之中。

对于《大海啊，故乡》，我按照 [网上的吉他谱](http://yinyuesheng.cn/jita/pu/68133.html) 为它添加了如下的和弦模式：
```python
def chord(track):
    format = [0, 1, 2, 3, 2, 1]
    add_chord('C', 'M7', format, 3, track)
    add_chord('E', 'm7', format, 3, track)
    add_chord('C', 'M7', format, 3, track)
    add_chord('D', 'm7', format, 3, track)
    add_chord('G', 'Mm7', format, 3, track, -1)
    add_chord('D', 'm7', format, 3, track)
    add_chord('F', 'M7', format, 3, track, -1)
    add_chord('C', 'M7', format, 3,  track)

    add_chord('C', 'M7', format, 3,  track)
    add_chord('G', 'M7', format, 3, track, -1)
    add_chord('F', 'M7', format, 3, track, -1)
    add_chord('G', 'M7', format, 3, track, -1)
    add_chord('C', 'M7', format, 3, track)
    add_chord('D', 'm7', format, 3, track)
    add_chord('G', 'Mm7', format, 3, track)
    add_chord('C', 'M7', format, 3, track)

    add_chord('C', 'M7', format, 3,  track)
    add_chord('G', 'M7', format, 3, track, -1)
    add_chord('F', 'M7', format, 3, track, -1)
    add_chord('G', 'M7', format, 3, track, -1)
    add_chord('C', 'M7', format, 3, track)
    add_chord('D', 'm7', format, 3, track)
    add_chord('G', 'Mm7', format, 3, track)
    add_chord('C', 'M7', format, 3, track)
```
## 总结
加上和弦之后，我的MIDI文件在MidiEditor中显示如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019100316225690.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RydWVkaWNrRGluZw==,size_16,color_FFFFFF,t_70)感谢阅读，完整的项目代码详见 [Github - Truedick23/MusicCritique
](https://github.com/Truedick23/MusicCritique)
## 参考资料
[和弦 - 维基百科](https://zh.wikipedia.org/zh-hans/%E5%92%8C%E5%BC%A6)
[十二平均律 - 维基百科](https://zh.wikipedia.org/zh-hans/%E5%8D%81%E4%BA%8C%E5%B9%B3%E5%9D%87%E5%BE%8B)

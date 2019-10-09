Python编曲实践（一）：通过Mido和PyGame来编写和播放单轨MIDI文件 
======================
## 前言
人工智能编曲是一个十分复杂的话题，而这一话题的起点便是选择一个良好的编曲媒介，使得开发者能够将AI的音乐灵感记录下来，并且能够很方便地将其播放、编辑、分享。
MIDI文件是电脑编曲的一种通用格式，它容易通过音乐编辑软件导入、导出，也有很多现成的库函数来对其进行编辑加工。
首先，我找到了[PythonWiki提供的音乐库合集 - PythonInMusic](https://wiki.python.org/moin/PythonInMusic)，在这里上百个库之中，仅有寥寥几个是支持Python3且仍有活力的，在其中[Mido](https://wiki.python.org/moin/PythonInMusic)和[PyGame.midi](https://www.pygame.org/docs/ref/midi.html)库是其中比较好用的两个库，本篇文章就采用这两个库来进行MIDI文件的编写和播放

## Mido编曲
关于用Mido库来创建一个新的MIDI文件，[官方文档](https://mido.readthedocs.io/en/latest/midi_files.html#creating-a-new-file)给出了如下示例代码：

```python
from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=12, time=0))
track.append(Message('note_on', note=64, velocity=64, time=32))
track.append(Message('note_off', note=64, velocity=127, time=32))

mid.save('new_song.mid')
```
这段示例代码虽然短，可是已经将编写MIDI文件的基本思路完全表达出来了：

 - 首先创建一个MidiFile对象
 - 创建一个（或多个）MidiTrack对象，并将其append到MidiFile中
 - 向一个（或多个）MidiTrack对象内添加Message对象（包括program_change、note_on、note_off等）和MetaMessage对象（用以表示MIDI文件的节拍、速度、调式等属性）
 - 保存MidiFile对象

下面我通过对Message和MetaMessage这两个十分重要的概念的进一步说明，来加深大家理解

### Message
Message对象的类型十分复杂，是根据MIDI文件的格式实现的，[官方文档有详细列表](https://mido.readthedocs.io/en/latest/message_types.html)，在此我们不一一列举，而仅对我使用到的三种Message来进行分析：

 #### 1. control_change
 program_change是用于更改不同channel的乐器音色的，格式为：
 

```python
Message('program_change', channel, program, time=0)
```
- channel是指定的0~15的一个值，因为MIDI文件给我们提供了默认的16个通道，通过这个值可以选择更改乐器的通道编号；
- program对于乐器编号，点[此](https://blog.csdn.net/ruyulin/article/details/84103186)可以查到不同乐器对应的编号
#### 2.note_on
note_on消息，可以理解为音符的开始，其格式为
```python
Message('note_on', note, velocity, time, channel)
```
- 其中note是0\~127的一个数字，代表音符的高低，通过实践证明60代表的音高是C4，仅供参考；
- velocity代表音强，也是0\~127的一个数字，默认为64，若要体现音符强度的变化可以修改它；
- time是时间变量，是十分复杂的一个参数，在note_on信息这里可以理解为该音符写在前一个音符结束多久之后，单位是微秒（ms）；
- channel同上一个函数一样，代表通道的编号，即将这个音符写到哪个通道之上，这可能起到更改乐器的效果
#### 3.note_off
note_off消息，可以理解为音符的结束，一般紧跟在note_on消息之后，其格式与上面的相同
```python
Message('note_off', note, velocity, time, channel)
```
- note参数与note_on消息保持一致，否则有可能不能成功写入
- velocity同note_on保持一致就好
- time在此处表示的意义是音符的持续时间，也是以微秒（ms）为单位
- channel也是表示通道号，与note_on保持相同即可

### MetaMessage
MetaMessage的种类也很多，可以参考[官方文档](https://mido.readthedocs.io/en/latest/message_types.html)，我只使用了3种MetaMessage，列举在下面：
```python
	tempo = 75
    tempo = mido.bpm2tempo(bpm)
    meta_time = MetaMessage('time_signature', numerator=3, denominator=4)
    meta_tempo = MetaMessage('set_tempo', tempo = tempo, time=0)
    meta_tone = MetaMessage('key_signature', key='C')
```
- 其中time_signature是对于节拍的表示，在此处即3/4，参数以分子和分母来命名，十分清晰
- set_tempo是用于设置音乐的节奏快慢，由于这里tempo的单位不是BPM（Beat Per Minute），故一般配合bpm2tempo来使用
- key_signature是用于设置音乐的调式的，在此处我设置为C大调，若是小调的话仅需要在后面添加小写字母m，如Cm表示C小调

## 编程实现
 #### 1. play_note函数
 由于Message对象需要的参数比较多而且单位转换复杂繁琐，故我自己编写了一个play_note函数来更加方便编曲：
 ```python
 def play_note(note, length, track, base_num=0, delay=0, velocity=1.0, channel=0):
    meta_time = 60 * 60 * 10 / bpm
    major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
    base_note = 60
    track.append(Message('note_on', note=base_note + base_num*12 + sum(major_notes[0:note]), velocity=round(64*velocity), time=round(delay*meta_time), channel=channel))
    track.append(Message('note_off', note=base_note + base_num*12 + sum(major_notes[0:note]), velocity=round(64*velocity), time=round(meta_time*length), channel=channel))
 ```
- 由于我要编的歌曲是大调曲式，而大调的音阶结构是“全全半全全全半”（这一规律可以通过钢琴键盘的黑白键安排来得到，在此不赘述乐理知识），故我创建一个major_notes数组，用于根据根音计算出某一个音符的音高；
- meta_time是根据bpm而计算出的每个节拍的时间长度，用于得到Message中的time参数
- base_note是通过实验得到的C4的音高，作为根音来搭配major_notes得到每个音符的音高
- base_num用于切换目前所在的音域，负值表示低几度，正值表示高几度
- velocity是一个0~2的浮点数，以64为基准来进行比较

 #### 2. 编曲
 下面开始正式编曲了，我选择的是《大海啊，故乡》这首歌，简谱如下：
 ![大海啊，故乡](https://imgconvert.csdnimg.cn/aHR0cDovL3d3dy5nZXB1d2FuZy5uZXQvZC9maWxlL3AvMjAxNjEyMTgvYTk3Nzg2ZmI0ZTRkZmExMDY5NGQyZDQ3NzZhNDM3YmEuanBn?x-oss-process=image/format,png)
 
  由于我们是纯乐器演奏，而前奏与后面重复率极高，故略过前奏。之后我将此音乐以八小节为单位分为3个部分，其中后两部分仅一个半音部分有区别。根据此特征，我编写了chorus和verse两个函数，代码如下：
 ```python
 def verse(track):
    play_note(1, 0.5, track)       # 小
    play_note(2, 0.5, track)       # 时
    play_note(1, 1.5, track)       # 候
    play_note(7, 0.25, track, -1)  # 妈
    play_note(6, 0.25, track, -1)  # 妈

    play_note(5, 0.5, track, -1, channel=1)  # 对
    play_note(3, 0.5, track, channel=1)      # 我
    play_note(3, 2, track, channel=1)        # 讲

    play_note(3, 0.5, track)           # 大
    play_note(4, 0.5, track)
    play_note(3, 1.5, track)           # 海
    play_note(2, 0.25, track)          # 就
    play_note(1, 0.25, track)          # 是

    play_note(6, 0.5, track, -1, channel=1)  # 我
    play_note(2, 0.5, track, channel=1)      # 故
    play_note(2, 2, track, channel=1)        # 乡

    play_note(7, 0.5, track, -1)  # 海
    play_note(1, 0.5, track)
    play_note(7, 1.5, track, -1)  # 边
    play_note(6, 0.25, track, -1)
    play_note(5, 0.25, track, -1)

    play_note(5, 0.5, track, -1, channel=1)  # 出
    play_note(2, 0.5, track, channel=1)
    play_note(2, 2, track, channel=1)        # 生

    play_note(4, 1.5, track)       # 海
    play_note(3, 0.5, track)       # 里
    play_note(1, 0.5, track)       # 成
    play_note(6, 0.5, track, -1)

    play_note(1, 3, track)         # 长


def chorus(track, num):
    play_note(5, 0.5, track)  # 大
    play_note(6, 0.5, track)
    play_note(5, 1.5, track)  # 海
    play_note(3, 0.5, track)  # 啊

    play_note(5, 0.5, track, channel=1)  # 大
    play_note(6, 0.5, track, channel=1)
    play_note(5, 2, track, channel=1)    # 海

    play_note(6, 0.5, track)  # 是（就）
    play_note(5, 0.5, track)  # 我（像）
    play_note(4, 0.5, track)  # 生（妈）
    if num == 1:
        play_note(1, 0.25, track, channel=1) # 活
        play_note(1, 0.25, track, channel=1) # 的
    if num == 2:
        play_note(1, 0.5, track, channel=1)  # (妈)
    play_note(6, 0.5, track, channel=1)      # 地（一）
    play_note(5, 0.5, track, channel=1)

    play_note(5, 3, track, channel=1)        # 方（样）

    play_note(3, 0.5, track)  # 海（走）
    play_note(4, 0.5, track)  # 风（遍）
    play_note(3, 1.5, track)  # 吹（天）
    play_note(2, 0.25, track) # (涯)
    play_note(1, 0.25, track)

    play_note(6, 0.5, track, -1, channel=1)  # 海（海）
    play_note(2, 0.5, track, channel=1)      # 浪
    play_note(2, 2, track, channel=1)        # 涌（角）

    play_note(4, 0.5, track)              # 随（总）
    play_note(5, 0.5, track)              # 我（在）
    play_note(4, 0.5, track)              # 漂（我）
    play_note(3, 0.5, track)              # 流（的）
    play_note(1, 0.5, track, channel=1)   # 四（身）
    play_note(6, 0.5, track, -1, channel=1)

    play_note(1, 3, track, channel=1)     # 方（旁）
 ```
 #### 3. play_midi函数
 只能编曲而不会识曲的AI不是好AI，PyGame的midi模块为我们提供了一个很好的播放midi的功能，由于代码非原创，故仅仅贴出这个函数：
 ```python
 def play_midi(file):
    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 1024
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(1)
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(file)
    except:
        import traceback
        print(traceback.format_exc())
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)
 ```
 ## 总结
 - 至此编曲工作已经告一段落，顺便向大家推荐一款免费MIDI播放与编辑软件[MidiEditor](https://www.midieditor.org/)。虽然没有Pro Tools和Cubase等专业编曲软件的全面功能，但是对于电脑编曲的基本需求而言足够了，我们的作品在MidiEditor像这样：
 ![MidiEditor](https://img-blog.csdnimg.cn/20190930221312877.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RydWVkaWNrRGluZw==,size_16,color_FFFFFF,t_70) 
 - 为了庆祝Davie504订阅数破四百万，我使用两种贝斯作为乐器。
 - 单音轨的音乐听起来还是比较单薄，这篇文章也是我进行智能编曲的尝试和敲门砖，争取之后能够使用更简便的方法做出更复杂更动听的音乐，谢谢关注！
 - 完整工程见 [Github](https://github.com/Truedick23/MusicCritique)

## 参考资料
1. [Mido官方文档](https://mido.readthedocs.io/en/latest/index.html)
2. [PyGame播放MIDI文件参考](https://gist.github.com/guitarmanvt/3b6a91cefb2f5c3098ed)
3. [简谱来源](http://www.gepuwang.net/e/php/img.php?id=45920)
4. [MIDI Messages 深入解析](https://www.midi.org/articles-old/about-midi-part-3-midi-messages)
5. [MIDI音色代码](https://blog.csdn.net/ruyulin/article/details/84103186)

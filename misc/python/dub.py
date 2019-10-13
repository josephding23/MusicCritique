from pydub import AudioSegment

AudioSegment.converter = 'C:/ffmpeg/bin/ffmpeg.exe'
AudioSegment.ffmpeg = 'C:/ffmpeg/bin/ffmpeg.exe'
AudioSegment.ffprobe = 'C:/ffmpeg/bin/ffprobe.exe'


song = AudioSegment.from_mp3('D:/PycharmProjects/MusicCritique/data/mp3/roundabout_early.mp3')

ten_seconds = 10 * 1000
five_seconds = 5 * 1000

last_10_seconds = song[:-ten_seconds]
first_10_seconds = song[ten_seconds:]

beginning = first_10_seconds + 6
end = last_10_seconds - 3

without_the_middle = beginning + end

print(without_the_middle.duration_seconds)
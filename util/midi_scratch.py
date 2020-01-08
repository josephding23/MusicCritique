import requests
from bs4 import BeautifulSoup
import traceback
from random import choice, uniform
import os
from pymongo import MongoClient
import rarfile
import time
import socket
from urllib import request
import mido
import http.cookiejar

myHeaders = ["Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
             "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
             "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
             "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
             "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
             "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
             "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
             "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
             "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
             "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
             "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
             "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
             "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
             "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
             "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
             "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52"]

cookie_str = '_ga=GA1.2.279200440.1578048264; _GPSLSC=iUzgdaN6J2; _gid=GA1.2.2026261794.1578441620; PHPSESSID=rrmoq7gtgjv1636qp1nr1i9783; _gat=1'
cookie_dict = {
    '_ga': 'GA1.2.279200440.1578048264',
    '_GPSLSC': 'iUzgdaN6J2',
    'PHPSESSID': 'rrmoq7gtgjv1636qp1nr1i9783',
    '_gid': 'GA1.2.2026261794.1578441620',
    '_gat': '1'
}
cookies = requests.utils.cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True)

def get_performer_collection():
    client = MongoClient(connect=False)
    return client.free_midi.performers

def get_genre_collection():
    client = MongoClient(connect=False)
    return client.free_midi.genres

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def get_html_text(url, params):
    global attributeErrorNum, httpErrorNum
    try:
        proxy = {'https:': '127.0.0.1:1080', 'http:': '127.0.0.1:1080'}
        r = requests.get(url, proxies=proxy)

        r.headers = params
        r.encoding = 'utf-8'
        status = r.status_code
        if status != 200:
            print('404', url)
            return ''
        return r.text
    # ['HTTPError', 'AttributeError', 'TypeError', 'InvalidIMDB']
    except:
        print(url)
        print(traceback.format_exc())

def get_all_performers_url(url):
    text = get_html_text(url)
    soup = BeautifulSoup(text, 'html.parser')
    performers_collection = get_performer_collection()

    for item in soup.find_all(name='li'):
        try:
            src = item.a['href']
            if src[0] == '/' and src[-3:] == 'php' and src not in ['/index.php', '/disclaimer.php', '/contact.php', '/free_rock_emoticons_avatars.php']:
                print(url[:-1] + src, item.a.text)

                performers_collection.insert_one({
                    'Url': url[:-1] + src,
                    'Name': item.a.text,
                    'Finished': False
                })
        except:
            pass

def get_midi_from_performer(url, name):
    text = get_html_text(url)
    soup = BeautifulSoup(text, 'html.parser')
    midi_collection = get_midi_collection()
    songs = []
    for item in soup.find_all(name='a'):
        try:
            url = item['href']
            if url[-3:] == 'rar':
                # print(item.text)
                songs.append('http://rock.freemidis.net/' + url)
        except:
            pass
    num = len(songs)
    for song in songs:
        print(song, name)
        midi_collection.insert_one({
            'Url': song,
            'Performer': name,
            'Downloaded': False
        })
    return num


def get_all_performers():
    performers_collection = get_performer_collection()
    for performer in performers_collection.find({'Finished': False}):
        name = performer['Name']
        url = performer['Url']
        num = get_midi_from_performer(url, name)
        performers_collection.update_one(
            {'Url': url},
            {'$set': {'Finished': True, 'Num': num}}
        )

def download_midi():
    root_dir = 'E:/MIDI_files'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'Downloaded': False}):
        url = midi['Url']
        performer = midi['Performer'].replace('/', '_')
        path = root_dir +  '/' + performer + ' - ' + url.split('/')[-1]
        print(path)
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
            'Accept': 'application/signed-exchange;v=b3;q=0.9,*/*;q=0.8',
            'Referer': 'http://rock.freemidis.net/europe_midi_europe_midis.php',
            'Accept-Encoding': 'gzip, deflate',
            'Proxy-Connection': 'keep-alive'
        }
        socket.setdefaulttimeout(3)
        try:
            wb = request.Request(url, headers=header)
            f = request.urlopen(url=wb).read()
            open(path, 'wb').write(f)

            midi_collection.update_one(
                {'Url': url},
                {'$set': {'Downloaded': True}}
            )
            print('Progress: {:.2%}'.format(midi_collection.count({'Downloaded': True}) / midi_collection.count()))
        except:
            continue


def unzip_all():
    root_dir = 'E:/MIDI_files'
    os.chdir(root_dir)
    for path in os.listdir(root_dir):
        # print(dir)
        path = root_dir + '/' + path
        if os.path.isfile(path):
            try:
                rar = rarfile.RarFile(path)
                rar.extractall()
                rar.close()
            except:
                print(traceback.format_exc())

def free_midi_hack_download_test():
    download_header = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
        'Referer': 'https://freemidi.org/download3-26199-empire-state-of-mind-part-ii-alicia-keys',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    }

    socket.setdefaulttimeout(3)
    session = requests.session()
    session.params.update(download_header)

    url = 'https://freemidi.org/getter-26199'
    got_cookie = requests.get(url, params=download_header).cookies

    path = './test/free_midi.midi'
    try:
        opener = request.build_opener()
        opener.addheaders = [(key, value) for key, value in download_header.items()]
        request.install_opener(opener)
        # text = get_html_text(url, download_header)
        # soup = BeautifulSoup(text, 'html.parser')
        # print(soup)
        # request.urlretrieve(url, path)
    except:
        print(traceback.format_exc())

get_genres = lambda : ['pop', 'rock', 'hip-hop-rap', 'jazz', 'blues', 'classical', 'rnb-soul',
                       'bluegrass',  'country', 'christian-gospel',  'dance-eletric', 'newage',
                       'reggae-ska',  'folk', 'punk', 'disco', 'metal']

def free_midi_get_genres():
    genres_collection = get_genre_collection()
    for genre in get_genres():
        if genres_collection.count({'name': genre}) != 0:
            continue
        url = 'https://freemidi.org/genre-' + genre
        text = get_html_text(url)
        soup = BeautifulSoup(text, 'html.parser')
        urls = []
        performers = []
        for item in soup.find_all(name='div', attrs={'class': 'genre-link-text'}):
            try:
                href = item.a['href']
                name = item.text
                urls.append(href)
                performers.append(name)
            except:
                pass
        genres_collection.insert_one({
            'name': genre,
            'performers_num': len(urls),
            'performers': performers,
            'performer_urls': urls
        })
        print(genre, len(urls))

def free_midi_get_performers():
    root_url = 'https://freemidi.org/'
    genres_collection = get_genre_collection()
    performers_collection = get_performer_collection()
    for genre in genres_collection.find({'Finished': False}):
        genre_name = genre['Name']
        performers = genre['Performers']
        performer_urls = genre['PerformersUrls']
        num = genre['PerformersNum']
        for index in range(num):
            name = performers[index]
            url = root_url + performer_urls[index]
            print(name, url)

            performers_collection.insert_one({
                'Name': name,
                'Url': url,
                'Genre': genre_name,
                'Finished': False
            })
        genres_collection.update_one(
            {'_id': genre['_id']},
            {'$set': {'Finished': True}})
        print('Progress: {:.2%}\n'.format(genres_collection.count({'Finished': True}) / genres_collection.count()))

def get_free_midi_songs_and_add_performers_info():
    root_url = 'https://freemidi.org/'
    midi_collection = get_midi_collection()
    performer_collection = get_performer_collection()
    while performer_collection.count({'Finished': False}) != 0:
        for performer in performer_collection.find({'Finished': False}):
            num = 0
            performer_url = performer['Url']
            performer_name = performer['Name']
            genre = performer['Genre']
            try:
                params = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
                    'Cookie': cookie,
                    'Referer': root_url + genre,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Connection': 'keep-alive'
                }
                text = get_html_text(performer_url, params)
                if text == '':
                    print('connection error')
                    continue
                soup = BeautifulSoup(text, 'html.parser')
                # print(soup)
                for item in soup.find_all(name='div', attrs={'itemprop': 'tracks'}):
                    try:
                        download_url = root_url + item.span.a['href']
                        name = item.span.text
                        if midi_collection.count({'Genre': genre, 'Name': name}) == 0:
                            midi_collection.insert_one({
                                'Name': name.replace('\n', ''),
                                'DownloadPage': download_url,
                                'Performer': performer_name,
                                'PerformerUrl': performer_url,
                                'Genre': genre,
                                'Downloaded': False
                            })
                        num = num + 1
                    except:
                        pass
                if num != 0:
                    performer_collection.update_one(
                        {'_id': performer['_id']},
                        {'$set': {'Finished': True, 'Num': num}}
                    )
                    time.sleep(uniform(1, 1.6))
                    print('Performer ' + performer_name + ' finished.')
                    print('Progress: {:.2%}\n'.format(performer_collection.count({'Finished': True}) / performer_collection.count()))
            except:
                print('Error connecting.')

def is_valid_midi(path):
    try:
        midi = mido.MidiFile(path)
        return True
    except:
        return False

def output_cookies():
    path = './cookies.txt'
    cookie = http.cookiejar.LWPCookieJar(path)
    handler = request.HTTPCookieProcessor(cookie)
    opener = request.build_opener(handler)
    response = opener.open('https://freemidi.org')
    cookie.save(ignore_discard=True)


def download_free_midi():
    root_url = 'https://freemidi.org/'
    root_path = 'E:/free_MIDI'
    cookie_path = './cookies.txt'
    params = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
        # 'Cookie': cookie,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive'
    }
    midi_collection = get_midi_collection()

    session = requests.Session()
    session.headers.update(params)
    # session.cookies.update(cookie_dict)
    session.cookies = http.cookiejar.LWPCookieJar()
    session.cookies.load(cookie_path, ignore_discard=True)
    # print(session.cookies)
    # session.cookies = cookie
    # session.cookies = cookie_dict
    # opener = request.build_opener(request.HTTPCookieProcessor(cj))
    for midi in midi_collection.find({'Downloaded': False}, no_cursor_timeout = True):
        performer_link = midi['PerformerUrl']
        download_link = midi['DownloadPage']
        name = midi['Name']
        genre = midi['Genre']
        performer = midi['Performer']
        try:
            params = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
                'Cookie': cookie_str,
                'Referer': performer_link,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Connection': 'keep-alive'
            }
            session.headers.update(params)
            # text = get_html_text(download_link, params)
            r = session.get(download_link)
            r.encoding = 'utf-8'
            if r.cookies.get_dict():
                print(r.cookies.get_dict())
                session.cookies = r.cookies
            if r.status_code != 200:
                print('connection error ' + r.status_code)
            soup = BeautifulSoup(r.text, 'html.parser')
            try:
                getter_link = root_url + soup.find(name='a', attrs={'id': 'downloadmidi'})['href']
                print(getter_link)
                download_header = {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Connection': 'keep-alive',
                    'Referer': download_link,
                    # 'Cookie': cookie_str,
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
                }
                session.headers.update(download_header)
                dir = root_path + '/' + genre
                if not os.path.exists(dir):
                    os.mkdir(dir)
                file_name = name + ' - ' + performer + '.mid'
                path = dir + '/' +  file_name
                try:
                    cj = http.cookiejar.LWPCookieJar()
                    cj.load(cookie_path, ignore_discard=True)
                    # cookie_handler = request.HTTPCookieProcessor(cj)
                    # cookie_opener = request.build_opener(cookie_handler)

                    # opener = request.build_opener()
                    # opener.addheaders = [(key, value) for key, value in download_header.items()]
                    # cookie_opener.addheaders = [(key, value) for key, value in download_header.items()]
                    # request.install_opener(opener)
                    # request.urlretrieve(getter_link, path)
                    socket.setdefaulttimeout(4)
                    with open(path, 'wb') as output:
                        r = session.get(getter_link)
                         # print(response)
                        output.write(r.content)
                        if r.cookies.get_dict():
                            print(r.cookies)
                            session.cookies.update(r.cookies)
                    # cookie_opener.open(getter_link)
                    time.sleep(uniform(1, 2))

                    # cj.save(cookie_path, ignore_discard=True)
                    if is_valid_midi(path):
                        print(file_name + ' downloaded')
                        midi_collection.update_one(
                            {'_id': midi['_id']},
                            {'$set': {'Downloaded': True, 'GetterLink': getter_link}}
                        )
                        print('Progress: {:.2%}\n'.format(midi_collection.count({'Downloaded': True}) / midi_collection.count()))
                    else:
                        print('Cannot successfully download midi.')
                        os.remove(path)
                except:
                    print(traceback.format_exc())
            except:
                print('Found no download link')
        except:
            print(traceback.format_exc())

def verify_midi_completeness():
    performer_collection = get_performer_collection()
    midi_collection = get_midi_collection()
    for performer in performer_collection.find():
        should_num = performer['Num']
        name = performer['Name']
        actual_num = midi_collection.count({'Performer': name})
        if should_num != actual_num:
            print(should_num, actual_num, name)

if __name__ == '__main__':
    output_cookies()
    download_free_midi()
    # free_midi_hack_download_test()
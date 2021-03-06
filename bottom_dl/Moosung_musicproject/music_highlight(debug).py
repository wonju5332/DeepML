from librosa import load, stft, feature, get_duration, display as ld
from sklearn.metrics.pairwise import cosine_similarity
from pygame import mixer,init, display, time, quit
from ctypes import windll
from copy import deepcopy
from random import choice
import matplotlib.pyplot as plt
import numpy as np
import os

################## 플레이 정보 ##################
play_duration = 10                # 노래 재생 시간
file_loc = 'c:/python/data/music' # 노래 폴더 위치
show_chroma = True               # 크로마그램 출력
###############################################


################## 노래 클래스 ##################
class Song(object):
    def __init__(self,show=show_chroma):
        self.music_dict = {}  # 노래 이름을 key로, 파일경로를 value로 하는 딕셔너리
        self.music = ''       # 재생할 노래 이름
        self.time = 0         # 노래 총 길이
        self.show = show      # 크로마그램 화면에 출력할지 말지 여부
        self.best_repeated_part = []      # 최종 하이라이트 시간 정보를 담는 변수

    ##### 노래 로딩 메소드 #####

    # (노래 로딩 Main) 노래 로드 메인 메소드(여기서 리턴한 노래의 코사인 유사도 정보를 Analysis() 메소드에서 분석)
    def LoadSong(self):
        self.music_dict = self.SearchSong(file_loc)  # 노래, 파일경로 딕셔너리 생성
        self.music = self.Input()                    # 재생할 노래 이름 입력

        if self.music.upper() == 'RANDOM':           # 랜덤 입력시 노래 리스트 중 하나 랜덤으로 뽑기
            self.music = choice(list(self.music_dict.keys()))

        y, sr = load(self.music_dict[self.music], sr=882) # librosa 모듈의 노래 불러오기(샘플링은 0.5초 당 하나씩)
        s = np.abs(stft(y)**2)                            # 노래의 파워 데이터(주파수의 진폭)
        self.time = get_duration(y=y, sr=sr)              # 노래의 총 길이(샘플링 데이터 갯수)
        chroma = feature.chroma_stft(S=s, sr=sr)          # 크로마그램으로 변환
        print('chroma.  행 : {0}, 열 : {1}\n'.format(len(chroma),len(chroma[0])),chroma)
        self.Chromagram(chroma, show=self.show)  # self.show = True 일 경우 노래 시작 전 크로마그램 그래프를 출력
        chromaT = np.transpose(chroma,axes=(1,0))         # time-time 코사인 유사도 구하기 위해 전치행렬로 변환
        print('chromaT.  행 : {0}, 열 : {1}\n'.format(len(chromaT),len(chromaT[0])),chromaT)
        self.Chromagram(chromaT, show=self.show)  # self.show = True 일 경우 노래 시작 전 크로마그램 그래프를 출력
        print('\nLoading Finished!')
        return cosine_similarity(chromaT)                 # 노래 각 부분의 코사인 유사도를 리턴

    # (노래 로딩 SUB) 폴더에서 mp3 파일 리스트를 뽑아서 딕셔너리에 담는 메소드
    def SearchSong(self, dirname):
        filenames = os.listdir(dirname)                     # 지정된 폴더 내 파일이름들 불러오기
        music_dict = {}
        for filename in filenames:
            full_filename = os.path.join(dirname, filename) # full_filename = 경로+파일이름
            ext = os.path.splitext(full_filename)[-1]       # ext에 확장자 넣기
            file = os.path.splitext(filename)[0]            # file에 확장자를 제외한 파일이름만 넣기
            if ext == '.mp3':                               # 확장자가 mp3 인 파일만 music_dict 딕셔너리에 넣기
                music_dict[file] = full_filename            # 파일이름(key), 경로+파일이름(value)
        return music_dict                                   # music_dict 딕셔너리 리턴

    # (노래 로딩 SUB) 처음에 노래 리스트를 화면에 출력하고 재생할 노래 입력받는 메소드
    def Input(self):
        print('---------------------------------------------------'
              'Music List---------------------------------------------------')
        music_dict_list = list(self.music_dict.keys())
        for idx in range(len(music_dict_list)//5 + 1):
            try:
                print(' '.join([i.ljust(25) for i in music_dict_list[5 * idx : 5 * idx +5]]))
            except IndexError:
                print(' '.join([i.ljust(25) for i in music_dict_list[5 * idx: -1]]))

        return input('\n원하는 노래 제목을 입력하세요.(랜덤 원할 경우 random 입력) ')

    ##### 노래 분석 메소드 #####

    # (노래 분석 MAIN) 하이라이트를 뽑아서 리턴하는 메소드(여기서 리턴한 하이라이트 정보를 Play클래스의 PlaySong()메소드에서 재생)
    def Analysis(self):
        chroma = self.Denoising()                 # 코사인유사도 정보를 Denoising()메소드를 이용해 노이즈 제거
        print('노이즈 제거 후. 행 : {0}, 열 : {1}\n'.format(len(chroma),len(chroma[0])),chroma)
        self.Chromagram(chroma, show=self.show)   # self.show = True 일 경우 노래 시작 전 크로마그램 그래프를 출력
        return self.best_repeated_part[0] - 1.5               # 하이라이트 시간(self.result[0])-1.5초를 리턴(하이라이트 직전부터 재생)

    # (노래 분석 SUB) 코사인 유사도의 노이즈를 제거하는 메소드(Filtering()메소드와 Tensor()메소드를 이용해 노이즈 제거)
    def Denoising(self):
        chroma = self.LoadSong()                  # 코사인유사도를 chroma에 넣기
        print('코사인 유사도.  행 : {0}, 열 : {1}\n'.format(len(chroma),len(chroma[0])), chroma)
        self.Chromagram(chroma, show=self.show)
        converttime = (self.time / len(chroma))   # 샘플링된 노래 시간 정보를 실제 노래 시간 정보로 변환해주기 위한 변수
        filtered_chroma = self.Filtering(chroma)  # Filtering()메소드로 chroma 의 노이즈 제거
        filterrate = 0.25                         # Filtering()메소드의 필터링 비율(하위25% 값을 제거한다는 의미)
        while filtered_chroma.all() == 0 :        # 필터링 비율이 높아 전체 값이 제거되는 경우 필터링 비율을 낮춰줌
            filtered_chroma = self.Filtering(chroma, filterrate= filterrate-0.05)
        self.FindBestParts(filtered_chroma, converttime) # 필터링된 chroma 를 FindBestParts()메소드에 넣어 하이라이트 찾기
        return filtered_chroma                    # 필터링된 chroma를 리턴

    # (노래 분석 SUB) 가장 많이 반복된 부분을 찾아서 하이라이트로 지정하는 메소드
    def FindBestParts(self,chroma,converttime):   #
        for rn in range(len(chroma)):          # chroma에서 행과 열이 같은 부분(중복 부분)제거하여 노이즈 추가 제거
            for cn in range(rn):
                chroma[rn][cn] = 0
        chroma[chroma <= 0] = 0                # 노이즈 추가 제거 2
        repeatedcnt = self.LineFilter(chroma)  # LineFilter()메소드 사용해 시간대별 반복횟수 카운트한 리스트 생성
        best_repeated = round(converttime * max(repeatedcnt, key = lambda item: item[1])[0],1) # 최다 반복된 부분 선택
        print('\nMusic Name : {}'.format(self.music.upper()))
        print('Highlight : {}m {}s'.format(int(best_repeated//60), int(best_repeated%60)))
        self.best_repeated_part.append(best_repeated)  # best_repeated_part 에 하이라이트 부분 넣기

    ##### 노래 필터링 메소드 #####

    # (필터링1 MAIN) 노이즈 제거 시 쓰는 필터링 메소드
    def Filtering(self, chroma, cnt=3, filterrate = 0.25):
        recursive_cnt = cnt                                  # 재귀횟수
        print('재귀횟수 : {2}, 행 : {0}, 열 : {1}\n'.format(len(chroma),len(chroma[0]), cnt), chroma)
        self.Chromagram(chroma, show=self.show)  # self.show = True 일 경우 노래 시작 전 크로마그램 그래프를 출력
        chroma_copy = deepcopy(chroma)                       # 원본 데이터 변경 방지를 위한 deepcopy 백업
        chroma = np.zeros((len(chroma), (len(chroma))))      # 백업 후 chroma의 모든 값을 0으로 초기화

        # Tensor()메소드를 이용해 9x9의 작은 tensor로 chroma 내 대각선(대각선은 코사인 유사도가 높은 부분을 의미)을 강조
        for rn in range(len(chroma) - 8):
            for cn in range(len(chroma) - 8):
                chroma[rn:rn + 9, cn:cn + 9] += self.Tensor(chroma_copy[rn:rn + 9, cn:cn + 9])
        chroma[chroma <= filterrate * np.max(chroma)] = 0    # 노이즈 제거

        if cnt == 0:                                         # 마지막 재귀 시 chroma 를 정규화시켜서 데이터 정제하고 리턴
            return self.Normalization(chroma)

        print('Count down', recursive_cnt)
        return self.Filtering(chroma, cnt=recursive_cnt - 1) # 재귀를 3번 돌리며 필터링

    # (필터링1 SUB) 필터링 시 사용하는 Tensor 를 만드는 메소드
    def Tensor(self,chroma):
        tensor = np.zeros((9, 9))

        # 대각선 부분은 1을, 그 외 부분은 (-1)을 곱해서 대각선 부분의 값을 증폭시킨 후 그 평균값을 array의 중앙값으로 설정
        # 예 ) 3x3 tensor 를 사용한 원본데이터 필터링
        #     원본데이터           3x3 tensor          변환된 값
        #    1 / 0 / 0 --->     1 / -1 / -1 --->    0 /  0  / 0
        #    2 / 2 / 1 --->    -1 /  1 / -1 --->    0 / 0.8 / 0
        #    0 / 0 / 1 --->    -1 / -1 /  1 --->    0 /  0  / 0

        plus = (chroma[0][0] + chroma[1][1] + chroma[2][2] +
                chroma[3][3] + chroma[4][4] + chroma[5][5] +
                chroma[6][6] + chroma[7][7] + chroma[8][8])
        tensor[4][4] = ((10 / 9) * plus - (1 / 9) * np.sum(chroma)) / 9

        return tensor

    # (필터링2) 코사인 유사도가 높은 대각선 방향의 패턴을 카운팅하는 메소드
    def LineFilter(self,chroma, mincorrectcnt=3, line=25):
        mincnt = mincorrectcnt  # 최소 일치 횟수(대각선 패턴이 일정 횟수 이상 반복되는 경우에 리턴시키도록 제한하여 필터링 질 향상)
        shorterline = line      # 패턴 반복 확인 시 사용할 기준 대각선의 길이
        repeatedcnt = []        # 시간대별 반복횟수 저장할 리스트

        for cn in range(len(chroma)-line):
            correctcnt = 0      # 패턴 일치 횟수
            for rn in range(len(chroma)-line):
                cnt = 0
                while chroma[rn+cnt][cn+cnt] != 0 and cnt < line:
                    cnt += 1
                if cnt == line: # 패턴 일치할 경우 correctcnt 하나씩 증가
                    correctcnt += 1
            repeatedcnt.append([cn,correctcnt]) # 시간대와 시간대별 일치횟수를 리스트에 저장

        #--- 이하 리턴절은 3가지의 시나리오로 나뉘어짐 ---#

        # 3.기준 대각선의 길이가 5가 될때가지 최소 일치 횟수를 충족시키지 못한경우 재귀하여 최소 일치 횟수를 줄여서 패턴 확인
        if line <= 5 :
            return self.LineFilter(chroma, mincorrectcnt=mincnt-1, line=25)

        # 1.최소 일치 횟수 이상 패턴이 반복될 경우 정상적으로 리턴
        if max(repeatedcnt, key=lambda k:k[1])[1] >= mincnt:
            return repeatedcnt
        # 2.최소 일치 횟수 이상 패턴이 반복되지 않는 경우 기준 대각선의 길이를 5만큼 줄여서 재귀하여 패턴 확인
        return self.LineFilter(chroma, mincorrectcnt=mincnt, line=shorterline-5)

    ##### 기타 메소드 #####

    # 정규화 메소드
    def Normalization(self,chroma):
        for idx in range(len(chroma)):
            chroma[idx][idx] = 0
        return (chroma-np.mean(chroma))/np.max(chroma)

    # 크로마그램 출력 메소드
    def Chromagram(self, chroma, show=False):
        if show == True:
            plt.figure(figsize=(10, 10))
            ld.specshow(chroma)
            plt.colorbar()
            plt.title('{}'.format(self.music.upper()))
            plt.tight_layout()
            plt.show()

################## 재생 클래스 ##################
class Play(object):
    @staticmethod
    def PlaySong():
        song = Song()
        highlight = song.Analysis()

        init()
        mixer.init()
        display.set_mode((100,100))
        SetWindowPos = windll.user32.SetWindowPos
        SetWindowPos(display.get_wm_info()['window'], -1, 0, 0, 0, 0, 0x0003)
        mixer.music.load(song.music_dict[song.music])
        print('Music Start!')
        mixer.music.play(start=highlight)
        time.wait(play_duration * 1000)
        quit()

################## 메인 실행절 ##################
if __name__ == '__main__':
    Play.PlaySong()
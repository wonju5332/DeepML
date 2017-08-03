import numpy as np



class CNN:
    def __init__(self):
        self.input = None
        self.filter = None
        self.output_size = None
        self.x_pad = None
        self.stride = None

    def calCulateOutputSize(self,x,flt, pad=False, stride=1):
        self.input = x
        self.filter = flt
        self.stride = stride
        ##size 계산

        # padding 적용 안했을 때
        input_size = self.input.shape[0]
        filter_size = self.filter.shape[0]
        if pad is False:
            self.output_size = ((input_size-filter_size)/stride)+1
            return input_size,self.output_size

        # padding 적용했을 때
        else:
            p = int(self.padding())
            x_pad = np.pad(self.input, pad_width=(p,p), mode = 'constant')  # padding 적용
            self.x_pad = x_pad
            x_pad_size = x_pad.shape[0]  # 9x9로 변환
            print(x_pad.shape)
            self.output_size = ((x_pad_size-filter_size)/stride)+1
            return input_size,self.output_size

    def adJustPad(self):

        #패딩 적용했을경우 추출하기
        result = []
        f = self.filter
        f_size = f.shape[0]
        stride = self.stride
        output_size = int(self.output_size)


        for i in range(0,output_size,stride):
            for j in range(0,output_size,stride):
                temp = self.x_pad[i:(f_size+i),
                       j:(f_size+j)]
                result.append(np.sum(temp*f))
        return self.x_pad, result

    def padding(self):
        OH,H = self.input.shape  #입력과 출력의 크기는 같으므로.
        S = self.stride
        FH = self.filter.shape[0]
        return ((OH - 1) * S - H + FH) / 2



# 입력값 생성 (7,7)
x = np.arange(49)
x = x.reshape(7,7)

# 임의의 필터 생성 (3,3)
flt = np.array([[4,2,0],[3,2,6],[2,6,2]])

cnn = CNN()
cnn2 = CNN()


# 패딩 적용하지 않은 상태
just_input_size, just_output_size = cnn2.calCulateOutputSize(x,flt, pad=False)
pad_input_size, pad_output_size = cnn.calCulateOutputSize(x,flt, pad=True)

# print('패딩 적용 안했을 때',just_input_size==just_output_size)
# print('패딩 적용 후 ',pad_input_size==pad_output_size)



x_pad, result = cnn.adJustPad()
print(x)
print(x_pad)

print(pad_output_size)
p = int(pad_output_size)
print(np.array(result).reshape(p,-1))

########무성이형 문제

x = np.arange(144)
x = x.reshape(12,-1)
print(x)
f = np.eye(4,4)
print(f)
cnn3 = CNN()

input_size , output_size = cnn3.calCulateOutputSize(x,f,pad=True)

print(input_size, output_size)

x_pad, result = cnn3.adJustPad()

# print(np.array(result).reshape(12,-1).shape)










x = np.arange(16).reshape(4,4)
f = np.arange(9).reshape(3,-1)

print(x)
print(f)


x = np.arange(25).reshape(5,-1)
f = np.arange(4).reshape(2,-1)

cnn5 = CNN()

a,b = cnn5.calCulateOutputSize(x,f,pad=True)

print(a)
print(b)





np.arange(120)

data_list = []
filter_list = []
for i in range(10):
    data_list.append(np.arange(120).reshape(15,8))
    filter_list.append(np.arange(9).reshape(3,3))
print(data_list[0].shape)  #15행, 8열 확인
print(filter_list[0].shape)  #3행 3열 확인



x = np.array(data_list)

f = np.array(filter_list)


import numpy as np


class Quantization(object):
    __input_length = 336
    __syms = [' ', ',', ';', '.', '!', '?', ':', '/', '\\', '|',
              '_', '@', '#', '$', '%', '^', '&', '*', '~', '`',
              '+', '-', '=', '<', '>', '(', ')', '[', ']', '{', '}',
              '\'', '\"']

    def __init__(self):
        return

    def sent_quantize(self, sent):
        vec_list = []
        for char in sent:
            num = ord(char)
            if num >= 44032:
                vec_list.extend(self.__hangul_quantize(num))
            elif (num >= 48) and (num <= 57):
                vec_list.append(self.__num_quantize(num))
            elif self.__sym_quantize(char):
                vec_list.append(self.__sym_quantize(char))
        if len(vec_list) > self.__input_length:
            return np.array(vec_list[:self.__input_length])
        else:
            vec_list.extend([[0 for j in range(111)] for i in range(self.__input_length - len(vec_list))])
            return np.array(vec_list)

    def sent_quantize2(self, sent):
        vec_list = []
        for char in sent:
            num = ord(char)
            if num >= 44032:
                num -= 44032
                init = num // (21 * 28)
                mid = (num % (21 * 28)) // 28
                fin = num % 28
                vec_list.extend([init, mid+19])
                if fin > 0:
                    vec_list.append(fin+40)
            elif (num >= 48) and (num <= 57):
                vec_list.append(68+num-48)
            elif char in self.__syms:
                vec_list.append(self.__syms.index(char) + 78)
        if len(vec_list) > self.__input_length:
            return np.array(vec_list[:self.__input_length])
        else:
            vec_list.extend([self.__syms.index(' ') + 78 for i in range(self.__input_length - len(vec_list))])
            return np.array(vec_list)

    @staticmethod
    def __hangul_quantize(num):
        num -= 44032
        init = num // (21 * 28)
        mid = (num % (21 * 28)) // 28
        fin = num % 28
        vec_init = [0 for i in range(111)]
        vec_init[init] = 1
        vec_mid = [0 for i in range(111)]
        vec_mid[19 + mid] = 1
        if fin > 0:
            vec_fin = [0 for i in range(111)]
            vec_fin[40 + fin] = 1
            return [vec_init, vec_mid, vec_fin]
        return [vec_init, vec_mid]

    @staticmethod
    def __num_quantize(num):
        vec_num = [0 for i in range(111)]
        vec_num[68 + num - 48] = 1
        return vec_num

    def __sym_quantize(self, sym):
        if sym in self.__syms:
            vec_sym = [0 for i in range(111)]
            vec_sym[self.__syms.index(sym) + 78] = 1
            return vec_sym
        else:
            return
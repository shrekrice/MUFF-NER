input_file="../data/ctb.50d.vec"
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():#isdigit判定该char是否为数字，或者该字符串仅含数字，如果仅为数字则加’0‘
            new_word += '0'
        else:
            new_word += char
    return new_word


def build_alphabet( input_file):
    in_lines = open(input_file, 'r', encoding="utf-8").readlines()
    seqlen = 0
    f=[]
    for idx in range(len(in_lines)):  # 长度共为704368
        line = in_lines[idx]  # 读取每一行
        if len(line) > 2:
            pairs = line.strip().split()  # 首位空格去掉，并存入一个列表之中
            word = pairs[0]  # 存入lattice词典的每行首个单词
            label = pairs[-1]
            if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                biword = word + in_lines[idx + 1].strip().split()[0]
            else:
                biword = word + 'NULLKEY'
            f.append(label)
    return f
if __name__ == '__main__':

 text=build_alphabet(input_file)
 print(text)



            # if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
            #     biword = word + in_lines[idx + 1].strip().split()[0]
            # else:
            #     biword = word + NULLKEY
            # self.biword_alphabet.add(biword)
    #         # biword_index = self.biword_alphabet.get_index(biword)
    #         self.biword_count[biword] = self.biword_count.get(biword, 0) + 1
    #         for char in word:
    #             self.char_alphabet.add(char)
    #
    #         seqlen += 1
    #     else:
    #         seqlen = 0
    #
    # self.word_alphabet_size = self.word_alphabet.size()
    # self.biword_alphabet_size = self.biword_alphabet.size()
    # self.char_alphabet_size = self.char_alphabet.size()
    # self.label_alphabet_size = self.label_alphabet.size()
    # startS = False
    # startB = False
    # for label, _ in self.label_alphabet.iteritems():
    #     if "S-" in label.upper():
    #         startS = True
    #     elif "B-" in label.upper():
    #         startB = True
    # if startB:
    #     if startS:
    #         self.tagScheme = "example_BMES.dev"
    #     else:
    #         self.tagScheme = "BIO"
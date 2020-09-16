# -*- coding: utf-8 -*-

# 正则包
import re
# html 包
import html
# 自然语言处理包
import jieba
import jieba.analyse
# 机器学习包
from sklearn.metrics.pairwise import cosine_similarity
import sys







class CosineSimilarity(object):
    """
    余弦相似度
    """

    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 正则过滤 html 标签
        re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
        content = re_exp.sub(' ', content)
        # html 转义符实体化
        content = html.unescape(content)
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=200, withWeight=False)
        return keywords

    @staticmethod
    def one_hot(word_dict, keywords):  # oneHot编码
        # cut_code = [word_dict[word] for word in keywords]
        cut_code = [0]*len(word_dict)
        for word in keywords:
            cut_code[word_dict[word]] += 1
        return cut_code

    def main(self):



        # 提取关键词
        keywords1 = self.extract_keyword(self.s1)
        keywords2 = self.extract_keyword(self.s2)
        # 词的并集
        union = set(keywords1).union(set(keywords2))
        # 编码
        word_dict = {}
        i = 0
        for word in union:
            word_dict[word] = i
            i += 1
        # oneHot编码
        s1_cut_code = self.one_hot(word_dict, keywords1)
        s2_cut_code = self.one_hot(word_dict, keywords2)
        # 余弦相似度计算
        sample = [s1_cut_code, s2_cut_code]
        # 除零处理
        try:
            sim = cosine_similarity(sample)
            return sim[1][0]
        except Exception as e:
            print(e)
            return 0.0


# 测试
if __name__ == '__main__':
    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x1, open('F:/qq/sim_0.8/orig_0.8_add.txt', 'r',
                                                                           encoding="UTF-8") as y1:
        content_x1 = x1.read()
        content_y1 = y1.read()
        similarity = CosineSimilarity(content_x1, content_y1)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))
    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x2, open('F:/qq/sim_0.8/orig_0.8_del.txt', 'r',
                                                                           encoding="UTF-8") as y2:
        content_x2 = x2.read()
        content_y2 = y2.read()
        similarity = CosineSimilarity(content_x2, content_y2)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))
    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x3, open('F:/qq/sim_0.8/orig_0.8_dis_1.txt', 'r',
                                                                           encoding="UTF-8") as y3:
        content_x3 = x3.read()
        content_y3 = y3.read()
        similarity = CosineSimilarity(content_x3, content_y3)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))
    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x4, open('F:/qq/sim_0.8/orig_0.8_dis_3.txt', 'r',
                                                                           encoding="UTF-8") as y4:
        content_x4 = x4.read()
        content_y4 = y4.read()
        similarity = CosineSimilarity(content_x4, content_y4)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))

    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x6, open('F:/qq/sim_0.8/orig_0.8_dis_7.txt', 'r',
                                                                           encoding="UTF-8") as y6:
        content_x6 = x6.read()
        content_y6 = y6.read()
        similarity = CosineSimilarity(content_x6, content_y6)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))
    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x7, open('F:/qq/sim_0.8/orig_0.8_dis_10.txt', 'r',
                                                                           encoding="UTF-8") as y7:
        content_x7 = x7.read()
        content_y7 = y7.read()
        similarity = CosineSimilarity(content_x7, content_y7)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))
    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x8, open('F:/qq/sim_0.8/orig_0.8_dis_15.txt', 'r',
                                                                           encoding="UTF-8") as y8:
        content_x8 = x8.read()
        content_y8 = y8.read()
        similarity = CosineSimilarity(content_x8, content_y8)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))
    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x9, open('F:/qq/sim_0.8/orig_0.8_mix.txt', 'r',
                                                                           encoding="UTF-8") as y9:
        content_x9 = x9.read()
        content_y9 = y9.read()
        similarity = CosineSimilarity(content_x9, content_y9)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))
    with open('F:/qq/sim_0.8/orig.txt', 'r', encoding="UTF-8") as x0, open('F:/qq/sim_0.8/orig_0.8_rep.txt', 'r',
                                                                           encoding="UTF-8") as y0:
        content_x0 = x0.read()
        content_y0 = y0.read()
        similarity = CosineSimilarity(content_x0, content_y0)
        similarity = similarity.main()
        print('相似度: %.2f%%\n' % (similarity * 100))

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
    f = open(sys.argv[1], "r", encoding="UTF-8")
    content_x = f.read()
    f.close()
    g = open(sys.argv[2], "r", encoding="UTF-8")
    content_y = g.read()
    g.close()

    similarity = CosineSimilarity(content_x, content_y)
    similarity = similarity.main()
    print('相似度: %.2f%%' % (similarity*100))
    x = open(sys.argv[3], "w", encoding="UTF-8")
    x.write(str(similarity))
    x.close()






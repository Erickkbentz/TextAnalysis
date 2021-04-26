import matplotlib.pyplot as plt
import numpy as np
from pdfreader import SimplePDFViewer, PageDoesNotExist
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk import word_tokenize
from collections import Counter
import re

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def is_adjective(word, pos):
    skip = ["a", ",", ".", ":", "in", "and", ":", "PPROQUEST.COMPage", "to", "that", "``", "s", "''", "url=https",
            "[", "*", "accountid=11107", "lLnkThe", "accountid=11107Copyright", "Julian", "Document", "EnglishDocument",
            "t", "n't", "not", "now", "later", "last", "first", "many", "most", "then", "new", "also", "other", "even",
            "so", "just", "never", "more", "as", "top", "only", "least", "same", "well", "few"]
    if skip.__contains__(word):
        return False
    if pos == 'JJ' or pos == 'JJS' or pos == 'JJR' or pos == "RB" or pos == "RBS":
        return True


def get_pdf_string(path):
    fd = open(path, 'rb')
    viewer = SimplePDFViewer(fd)

    plain_text = ""
    try:
        while True:
            viewer.render()
            plain_text += "".join(viewer.canvas.strings)
            viewer.next()
    except PageDoesNotExist:
        pass
    fd.close()
    return plain_text


def split_bodies(path):
    body_pattern = "{START}(.*?){END}"

    f = open(path, "r+")
    str = f.read().replace("\n", " ")

    f.close()

    bodies = re.findall(body_pattern, str)

    return bodies


def main():
    f = open("navalFull.txt", "r+")
    navalny = f.read()

    text_navalny = word_tokenize(navalny)
    tagged_navalny = nltk.pos_tag(text_navalny)
    count_navalny = Counter(word for (word, pos) in tagged_navalny if is_adjective(word, pos))
    print(count_navalny)

    tagged_navalny.append()

    f = open("naval.txt", "w+")
    f.write("Naval\n\n")

    naval_d = {}
    for k, v in count_navalny.most_common(20):
        f.write("{} {}\n".format(k, v))
        naval_d[k] = v
    f.close()

    s = open("assangeFull.txt", "r+")
    assange = s.read()

    text_assange = word_tokenize(assange)
    tagged_assange = nltk.pos_tag(text_assange)
    count_assange = Counter(word for (word, pos) in tagged_assange if is_adjective(word, pos))
    print(count_assange)

    s = open("assange.txt", "w+")
    s.write("Assange\n\n")

    assange_d = {}
    for k, v in count_assange.most_common(20):
        s.write("{} {}\n".format(k, v))
        assange_d[k] = v
    s.close()


def descriptors_chart(tagged, title, path):
    a_keys = tagged.keys()
    y_pos = np.arange(len(a_keys))
    # get the counts for each key, assuming the values are numerical
    performance = [tagged[k] for k in a_keys]
    # not sure if you want this :S
    plt.barh(y_pos, performance, align='center')
    plt.yticks(y_pos, a_keys)
    plt.subplots_adjust(left=0.2)
    plt.xlabel('Counts per key')
    plt.title(title)
    plt.savefig(path, format="png")
    plt.show()


def get_sentiment_analysis(paragraph_list):
    sentiment_list = []
    str_list = split_into_sentences(paragraph_list[0])
    print(str_list)
    sia = SentimentIntensityAnalyzer()

    for string in str_list:
        sentiment = sia.polarity_scores(string)
        sentiment_list.append(sentiment)

    return sentiment_list


if __name__ == '__main__':
    bodies = split_bodies("articles/AssangeParagraphs.txt")

    sent_list = get_sentiment_analysis(bodies)
    print(sent_list)

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
    skip = ["now", "later", "last", "first", "many", "most", "then", "new", "also", "other", "even",
            "so", "just", "never", "more", "as", "top", "only", "least", "same", "well", "few", "such", "own"]
    if skip.__contains__(word):
        return False
    if pos == 'JJ' or pos == 'JJS' or pos == 'JJR':
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


def descriptors_chart(counter, title, path):
    a_keys = counter.keys()
    y_pos = np.arange(len(a_keys))
    # get the counts for each key, assuming the values are numerical
    performance = [counter[k] for k in a_keys]
    # not sure if you want this :S
    plt.barh(y_pos, performance, align='center')
    plt.yticks(y_pos, a_keys)
    plt.subplots_adjust(left=0.2)
    plt.xlabel('Count per word')
    plt.title(title)
    plt.savefig(path, format="png")
    plt.show()


def get_sentiment_analysis_for_paragraph(paragraph):
    sentiment_list = []
    str_list = split_into_sentences(paragraph)
    print(str_list)
    sia = SentimentIntensityAnalyzer()

    for string in str_list:
        sentiment = sia.polarity_scores(string)
        sentiment_list.append(sentiment)

    return sentiment_list


def main():
    navalny_article_list = split_bodies("articles/NavalnyParagraphs.txt")
    count_navalny = Counter()

    for article in navalny_article_list:
        token_words = word_tokenize(article)
        navalny_tagged = nltk.pos_tag(token_words)
        count_navalny.update(word for (word, pos) in navalny_tagged if is_adjective(word, pos))

    navalny_d = {}
    for k, v in count_navalny.most_common(25):
        navalny_d[k] = v

    descriptors_chart(navalny_d, "Navalny - 25 most used adjectives in articles", "navalAdjWords.png")

    assange_article_list = split_bodies("articles/AssangeParagraphs.txt")
    count_assange = Counter()

    for article in assange_article_list:
        token_words = word_tokenize(article)
        assange_tagged = nltk.pos_tag(token_words)
        count_assange.update(word for (word, pos) in assange_tagged if is_adjective(word, pos))

    assange_d = {}
    for k, v in count_assange.most_common(25):
        assange_d[k] = v

    descriptors_chart(assange_d, "Assange - 25 most used adjectives in articles", "assangeAdjWords.png")


if __name__ == '__main__':
    main()

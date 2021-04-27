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


def adjectives_chart(counter, title, path):
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


def get_sentiment_analysis_for_article(paragraph):
    sentiment_list = []
    str_list = split_into_sentences(paragraph)
    sia = SentimentIntensityAnalyzer()

    for string in str_list:
        sentiment = sia.polarity_scores(string)
        sentiment_list.append(sentiment)

    return sentiment_list


def main():
    # ------------------------------------------------NAVALNY----------------------------------------------------------
    navalny_article_list = split_bodies("articles/navalny/NavalnyParagraphs.txt")
    count_navalny = Counter()

    navalny_article_word_count = []
    navalny_entire_sentiment_list = []

    for article in navalny_article_list:
        # append sentence sentiment list
        navalny_entire_sentiment_list.append(get_sentiment_analysis_for_article(article))
        # get word count
        token_words = word_tokenize(article)
        navalny_article_word_count.append(len(token_words))
        # get pos of words and find adjectives
        navalny_tagged = nltk.pos_tag(token_words)
        count_navalny.update(word for (word, pos) in navalny_tagged if is_adjective(word, pos))

    print("Navalny word count per article: {}".format(navalny_article_word_count))

    navalny_d = {}
    for k, v in count_navalny.most_common(25):
        navalny_d[k] = v

    adjectives_chart(navalny_d, "Navalny - 25 most used adjectives in articles", "navalAdjWords.png")

    # Table calculations and creation
    sum = 0
    for count in navalny_article_word_count:
        sum += count

    navalny_avg_word_count = (sum // len(navalny_article_word_count))
    print("Navalny avg word count: {}".format(navalny_avg_word_count))

    # ------------------------------------------------ASSANGE----------------------------------------------------------
    assange_article_list = split_bodies("articles/assange/AssangeParagraphs.txt")
    count_assange = Counter()

    assange_article_word_count = []
    assange_entire_sentiment_list = []

    for article in assange_article_list:
        # append sentence sentiment list
        navalny_entire_sentiment_list.append(get_sentiment_analysis_for_article(article))
        # append word count
        token_words = word_tokenize(article)
        assange_article_word_count.append(len(token_words))
        # get pos of words and find adjectives
        assange_tagged = nltk.pos_tag(token_words)
        count_assange.update(word for (word, pos) in assange_tagged if is_adjective(word, pos))

    print("Assange word count per article: {}".format(assange_article_word_count))
    assange_d = {}
    for k, v in count_assange.most_common(25):
        assange_d[k] = v

    adjectives_chart(assange_d, "Assange - 25 most used adjectives in articles", "assangeAdjWords.png")

    # Table calculations and creation
    sum = 0
    for count in assange_article_word_count:
        sum += count

    assange_avg_word_count = (sum // len(assange_article_word_count))
    print("Assange avg word count: {}".format(assange_avg_word_count))


if __name__ == '__main__':
    main()

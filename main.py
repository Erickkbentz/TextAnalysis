from datetime import datetime
from statistics import mean
from time import strptime, strftime

import PIL
import dateutil
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from PIL.Image import Image
from pdfreader import SimplePDFViewer, PageDoesNotExist
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk import word_tokenize
from collections import Counter
import re
from dateutil.parser import parse

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
    skip = ["Julian"]
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


def get_doc_type_count(path):
    f = open(path, "r+")
    list_type = f.read().split("\n")
    f.close()
    return Counter(list_type)


def get_page_list(path):
    f = open(path, "r+")

    page_list = []
    for line in f:
        line = line.strip()
        if line != "n/a":
            page_list.append(int(line))
    f.close()
    return page_list


def get_avg_title_sent(path):
    f = open(path, "r+")

    count = 0
    neg = 0.0
    pos = 0.0
    neu = 0.0
    com = 0.0
    sia = SentimentIntensityAnalyzer()
    for line in f:
        count += 1
        sent = sia.polarity_scores(line)
        neg += sent['neg']
        pos += sent['pos']
        neu += sent['neu']
        com += sent['compound']

    f.close()
    avg_title_sent = {'neg': (neg / count), 'pos': (pos / count), 'neu': (neu / count), 'compound': (com / count)}
    return avg_title_sent


def get_publication_date_list_month(path):
    f = open(path, "r+")
    date_list = []
    for line in f:
        dt = parse(line)
        date_list.append(dt)

    return date_list


def get_avg_articles_per_month(path):
    date_list = get_publication_date_list_month(path)
    num_months = get_month_range(date_list)

    return len(date_list) / num_months


def get_month_range(date_list):
    dt_start = date_list[-1]
    dt_end = date_list[0]
    num_months = (dt_end.year - dt_start.year) * 12 + (dt_end.month - dt_start.month)
    return num_months


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

    navalny_d = {}
    for k, v in count_navalny.most_common(25):
        navalny_d[k] = v

    adjectives_chart(navalny_d, "Navalny - 25 most used adjectives in articles", "navalAdjWords.png")

    # -----------------------------------------Calculations for Table---------------------------------------------------
    print("\n----------------------------------Navalny--------------------------------------")
    print("25 Most common Adj: {}\n".format(navalny_d))
    # Avg word Count
    sum = 0
    for count in navalny_article_word_count:
        sum += count

    navalny_avg_word_count = (sum // len(navalny_article_word_count))
    print("Navalny avg word count: {}\n".format(navalny_avg_word_count))

    # Avg sentiment intensity per article
    naval_total_neg = 0.0
    naval_total_neu = 0.0
    naval_total_pos = 0.0
    naval_total_com = 0.0
    naval_sent_count = 0

    naval_article_sentiments = []
    for article_sentiment in navalny_entire_sentiment_list:
        article_sent_count = 0
        article_neg = 0.0
        article_neu = 0.0
        article_pos = 0.0
        article_com = 0.0
        for sentence_sentiment in article_sentiment:
            article_sent_count += 1
            naval_sent_count += 1
            naval_total_neg += sentence_sentiment['neg']
            article_neg += sentence_sentiment['neg']

            naval_total_pos += sentence_sentiment['pos']
            article_pos += sentence_sentiment['pos']

            naval_total_neu += sentence_sentiment['neu']
            article_neu += sentence_sentiment['neu']

            naval_total_com += sentence_sentiment['compound']
            article_com += sentence_sentiment['compound']

        naval_article_sentiments.append({'neg': (article_neg / article_sent_count),
                                         'neu': (article_neu / article_sent_count),
                                         'pos': (article_pos / article_sent_count),
                                         'compound': (article_com / article_sent_count)})

    count = len(naval_article_sentiments)
    neg = 0.0
    neu = 0.0
    pos = 0.0
    com = 0.0

    for article_sentiment_avg in naval_article_sentiments:
        neg += article_sentiment_avg['neg']
        neu += article_sentiment_avg['neu']
        pos += article_sentiment_avg['pos']
        com += article_sentiment_avg['compound']
    """
    print("AVERAGE SENTIMENT INTENSITY OF EACH ARTICLE")
    print("Average [neg]: {}".format((neg / count)))
    print("Average [neu]: {}".format((neu / count)))
    print("Average [pos]: {}".format((pos / count)))
    print("Average [com]: {}".format((com / count)))
    """
    print("\nAVERAGE SENTIMENT INTENSITY OF EACH SENTENCE")
    print("Average [neg]: {}".format((naval_total_neg / naval_sent_count)))
    print("Average [neu]: {}".format((naval_total_neu / naval_sent_count)))
    print("Average [pos]: {}".format((naval_total_pos / naval_sent_count)))
    print("Average [com]: {}".format((naval_total_com / naval_sent_count)))

    naval_avg_sentence_sent = {'neg': naval_total_neg / naval_sent_count,
                               'neu': naval_total_neu / naval_sent_count,
                               'pos': naval_total_pos / naval_sent_count,
                               'compound': naval_total_com / naval_sent_count}

    avg_title_sent_n = get_avg_title_sent("articles/navalny/NavalnyTitles.txt")
    print("\nAVERAGE TITLE SENTIMENT:\nAverage [neg]: {}\nAverage [neu]: {}\nAverage [pos]: {}\nAverage [com]: {}\n"
          .format(avg_title_sent_n['neg'], avg_title_sent_n['neu'], avg_title_sent_n['pos'],
                  avg_title_sent_n['compound']))

    # Number of doc Types
    doc_types = get_doc_type_count("articles/navalny/NavalnyDocumentType.txt")
    print(doc_types)

    # page location
    page_list_n = get_page_list("articles/navalny/NavalnyPages.txt")
    page_count_n = Counter(page_list_n)
    print("\nNavalny Times Appearing on Certain Page (most Common 10):\n{}".format(page_count_n.most_common(10)))
    avg_page_n = mean(page_list_n)
    print("\nNavalny avg page location: {}".format(avg_page_n))

    # ------------------------------------------------ASSANGE----------------------------------------------------------
    assange_article_list = split_bodies("articles/assange/AssangeParagraphs.txt")
    count_assange = Counter()

    assange_article_word_count = []
    assange_entire_sentiment_list = []

    for article in assange_article_list:
        # append sentence sentiment list
        assange_entire_sentiment_list.append(get_sentiment_analysis_for_article(article))
        # append word count
        token_words = word_tokenize(article)
        assange_article_word_count.append(len(token_words))
        # get pos of words and find adjectives
        assange_tagged = nltk.pos_tag(token_words)
        count_assange.update(word for (word, pos) in assange_tagged if is_adjective(word, pos))

    assange_d = {}
    for k, v in count_assange.most_common(25):
        assange_d[k] = v

    adjectives_chart(assange_d, "Assange - 25 most used adjectives in articles", "assangeAdjWords.png")

    # -----------------------------------------Calculations for Table---------------------------------------------------
    print("\n----------------------------------ASSANGE--------------------------------------")
    print("25 Most common Adj: {}\n".format(assange_d))

    sum = 0
    for count in assange_article_word_count:
        sum += count

    assange_avg_word_count = (sum // len(assange_article_word_count))
    print("Assange avg word count: {}\n".format(assange_avg_word_count))

    # Avg sentiment intensity per article
    assange_total_neg = 0.0
    assange_total_neu = 0.0
    assange_total_pos = 0.0
    assange_total_com = 0.0
    assange_sent_count = 0

    assange_article_sentiments = []
    for article_sentiment in assange_entire_sentiment_list:
        article_sent_count = 0
        article_neg = 0.0
        article_neu = 0.0
        article_pos = 0.0
        article_com = 0.0
        for sentence_sentiment in article_sentiment:
            article_sent_count += 1
            assange_sent_count += 1
            assange_total_neg += sentence_sentiment['neg']
            article_neg += sentence_sentiment['neg']

            assange_total_pos += sentence_sentiment['pos']
            article_pos += sentence_sentiment['pos']

            assange_total_neu += sentence_sentiment['neu']
            article_neu += sentence_sentiment['neu']

            assange_total_com += sentence_sentiment['compound']
            article_com += sentence_sentiment['compound']

        assange_article_sentiments.append({'neg': (article_neg / article_sent_count),
                                           'neu': (article_neu / article_sent_count),
                                           'pos': (article_pos / article_sent_count),
                                           'compound': (article_com / article_sent_count)})

    count = len(assange_article_sentiments)
    neg = 0.0
    neu = 0.0
    pos = 0.0
    com = 0.0
    for article_sentiment_avg in assange_article_sentiments:
        neg += article_sentiment_avg['neg']
        neu += article_sentiment_avg['neu']
        pos += article_sentiment_avg['pos']
        com += article_sentiment_avg['compound']

    """
    print("AVERAGE SENTIMENT INTENSITY OF EACH ARTICLE")
    print("Average [neg]: {}".format((neg / count)))
    print("Average [neu]: {}".format((neu / count)))
    print("Average [pos]: {}".format((pos / count)))
    print("Average [com]: {}".format((com / count)))
    """

    print("\nAVERAGE SENTIMENT INTENSITY")
    print("Average [neg]: {}".format((assange_total_neg / assange_sent_count)))
    print("Average [neu]: {}".format((assange_total_neu / assange_sent_count)))
    print("Average [pos]: {}".format((assange_total_pos / assange_sent_count)))
    print("Average [com]: {}".format((assange_total_com / assange_sent_count)))

    assange_avg_sentence_sent = {'neg': assange_total_neg / assange_sent_count,
                                 'neu': assange_total_neu / assange_sent_count,
                                 'pos': assange_total_pos / assange_sent_count,
                                 'compound': assange_total_com / assange_sent_count}

    avg_title_sent_a = get_avg_title_sent("articles/assange/AssangeTitles.txt")
    print("\nAVERAGE TITLE SENTIMENT:\nAverage [neg]: {}\nAverage [neu]: {}\nAverage [pos]: {}\nAverage [com]: {}\n"
          .format(avg_title_sent_a['neg'], avg_title_sent_a['neu'], avg_title_sent_a['pos'],
                  avg_title_sent_a['compound']))

    # Number of doc Types
    doc_types_a = get_doc_type_count("articles/assange/AssangeDocumentType.txt")
    print(doc_types_a)

    # page location
    page_list_a = get_page_list("articles/assange/AssangePages.txt")
    page_count_a = Counter(page_list_a)
    print("\nAssange Times Appearing on Certain Page (most Common 10):\n{}".format(page_count_a.most_common(10)))

    avg_page_a = mean(page_list_a)
    print("\nAssange avg page location: {}".format(avg_page_a))

    n_avg_per_month = get_avg_articles_per_month("articles/navalny/NavalnyPublicationDate.txt")
    n_date_list = get_publication_date_list_month("articles/navalny/NavalnyPublicationDate.txt")
    n_month_range = get_month_range(n_date_list)

    a_avg_per_month = get_avg_articles_per_month("articles/assange/AssangePublicationDate.txt")
    a_date_list = get_publication_date_list_month("articles/assange/AssangePublicationDate.txt")
    a_month_range = get_month_range(a_date_list)

    print()
    print("NAVALNY- Range: {}, Amount: {}, Avg: {:0.2f}".format(n_month_range, len(n_date_list), n_avg_per_month))
    print("ASSANGE- Range: {}, Amount: {}, Avg: {:0.2f}".format(a_month_range, len(n_date_list), a_avg_per_month))

    # -----------------------------------------------TABLE-------------------------------------------------------------
    fig = go.Figure(data=[go.Table(
        header=dict(values=['News Event', '# of Articles', 'Avg. Articles per Month', 'Publication Date Range in Months',
                            'Avg. Word Count', '# of Section Front Pages', '% Front Pages', 'Avg. Section Page'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[["Navalny", "Assange"], [len(navalny_article_list), len(assange_article_list)],
                           ["{:0.2f}".format(n_avg_per_month), "{:0.2f}".format(a_avg_per_month)],
                           [n_month_range, a_month_range],
                           [navalny_avg_word_count, assange_avg_word_count],
                           [page_count_n[1], page_count_a[1]],
                           ["{:0.2f}%".format((page_count_n[1] / len(navalny_article_list)) * 100),
                            "{:0.2f}%".format((page_count_a[1] / len(assange_article_list)) * 100)],
                           ["{:0.2f}".format(avg_page_n), "{:0.2f}".format(avg_page_a)]],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.write_image("Assange_vs_Navalny.png")
    fig.show()

    fig2 = go.Figure(data=[go.Table(
        header=dict(values=[],
                    fill_color='black',
                    align='left'),
        cells=dict(values=[["Navalny", "", "", "", "",
                            "Assange", "", "", "", ""],
                           ["Average Title Sentiment Intensity",
                            "Negative: {:0.2f}%".format(avg_title_sent_n['neg'] * 100),
                            "Neutral: {:0.2f}%".format(avg_title_sent_n['neu'] * 100),
                            "Positive: {:0.2f}%".format(avg_title_sent_n['pos'] * 100),
                            "Compound: {:0.2f}%".format(avg_title_sent_n['compound'] * 100),
                            "",
                            "Negative: {:0.2f}%".format(avg_title_sent_a['neg'] * 100),
                            "Neutral: {:0.2f}%".format(avg_title_sent_a['neu'] * 100),
                            "Positive: {:0.2f}%".format(avg_title_sent_a['pos'] * 100),
                            "Compound: {:0.2f}%".format(avg_title_sent_a['compound'] * 100)],
                           ["Average Sentence Sentiment Intensity",
                            "Negative: {:0.2f}%".format(naval_avg_sentence_sent['neg'] * 100),
                            "Neutral: {:0.2f}%".format(naval_avg_sentence_sent['neu'] * 100),
                            "Positive: {:0.2f}%".format(naval_avg_sentence_sent['pos'] * 100),
                            "Compound: {:0.2f}%".format(naval_avg_sentence_sent['compound'] * 100),
                            "",
                            "Negative: {:0.2f}% ".format(assange_avg_sentence_sent['neg'] * 100),
                            "Neutral: {:0.2f}%".format(assange_avg_sentence_sent['neu'] * 100),
                            "Positive: {:0.2f}%".format(assange_avg_sentence_sent['pos'] * 100),
                            "Compound: {:0.2f}%".format(assange_avg_sentence_sent['compound'] * 100)]
                           ],
                   fill_color=[['paleturquoise', 'lavender', 'lavender', 'lavender', 'lavender',
                                'paleturquoise', 'lavender', 'lavender', 'lavender', 'lavender'] * 3],
                   align='center'))
    ])
    fig2.write_image("AssangeSent_vs_NavalnySent.png")
    fig2.show()

    im1 = PIL.Image.open('navalAdjWords.png')
    im2 = PIL.Image.open('assangeAdjWords.png')

    dst = PIL.Image.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.save('Naval&AssangeAdj.png')


if __name__ == '__main__':
    main()

from gensim.parsing import remove_stopwords
from nltk import sent_tokenize
import gensim.downloader as api
import re


def get_tokens(sentence):
    sentence = [char.lower() if char.isalpha() or char == "’" or char == "'" else " " for char in sentence]
    return remove_stopwords("".join(sentence)).split()


if __name__ == '__main__':
    chars_to_remove = re.compile('[“”"]')


    def get_lines(file, encoding="utf-8"):
        with open(file, encoding=encoding) as f:
            text = f.read().strip()

        filtered_text = chars_to_remove.sub("", text)
        lines = [line for line in filtered_text.splitlines() if line]
        return lines

    def get_sentence_tokens(file, encoding="utf-8"):
        lines = get_lines(file, encoding)
        sentences = [sentence for line in lines for sentence in sent_tokenize(line)]
        sentence_tokens = [get_tokens(sentence) for sentence in sentences]
        return list(zip(sentence_tokens, sentences, range(len(sentences))))


    en_sentence_tokens = get_sentence_tokens("content/test/en.txt")
    tr_sentence_tokens = get_sentence_tokens("content/test/translated.txt")

    print(en_sentence_tokens)
    print(tr_sentence_tokens)
    num_en_sentences = len(en_sentence_tokens)
    num_tr_sentences = len(tr_sentence_tokens)

    model = api.load('word2vec-google-news-300')
    for tr_tokens, tr_sentence, tr_index in tr_sentence_tokens:
        distance_list = []
        approximate_en_index = (tr_index * num_en_sentences) // num_tr_sentences
        window_size = (num_en_sentences * 5) // 100
        # print(f"Windows size = {window_size}")
        # print(f"Approximate en_index = {approximate_en_index}")
        # print(f"TR: {tr_index}: {tr_sentence}")
        # print(en_sentence_tokens[approximate_en_index - window_size:approximate_en_index + window_size])
        window_start = approximate_en_index - window_size
        window_end = approximate_en_index + window_size
        if window_start < 0:
            window_end -= window_start
            window_start = 0
        if window_end > num_en_sentences:
            window_start -= window_end - num_en_sentences
            window_end = num_en_sentences
        for en_tokens, en_sentence, en_index in en_sentence_tokens[window_start:window_end]:
            # print(f"EN: {en_index}: {en_sentence}")
            distance = model.wmdistance(tr_tokens, en_tokens)
            if distance != float("inf"):
                distance_list.append((distance, en_sentence))
        distance_list = sorted(distance_list)
        print(f"{tr_sentence}")
        for distance, sentence in distance_list[:5]:
            print(f"[{distance:.4f}]: {sentence}")
        if len(distance_list) == 0:
            print("[No results :(]")
        print()

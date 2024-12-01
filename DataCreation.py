import random
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api

model = api.load('word2vec-google-news-300')
lemmatizer = WordNetLemmatizer()

def normalize_word(word):
    word = word.lower() 
    word = word.replace("_", "") 
    return lemmatizer.lemmatize(word)  

def get_wordnet_relations(word):
    normalized_word = normalize_word(word)
    synonyms, antonyms, hypernyms, hyponyms, meronyms, holonyms, entailments = set(), set(), set(), set(), set(), set(), set()

    for synset in wn.synsets(normalized_word):
        # Synonyms
        synonyms.update(normalize_word(lemma.name()) for lemma in synset.lemmas() if "_" not in lemma.name() and normalize_word(lemma.name()) in model.key_to_index)
        # Antonyms
        antonyms.update(
            normalize_word(antonym.name())
            for lemma in synset.lemmas()
            for antonym in lemma.antonyms()
            if "_" not in antonym.name() and normalize_word(antonym.name()) in model.key_to_index
        )
        # Hypernyms
        hypernyms.update(
            normalize_word(related_word.name().split(".")[0])
            for related_word in synset.hypernyms()
            if "_" not in related_word.name().split(".")[0] and normalize_word(related_word.name().split(".")[0]) in model.key_to_index
        )
        # Hyponyms
        hyponyms.update(
            normalize_word(related_word.name().split(".")[0])
            for related_word in synset.hyponyms()
            if "_" not in related_word.name().split(".")[0] and normalize_word(related_word.name().split(".")[0]) in model.key_to_index
        )
        # Meronyms
        meronyms.update(
            normalize_word(related_word.name().split(".")[0])
            for related_word in synset.part_meronyms() + synset.substance_meronyms()
            if "_" not in related_word.name().split(".")[0] and normalize_word(related_word.name().split(".")[0]) in model.key_to_index
        )
        # Holonyms
        holonyms.update(
            normalize_word(related_word.name().split(".")[0])
            for related_word in synset.member_holonyms() + synset.substance_holonyms()
            if "_" not in related_word.name().split(".")[0] and normalize_word(related_word.name().split(".")[0]) in model.key_to_index
        )
        # Entailments
        entailments.update(
            normalize_word(related_word.name().split(".")[0])
            for related_word in synset.entailments()
            if "_" not in related_word.name().split(".")[0] and normalize_word(related_word.name().split(".")[0]) in model.key_to_index
        )
    return synonyms, antonyms, hypernyms, hyponyms, meronyms, holonyms, entailments

def generate_pairs():
    direct_similar, indirect_similar, not_similar = set(), set(), set()
    vocab = list(model.key_to_index.keys())
    random.shuffle(vocab)

    for word in vocab:
        if not wn.synsets(word) or "_" in word:
            continue

        synonyms, antonyms, hypernyms, hyponyms, meronyms, holonyms, entailments = get_wordnet_relations(word)

        for related_word in synonyms | antonyms | hypernyms | hyponyms:
            if word != related_word:
                direct_similar.add((normalize_word(word), related_word, "direct", "synonym" if related_word in synonyms else
                                    "antonym" if related_word in antonyms else
                                    "hypernym" if related_word in hypernyms else "hyponym"))

        for related_word in meronyms | holonyms | entailments:
            if word != related_word:
                indirect_similar.add((normalize_word(word), related_word, "indirect", "meronym" if related_word in meronyms else
                                      "holonym" if related_word in holonyms else "entailment"))

        not_related_word = normalize_word(random.choice(vocab))
        if word != not_related_word and "_" not in not_related_word:
            not_similar.add((normalize_word(word), not_related_word, "not-related", ""))
            
        if len(direct_similar) >= 20000 and len(indirect_similar) >= 20000 and len(not_similar) >= 20000:
            break

    return list(direct_similar)[:20000], list(indirect_similar)[:20000], list(not_similar)[:20000]

direct, indirect, not_related = generate_pairs()

data = direct + indirect + not_related
df = pd.DataFrame(data, columns=["word1", "word2", "similarity_type", "relation"])
df.to_csv("word_pairs.csv", index=False)
print("Word pairs saved to 'word_pairs.csv'.")
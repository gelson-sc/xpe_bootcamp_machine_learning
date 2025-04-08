import spacy
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import wordnet
import stanza
from textblob import TextBlob  # muito utilizado para análise de sentimento
from deep_translator import GoogleTranslator

# import nltk.data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# tokenizacao_portugues = nltk.data.load('tokenizers/punkt/PY3/portuguese.pickle')
# print(tokenizacao_portugues.tokenize('Olá, como você está?'))
# texto_grande_sertao = "Sou só um sertanejo, nessas altas ideias navego mal. Sou muito pobre coitado. Inveja minha pura é de uns conforme o senhor, com toda leitura e suma doutoração."
# lista_sentencas = tokenizacao_portugues.tokenize(texto_grande_sertao)
# print(lista_sentencas)
# sentenca_1 = word_tokenize(lista_sentencas[0])
# print('sentenca', sentenca_1)
#
# tokenizacao = RegexpTokenizer("[\w']+")
# print(tokenizacao.tokenize(lista_sentencas[0]))
# # sinonimos -> funciona no inglês
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# palavra = wordnet.synsets('drive')
# print(palavra)
# print(palavra[0].name())
# print(palavra[0].definition())
# print(palavra[0].examples())
# # print(palavra[0].lemmas())
# # print(palavra[0].lemma_names())
# # print(palavra[0].hypernyms())
# # print(palavra[0].hyponyms())
# # print(palavra[0].part_holonyms())
# # print(palavra[0].part_meronyms())
# # print(palavra[0].member_holonyms())
# # print(palavra[0].member_meronyms())
# # print(palavra[0].substance_holonyms())
# # print(palavra[0].substance_meronyms())
# # print(palavra[0].entailments())
#
# #realizando o processo de stemização
# nltk.download('rslp') #A Stemming Algorithm for the Portuguese Language"
# stemmer_portugues = nltk.stem.RSLPStemmer()
# print(stemmer_portugues.stem('motor'))
# print(stemmer_portugues.stem('copiar'))
# print(stemmer_portugues.stem('casarão'))
# print(stemmer_portugues.stem('copiando'))
# print(stemmer_portugues.stem('casarões'))
#
# stanza.download('pt')
# nlp = stanza.Pipeline('pt')
# doc = nlp('Sou um sertanejo, nessas altas ideias navego mal. Sou muito pobre coitado. Inveja minha pura é de uns conforme o senhor, com toda leitura e suma doutoração.')
# print(doc.sentences[0].text)
#
# palavra = ""
# lemma = ""
# for sent in nlp("minha vida é uma alegria").sentences:
#     for word in sent.words:
#         palavra += word.text + "\t"
#         lemma += word.lemma + "\t"
#
# print(palavra)
# print(lemma)

texto_1 = 'Acabei de comprar um produto e estou gostando bastante'
texto_2 = 'Acabei de comprar um produto e não estou gostando'
analise_1 = TextBlob(texto_1)
analise_2 = TextBlob(texto_2)
print(analise_1.sentiment)
print(analise_2.sentiment)

traducao = GoogleTranslator(source='auto', target='en').translate("keep it up, you are awesome")
print(traducao)

analise_1 = TextBlob(traducao)
print(analise_1.sentiment)

#!python -m spacy download pt
#!python -m spacy download pt_core_news_sm

nlp = spacy.load("pt_core_news_sm")
print(nlp)
#identificação de entidades
texto = nlp(u'Você gostou do livro que eu te falei, Carla?')
list_data = [(token.orth_, token.pos_) for token in texto]
print(list_data)
#!pip install gensim
#!pip install --upgrade --force-reinstall numpy gensim


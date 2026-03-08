import re
import emoji
# import spacy.cli # A DECOMMENTER lors de la première exécution si utilisation d'anaconda
# spacy.cli.download("fr_core_news_sm") # A DECOMMENTER lors de la première exécution si utilisation d'anaconda
import spacy
from langdetect import detect
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from spellchecker import SpellChecker
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon', quiet=True) # A DECOMMENTER lors de la première exécution
vader_analyzer = SentimentIntensityAnalyzer()

nlp_fr = spacy.load("fr_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

spell_fr = SpellChecker(language='fr')
spell_en = SpellChecker(language='en')

def extraire_metriques_style(texte_brut):
    if not texte_brut or len(texte_brut.strip()) == 0:
        return {}

    metriques = {}

    # presence de hashtags
    metriques['presence_hashtags'] = len(re.findall(r'#\w+', texte_brut)) > 0
    # presence de mentions
    metriques['presence_mentions'] = len(re.findall(r'\[MENTION\]', texte_brut)) > 0
    # presence de urls
    metriques['presence_urls'] = len(re.findall(r'\[URL\]', texte_brut)) > 0
    # presence de nombres
    metriques['presence_nombres'] = len(re.findall(r'\[NB\]', texte_brut)) > 0
    texte_brut = re.sub(r'\[(MENTION|URL|NB)\]', '', texte_brut)

    liste_hashtags = re.findall(r'#\w+', texte_brut)
    
    nb_lettres = sum(1 for c in texte_brut if c.isalpha())
    nb_majuscules = sum(1 for c in texte_brut if c.isupper())
    
    metriques['ratio_majuscules'] = nb_majuscules / nb_lettres if nb_lettres > 0 else 0
    
    nb_exclamations = texte_brut.count('!')
    nb_interrogations = texte_brut.count('?')
    metriques['densite_ponct_agressive'] = (nb_exclamations + nb_interrogations) / len(texte_brut)

    # 'doc' contient maintenant le texte découpé intelligemment avec la grammaire !
    try:
        langue = detect(texte_brut)
    except:
        langue = 'en'
    if langue == 'fr':
        doc = nlp_fr(texte_brut)
        langue_emoji = 'fr'
    else:
        doc = nlp_en(texte_brut)
        langue_emoji = 'en'
    
    # On filtre pour ne garder que les vrais mots (on exclut la ponctuation et les espaces)
    vrais_mots = [token for token in doc if not token.is_punct and not token.is_space]
    nb_mots = len(vrais_mots)
    
    if nb_mots == 0:
        return metriques

    mots_uniques = set([token.text.lower() for token in vrais_mots])
    metriques['richesse_lexicale'] = len(mots_uniques) / nb_mots
    
    longueur_totale_mots = sum(len(token.text) for token in vrais_mots)
    metriques['longueur_moyenne_mots'] = longueur_totale_mots / nb_mots
    
    # spaCy détecte automatiquement les phrases (sentences)
    nb_phrases = len(list(doc.sents))
    metriques['longueur_moyenne_phrases'] = nb_mots / nb_phrases if nb_phrases > 0 else 0

    # ADJ = Adjectif, PRON = Pronom, ADV = Adverbe, VERB = Verbe, HASHTAG = Hashtag
    nb_adjectifs = sum(1 for token in vrais_mots if token.pos_ == "ADJ")
    nb_pronoms = sum(1 for token in vrais_mots if token.pos_ == "PRON")
    nb_adverbes = sum(1 for token in vrais_mots if token.pos_ == "ADV")
    nb_verbes = sum(1 for token in vrais_mots if token.pos_ == "VERB")
    nb_emojis = nb_emojis = emoji.emoji_count(texte_brut)
    nb_hashtags = len(liste_hashtags)  

    metriques['ratio_adjectifs'] = nb_adjectifs / nb_mots
    metriques['ratio_pronoms'] = nb_pronoms / nb_mots
    metriques['ratio_adverbes'] = nb_adverbes / nb_mots
    metriques['ratio_verbes'] = nb_verbes / nb_mots
    metriques['nb_hashtags'] = nb_hashtags
    metriques['nb_emojis'] = nb_emojis

    mots_a_verifier = [token.text.lower() for token in vrais_mots if token.pos_ != "PROPN"]
    
    if langue == 'fr':
        fautes = spell_fr.unknown(mots_a_verifier)
    else:
        fautes = spell_en.unknown(mots_a_verifier)
        
    metriques['ratio_fautes_orthographe'] = len(fautes) / nb_mots if nb_mots > 0 else 0

    texte_emojis_mots = emoji.demojize(texte_brut, language=langue_emoji)
    texte_emojis_mots = re.sub(r':([^\s:]+):', r'\1', texte_emojis_mots)
    texte_emojis_mots = texte_emojis_mots.replace('_', ' ')
    # print(texte_emojis_mots)
    
    if langue == 'fr':
        blob = TextBlob(texte_emojis_mots, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        polarite_finale = blob.sentiment[0]
        subjectivite_finale = blob.sentiment[1]
    else:
        blob = TextBlob(texte_emojis_mots)
        polarite_finale = vader_analyzer.polarity_scores(texte_emojis_mots)['compound']
        subjectivite_finale = blob.sentiment.subjectivity

    # polarite : de -1 (très négatif) à +1 (très positif)
    # subjectivite : de 0 (très objectif/factuel) à 1 (très subjectif/opinion)
    metriques['polarite_sentiment'] = polarite_finale
    metriques['subjectivite'] = subjectivite_finale

    return metriques


def normaliser_texte(texte):
    # Remplacer les URLs par la balise [URL]
    texte_propre = re.sub(r'http\S+|www.\S+', '[URL]', texte)
    
    # Remplacer les mentions Twitter/X par [MENTION]
    texte_propre = re.sub(r'@\w+', '[MENTION]', texte_propre)
    
    # Remplacer les nombres par [NB]
    texte_propre = re.sub(r'\b\d+(?:[., :h]\d+)*\b', '[NB]', texte_propre)    
    texte_propre = re.sub(r'\b\d+\s*(?:min|lbs|kg|km|ml|ft|in|oz|°C|°F|°K|h|s|m|g|l|°|%|€|\$|£|¥)?(?!\w)', '[NB]', texte_propre)

    return texte_propre


def print_resultats(resuls_parse, texte_brut):
    texte_normalise, vecteur_style = resuls_parse
    print(f"Texte brut :      \n{texte_brut}\n")
    print(f"Texte normalisé : \n{texte_normalise}\n")
    print(f"Vecteur style : ")
    for cle, valeur in vecteur_style.items():
        print(f"{cle} : {round(valeur, 3)}")


def traitement_news(texte_brut):
    texte_normalise = normaliser_texte(texte_brut)
    vecteur_style = extraire_metriques_style(texte_normalise)

    return texte_normalise, vecteur_style


# texte_test = "ALERTE !!! Vous devez absolument lire ceci : https://www.google.com. Le gouvernement nous cache 10000 choses terribles et monstrueuses ! Réveillez-vous immédiatement @user ! #news 😡"
# texte_test = "ALERT! You must absolutely read this: https://www.google.com. The government is hiding 10,000 terrible and monstrous things! Wake up immediately @user! #news 😡"
# print_resultats(parse_news(texte_test), texte_test)
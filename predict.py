import joblib
from language_model import DICT_FEATURES

clf = joblib.load('classificador.pkl')

DICT_LANGUAGES = {
    0: 'Francês',
    1: 'Inglês',
    2: 'Italiano',
}

def language(texto):
    features_texto = []
    for _, func in DICT_FEATURES.items():
        features_texto.append(func(texto))
    
    return DICT_LANGUAGES[clf.predict([features_texto])[0]]


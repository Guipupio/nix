import joblib
import pandas as pd
import re

from nix.settings import BASE_DIR

VOGAIS = [u'ë', u'û', u'À', u'È', u'O', u'Ô', u'à', u'è', u'ì', u'o', u'ô', u'ü', u'É', u'é', u'ù', u'A', u'Â', u'E', u'Æ', u'I', u'Ê', u'U', u'Y', u'a', u'â', u'e', u'æ', u'i', u'ê', u'î', u'ò', u'u', u'ö', u'y', u'ï']
CONSOANTES = [u'D', u'p', u't', u'x', u'Ç', u'H', u'L', u'P', u'T', u'X', u'd', u'ç', u'h', u'l', u'C', u'G', u'K', u'S', u'W', u'c', u'g', u'k', u's', u'w', u'B', u'F', u'J', u'N', u'R', u'V', u'Z', u'b', u'f', u'j', u'n', u'ñ', u'r', u'v', u'z', u'M', u'Q', u'm', u'q']
ACENTOS = [u'ë', u'û', u'À', u'È', u'O', u'Ô', u'à', u'è', u'ì',  u'ô', u'ü', u'É', u'é', u'ù', u'Â', u'Ê', u'â', u'ê', u'î', u'ò', u'ö', u'ï']

DICT_FEATURES = {}

map_language = {
    'french': 0,
    'english': 1,
    'italian': 2,
}

def feature(func):
    return DICT_FEATURES.setdefault(func.__name__, func)

@feature
def get_vogal_relativa(texto):
    num = len(list(filter(lambda char: char in VOGAIS, texto)))
    return num / len(texto)

@feature
def get_consoante_relativa(texto):
    num = len(list(filter(lambda char: char in CONSOANTES, texto)))
    return num / len(texto)

@feature
def get_acento_relativo(texto):
    num = len(list(filter(lambda char: char in ACENTOS, texto)))
    return num / len(texto)

@feature
def get_palavras_final_vogal(texto):
    list_caracteres_ignorados = [',', '.', '"']
    list_palavras = extrai_lista_palavras(texto, list_caracteres_ignorados)
        
    num = len(list(filter(lambda palavra: palavra[-1] in VOGAIS, list_palavras)))
    return num / len(list_palavras)

@feature
def get_ocorrencia_letras_no_texto(texto):
    # expressao regular para o 'OR' das letras
    letras_buscadas = '[wky]'
    return len(re.findall(letras_buscadas, texto)) / len(texto)

@feature
def get_palavras_com_apostrofe(texto):
    list_palavras = extrai_lista_palavras(texto)

    num = len(list(filter(lambda palavra: "'" in palavra, list_palavras)))
    return num / len(list_palavras)


def extrai_lista_palavras(texto: str, list_caracteres_ignorados: list = []) -> list:
    
    for char in list_caracteres_ignorados:
        texto = texto.replace(char, '')
    
    return list(filter(lambda x: x != '', texto.split(' ')))

def extrai_features_df(df: pd.DataFrame) -> pd.DataFrame:
    
    def aplica_features(df):
        for func_name, func in DICT_FEATURES.items():
            df[func_name[4:]] = func(df.text)
        return df
    df_features = df.apply(aplica_features, axis=1)
    
    df_features['y'] = df_features.language.apply(lambda x: map_language[x]) 

    return df_features
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_validate, train_test_split

if __name__ == "__main__":
    LABELS = ['Francês', 'Inglês', 'Italiano']
    INPUT = [func_name[4:] for func_name, _ in DICT_FEATURES.items()]

    # obtem DF
    df = pd.read_csv('/'.join([BASE_DIR, 'dataset', 'language', 'csv', 'language_dataset.csv']))
    

    # Extrai features
    df_features = extrai_features_df(df)
    x_train, x_test, y_train, y_test = train_test_split(df_features[INPUT], df_features.y, test_size=.33)

    clf = RandomForestClassifier()

    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)

    print(classification_report(y_test, y_predict, target_names=LABELS))
    print(confusion_matrix(y_test, y_predict))
    print('A acurácia do modelo é: ', accuracy_score(y_test, y_predict))

    joblib.dump(clf, 'classificador.pkl')

    # cv = cross_validate(clf, df_features[INPUT], df_features.y, cv=3)
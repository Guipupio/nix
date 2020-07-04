import pandas as pd

from nix.settings import BASE_DIR

VOGAIS = [u'ë', u'û', u'À', u'È', u'O', u'Ô', u'à', u'è', u'ì', u'o', u'ô', u'ü', u'É', u'é', u'ù', u'A', u'Â', u'E', u'Æ', u'I', u'Ê', u'U', u'Y', u'a', u'â', u'e', u'æ', u'i', u'ê', u'î', u'ò', u'u', u'ö', u'y', u'ï']
CONSOANTES = [u'D', u'p', u't', u'x', u'Ç', u'H', u'L', u'P', u'T', u'X', u'd', u'ç', u'h', u'l', u'C', u'G', u'K', u'S', u'W', u'c', u'g', u'k', u's', u'w', u'B', u'F', u'J', u'N', u'R', u'V', u'Z', u'b', u'f', u'j', u'n', u'ñ', u'r', u'v', u'z', u'M', u'Q', u'm', u'q']

DICT_FEATURES = {}

map_language = {
    'french': 0,
    'english': 1,
    'italian': 2
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


def extrai_features_df(df: pd.DataFrame) -> pd.DataFrame:
    
    def aplica_features(df):
        for func_name, func in DICT_FEATURES.items():
            df[func_name[4:]] = func(df.text)
        return df
    df_features = df.apply(aplica_features, axis=1)
    
    df_features['y'] = df_features.language.apply(lambda x: map_language[x]) 

    return df_features
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

if __name__ == "__main__":
    # obtem DF
    df = pd.read_csv('/'.join([BASE_DIR, 'dataset', 'language', 'csv', 'language_dataset.csv']))

    # Extrai features
    df_features = extrai_features_df(df)

    clf = RandomForestClassifier()

    cv = cross_validate(clf, df_features[['vogal_relativa', 'consoante_relativa']], df_features.y, cv=3)
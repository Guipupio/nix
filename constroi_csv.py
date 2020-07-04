from nix.settings import BASE_DIR
import pandas as pd

with open('/'.join([BASE_DIR, 'dataset', 'language', 'raw', 'set_data.txt'])) as _file:
    # Extraimos o dataSet
    dataset = _file.readlines()


dataset = list(map(lambda txt: txt.replace("\n", '').split('@'), dataset))
df = pd.DataFrame(dataset, columns=['text', 'language'])

df.to_csv('/'.join([BASE_DIR, 'dataset', 'language', 'csv', 'language_dataset.csv']), index=False)
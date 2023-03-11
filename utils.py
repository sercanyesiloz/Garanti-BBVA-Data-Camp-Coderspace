import re
import itertools
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def my_tokenizer(text):
    return re.split("\\s+",text)

def translation(x):
    return x.translate(x.maketrans("ÜüÖöİıĞğŞşÇç", "UuOoIiGgSsCc"))

def plot_confusion_matrix(cm,
                          classes,
                          title,
                          save:bool=False):

    plt.imshow(cm, interpolation='nearest', cmap='Greens')
    plt.title(title, size = 12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.0f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    if save:
        plt.savefig(f'plots/{title}.jpg')
    plt.show()

def load_tr_cities() -> list:

    return ['ISTANBUL', 'BALIKESIR', 'BURSA', 'TEKIRDAG', 'CANAKKALE', 'YALOVA', 'KOCAELI', 'KIRKLARELI', 'EDIRNE', 'BILECIK', 'SAKARYA', 'IZMIR', 'MANISA', 'AYDIN', 'DENIZLI', 'USAK', 'AFYONKARAHISAR', 'KUTAHYA', 'MUGLA', 'ANTALYA', 'ADANA', 'MERSIN', 'HATAY', 'BURDUR', 'OSMANIYE', 'KAHRAMANMARAS', 'ISPARTA', 'ANKARA', 'KONYA', 'KAYSERI', 'ESKISEHIR', 'SIVAS', 'KIRIKKALE', 'AKSARAY', 'KARAMAN', 'KIRSEHIR', 'NIGDE', 'NEVSEHIR', 'YOZGAT', 'CANKIRI', 'AMASYA', 'ARTVIN', 'BARTIN', 'BAYBURT', 'BOLU', 'CORUM', 'DUZCE', 'GUMUSHANE', 'GIRESUN', 'KARABUK', 'KASTAMONU', 'ORDU', 'RIZE', 'SAMSUN', 'SINOP', 'TOKAT', 'TRABZON', 'ZONGULDAK', 'AGRI', 'ARDAHAN', 'BITLIS', 'BINGOL', 'ELAZIG', 'ERZINCAN', 'ERZURUM', 'HAKKARI', 'IGDIR', 'KARS', 'MALATYA', 'MUS', 'TUNCELI', 'VAN', 'GAZIANTEP', 'DIYARBAKIR', 'SANLIURFA', 'BATMAN', 'ADIYAMAN', 'SIIRT', 'MARDIN', 'KILIS', 'SIRNAK']

def desc_stats(dataframe: pd.DataFrame, title=None) -> None:

    desc = dataframe.describe().T
    f, ax = plt.subplots(figsize=(14, desc.shape[0] * 0.65))
    sns.heatmap(
        desc,
        annot=True,
        cmap='Blues',
        fmt=".2f",
        ax=ax,
        linecolor="white",
        linewidths=1.25,
        cbar=False,
        annot_kws={"size": 9},
    )
    plt.xticks(size=10)
    plt.yticks(size=10, rotation=0)
    plt.title("Descriptive Statistics" if title == None else title, size=12)
    plt.show()

def corr_map(dataframe: pd.DataFrame, method="pearson", title=None) -> None:
    assert method in ["pearson", "spearman"], "Invalid Correlation Method"
    matrix = np.triu(dataframe.corr(method=method))
    f, ax = plt.subplots(figsize=(matrix.shape[0] * 0.55, matrix.shape[1] * 0.55))
    sns.heatmap(
        dataframe.corr(method=method),
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax,
        vmin=-1,
        vmax=1,
        mask=matrix,
        cmap="coolwarm",
        linewidth=0.4,
        linecolor="white",
        annot_kws={"size": 10},
    )
    plt.xticks(rotation=75, size=12)
    plt.yticks(rotation=0, size=12)
    plt.title(f"{method.title()} Correlation Map" if title == None else title, size=14)
    plt.show()

def plot_importances(model, features):
    sns.set(rc={"axes.facecolor": "gainsboro", "figure.facecolor": "gainsboro"})
    importances = model.feature_importances_
    indices = np.argsort(importances)
    indices = indices[-50:]
    plt.figure(figsize=(20, 10))
    plt.title("Feature Importances", size=10)
    plt.barh(
        range(len(indices)), importances[indices], color="royalblue", align="center"
    )
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Relative Importance", size=10)
    plt.show()
    matplotlib.rc_file_defaults()
    sns.reset_orig()


def check_missing(dataframe: pd.DataFrame) -> pd.DataFrame:

    return pd.DataFrame(
        {
            "feature": dataframe.columns,
            "n_missing": [dataframe[i].isnull().sum() for i in dataframe.columns],
            "missing_ratio": [dataframe[i].isnull().sum() / dataframe.shape[0] for i in dataframe.columns],
        }
    ).reset_index(drop=True).sort_values("n_missing", ascending=False)


def plot_missing(dataframe: pd.DataFrame, title=None):
    sns.set(rc={"axes.facecolor": "gainsboro", "figure.facecolor": "gainsboro"})
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=check_missing(dataframe),
        x="missing_ratio",
        y="feature",
        palette='Oranges',
    )
    plt.title("Missing Values" if title == None else title, size=12)
    plt.ylabel("Features", size=10)
    plt.xlabel("Missing Ratio", size=10)
    plt.show()
    matplotlib.rc_file_defaults()
    sns.reset_orig()

population = {
    "İstanbul": 15907951,
    "Ankara": 5782285,
    "İzmir": 4462056,
    "Bursa": 3194720,
    "Antalya": 2688004,
    "Konya": 2296347,
    "Adana": 2274106,
    "Şanlıurfa": 2170110,
    "Gaziantep": 2154051,
    "Kocaeli": 2079072,
    "Mersin": 1916432,
    "Diyarbakır": 1804880,
    "Hatay": 1686043,
    "Manisa": 1468279,
    "Kayseri": 1441523,
    "Samsun": 1368488,
    "Balıkesir": 1257590,
    "Kahramanmaraş": 1177436,
    "Van": 1128749,
    "Aydın": 1148241,
    "Tekirdağ": 1142451,
    "Sakarya": 1080080,
    "Denizli": 1056332,
    "Muğla": 1049185,
    "Eskişehir": 906617,
    "Mardin": 870374,
    "Trabzon": 818023,
    "Malatya": 812580,
    "Ordu": 763190,
    "Erzurum": 749754,
    "Afyonkarahisar": 747555,
    "Sivas": 634924,
    "Adıyaman": 635169,
    "Batman": 634491,
    "Tokat": 596454,
    "Elazığ": 591497,
    "Zonguldak": 588510,
    "Kütahya": 580701,
    "Osmaniye": 559405,
    "Çanakkale": 559383,
    "Şırnak": 557605,
    "Çorum": 524130,
    "Ağrı": 510626,
    "Giresun": 450862,
    "Isparta": 445325,
    "Aksaray": 433055,
    "Yozgat": 418442,
    "Edirne": 414714,
    "Düzce": 405131,
    "Muş": 399202,
    "Kastamonu": 378115,
    "Uşak": 375454,
    "Kırklareli": 369347,
    "Niğde": 365419,
    "Bitlis": 353988,
    "Rize": 344016,
    "Amasya": 338267,
    "Siirt": 331311,
    "Bolu": 320824,
    "Nevşehir": 310011,
    "Yalova": 296333,
    "Bingöl": 283112,
    "Kars": 282556,
    "Kırıkkale": 277046,
    "Hakkari": 275333,
    "Burdur": 273716,
    "Karaman": 258838,
    "Karabük": 249287,
    "Kırşehir": 242944,
    "Erzincan": 237351,
    "Bilecik": 228334,
    "Sinop": 218408,
    "Iğdır": 203159,
    "Bartın": 201711,
    "Çankırı": 196515,
    "Artvin": 169543,
    "Gümüşhane": 150119,
    "Kilis": 145826,
    "Ardahan": 94932,
    "Bayburt": 85042,
    "Tunceli": 83645,
}

def add_populations(dataframe: pd.DataFrame) -> pd.DataFrame:

    df_ = dataframe.copy()
    new_pops = pd.DataFrame()
    for k in population.keys():
        new_pops = new_pops.append(pd.DataFrame({'location': [translation(k.upper())], 'population': [population[k]]}))

    df_ = df_.merge(new_pops, on = ['location'], how = 'left')
    return df_

employment = {
'ADANA': 37816,
'ADIYAMAN': 6604,
'AFYONKARAHİSAR': 9935,
'AĞRI': 2415,
'AKSARAY': 7191,
'AMASYA': 4227,
'ANKARA': 98211,
'ANTALYA': 48241,
'ARDAHAN': 549,
'ARTVİN': 2462,
'AYDIN': 28689,
'BALIKESİR': 22292,
'BARTIN': 3885,
'BATMAN': 6125,
'BAYBURT': 877,
'BİLECİK': 5813,
'BİNGÖL': 3090,
'BİTLİS': 2458,
'BOLU': 7240,
'BURDUR': 3825,
'BURSA': 57833,
'ÇANAKKALE': 8463,
'ÇANKIRI': 2845,
'ÇORUM': 6743,
'DENİZLİ': 18420,
'DİYARBAKIR': 21139,
'DÜZCE': 5278,
'EDİRNE': 5062,
'ELAZIĞ': 9375,
'ERZİNCAN': 5015,
'ERZURUM': 8658,
'ESKİŞEHİR': 18636,
'GAZİANTEP': 41251,
'GİRESUN': 5384,
'GÜMÜŞHANE': 1449,
'HAKKARİ': 691,
'HATAY': 10498,
'IĞDIR': 2006,
'ISPARTA': 10257,
'İSTANBUL': 353908,
'İZMİR': 128150,
'KAHRAMANMARAŞ': 18440,
'KARABÜK': 6170,
'KARAMAN': 4034,
'KARS': 1810,
'KASTAMONU': 4070,
'KAYSERİ': 24198,
'KIRIKKALE': 6367,
'KIRKLARELİ': 6720,
'KIRŞEHİR': 3513,
'KİLİS': 959,
'KOCAELİ': 65767,
'KONYA': 21086,
'KÜTAHYA': 9288,
'MALATYA': 14569,
'MANİSA': 19876,
'MARDİN': 5518,
'MERSİN': 30139,
'MUĞLA': 21096,
'MUŞ': 4929,
'NEVŞEHİR': 3605,
'NİĞDE': 5306,
'ORDU': 9064,
'OSMANİYE': 7325,
'RİZE': 5260,
'SAKARYA': 27785,
'SAMSUN': 29292,
'SİİRT': 2576,
'SİNOP': 3327,
'SİVAS': 8074,
'ŞANLIURFA': 9040,
'ŞIRNAK': 5021,
'TEKİRDAĞ': 25505,
'TOKAT': 8159,
'TRABZON': 15368,
'TUNCELİ': 1666,
'UŞAK': 11117,
'VAN': 4961,
'YALOVA': 5445,
'YOZGAT': 5405,
'ZONGULDAK': 5420}

def add_employment(dataframe: pd.DataFrame) -> pd.DataFrame:

    df_ = dataframe.copy()
    new_emp = pd.DataFrame()
    for k in employment.keys():
        new_emp = new_emp.append(pd.DataFrame({'location': [translation(k.upper())], 'iskur_employment_2019': [employment[k]]}))

    df_ = df_.merge(new_emp, on = ['location'], how = 'left')
    return df_


def label_encode(
    le_cols: list, train_data: pd.DataFrame, test_data: pd.DataFrame = pd.DataFrame(), fillna: bool = False
):
    train_ = train_data.copy()
    test_ = test_data.copy()
    if test_.shape[0] == 0:
        for col in le_cols:
            encoder = LabelEncoder()
            train_[col] = encoder.fit_transform(train_[col])

        return train_
    else:
        for col in le_cols:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan
            )
            train_[col] = encoder.fit_transform(train_[col].values.reshape(-1, 1))
            test_[col] = encoder.transform(test_[col].values.reshape(-1, 1))
            if fillna and test_[col].isnull().sum() != 0:
                max_ = max(
                    train_[col].dropna().astype(int).max(),
                    test_[col].dropna().astype(int).max(),
                )
                test_[col] = test_[col].fillna(max_ + 1)
                train_[col] = train_[col].fillna(max_ + 1)

        return train_, test_
        

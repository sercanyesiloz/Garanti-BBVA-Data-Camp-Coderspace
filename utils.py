import re
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
            "missing_rate": [dataframe[i].isnull().sum() / dataframe.shape[0] for i in dataframe.columns],
        }
    ).reset_index(drop=True).sort_values("n_missing", ascending=False)


def plot_missing(dataframe: pd.DataFrame, title=None):
    sns.set(rc={"axes.facecolor": "gainsboro", "figure.facecolor": "gainsboro"})
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=check_missing(dataframe),
        x="missing_rate",
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
        

def fix_skills(dataframe: pd.DataFrame) -> pd.DataFrame:

    df_ = dataframe.copy()
    df_['skill'] = df_['skill'].apply(lambda x: x.strip())
    df_.loc[df_['skill'] == 'Web Geliştirme', 'skill'] = 'Web Development'
    df_.loc[df_['skill'] == 'web development', 'skill'] = 'Web Development'
    df_.loc[df_['skill'] == 'Git Hub', 'skill'] = 'GitHub'
    df_.loc[df_['skill'] == 'Github Actions', 'skill'] = 'GitHub'
    df_.loc[df_['skill'] == 'Github', 'skill'] = 'GitHub'
    df_.loc[df_['skill'] == 'Githup', 'skill'] = 'GitHub'
    df_.loc[df_['skill'] == 'Web Uygulamaları', 'skill'] = 'Web Applications'
    df_.loc[df_['skill'] == 'Proje Yönetimi', 'skill'] = 'Project Management'
    df_.loc[df_['skill'] == 'Programlama', 'skill'] = 'Programming'
    df_.loc[df_['skill'] == 'Object-Oriented Programming (OOP)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Object Oriented Programming (OOP)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP (Nesne Yönelimli Programlama)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP(Object Oriented Programming)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP (Object Oriented Programming)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Objektorientierte Programmierung (OOP)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Object-Oriented Programming(OOP)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP(Object Orianted Programming)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP (Nesne Yönelimli Programlama', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP programming and implementing design patterns', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP(Object-Oriented Programming)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Nesne Yönelimli Programlama(OOP)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Object Oriented Programming(OOP)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Programación orientada a objetos (OOP)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP ( Object Oriented Programming )', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP (Object-Oriented Programming)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP ( Object - Oriented Programming )', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP(Nesne Yönelimli Programlama)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Object-oriented Programming (OOP)', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'OOP Design', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Advanced OOP', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Python (Programming Language)', 'skill'] = 'Python'
    df_.loc[df_['skill'] == 'Python (Programmiersprache)', 'skill'] = 'Python'
    df_.loc[df_['skill'] == 'Python Programming Language', 'skill'] = 'Python'
    df_.loc[df_['skill'] == 'Ptyhon', 'skill'] = 'Python'
    df_.loc[df_['skill'] == 'Phyton', 'skill'] = 'Python'
    df_.loc[df_['skill'] == 'data science', 'skill'] = 'Data Science'
    df_.loc[df_['skill'] == 'Data science', 'skill'] = 'Data Science'
    df_.loc[df_['skill'] == 'Microsoft Teknolojileri', 'skill'] = 'Microsoft Technologies'
    df_.loc[df_['skill'] == 'Microsoft Sunucular', 'skill'] = 'Microsoft Servers'
    df_.loc[df_['skill'] == 'microsoft', 'skill'] = 'Microsoft'
    df_.loc[df_['skill'] == 'MsSQL database', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MsSQL Server', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MsSQL', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Amazon Web Services (AWS)', 'skill'] = 'AWS'
    df_.loc[df_['skill'] == 'Amazon Web Hizmetleri (AWS)', 'skill'] = 'AWS'
    df_.loc[df_['skill'] == 'Amazon Web Services', 'skill'] = 'AWS'
    df_.loc[df_['skill'] == 'machine learning', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Machine learning', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Makine Öğrenmesi/Machine Learning', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Artificial Intelligence (AI)', 'skill'] = 'Artificial Intelligence'
    df_.loc[df_['skill'] == 'Artificial intelligence', 'skill'] = 'Artificial Intelligence'
    df_.loc[df_['skill'] == 'Yazılım Proje Yönetimi', 'skill'] = 'Software Project Management'
    df_.loc[df_['skill'] == 'Proje Planlama', 'skill'] = 'Project Planning'
    df_.loc[df_['skill'] == 'İngilizce', 'skill'] = 'English'
    df_.loc[df_['skill'] == 'english', 'skill'] = 'English'
    df_.loc[df_['skill'] == 'git', 'skill'] = 'Git'
    df_.loc[df_['skill'] == 'github', 'skill'] = 'GitHub'
    df_.loc[df_['skill'] == 'Yazılım Geliştirme', 'skill'] = 'Software Development'
    df_.loc[df_['skill'] == 'Yazılım Mühendisliği', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Yazılım Mühendisleri', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Yazılım Tasarımı', 'skill'] = 'Software Design'
    df_.loc[df_['skill'] == 'Yazılım', 'skill'] = 'Software'
    df_.loc[df_['skill'] == 'yazılım', 'skill'] = 'Software'
    df_.loc[df_['skill'] == 'Açık Kaynak Yazılımı', 'skill'] = 'Open Source Software'
    df_.loc[df_['skill'] == 'Yazılım Geliştirme Metodolojileri', 'skill'] = 'Software Development Methodologies'
    df_.loc[df_['skill'] == 'Yazılım Kalitesi', 'skill'] = 'Software Quality'
    df_.loc[df_['skill'] == 'Yazılım Konfigürasyon Yönetimi', 'skill'] = 'Software Configuration Management'
    df_.loc[df_['skill'] == 'Yazılım Çözümleri', 'skill'] = 'Software Solutions'
    df_.loc[df_['skill'] == 'Yazılım Dokümantasyonu', 'skill'] = 'Software Documentation'
    df_.loc[df_['skill'] == 'Yazılım Kalite Güvencesi', 'skill'] = 'Software Quality Assurance'
    df_.loc[df_['skill'] == 'matlab', 'skill'] = 'Matlab'
    df_.loc[df_['skill'] == 'MATLAB', 'skill'] = 'Matlab'
    df_.loc[df_['skill'] == 'MATLAB®', 'skill'] = 'Matlab'
    df_.loc[df_['skill'] == 'Php', 'skill'] = 'PHP'   
    df_.loc[df_['skill'] == 'java', 'skill'] = 'Java' 
    df_.loc[df_['skill'] == 'java dili', 'skill'] = 'Java' 
    df_.loc[df_['skill'] == 'javascript', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'java script', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'html', 'skill'] = 'HTML'
    df_.loc[df_['skill'] == 'My Sql', 'skill'] = 'MySQL'
    df_.loc[df_['skill'] == 'English B2', 'skill'] = 'English'
    df_.loc[df_['skill'] == 'English C1', 'skill'] = 'English'
    df_.loc[df_['skill'] == 'jquery', 'skill'] = 'jQuery' 
    df_.loc[df_['skill'] == 'Mssql', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'mssql', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'proje yönetimi', 'skill'] = 'Project Management'
    df_.loc[df_['skill'] == 'project management', 'skill'] = 'Project Management'
    df_.loc[df_['skill'] == 'Proje Yönetim', 'skill'] = 'Project Management'
    df_.loc[df_['skill'] == 'Microsoft sql', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'docker', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'docker & kubernetes', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'docker-compose', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'docker in docker', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'docker swarm', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker Containerization (Python-Java APP docker containerization, Private-Cloud Docker Registry)', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker Products', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Containerization (Docker)', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker (Compose - Stac - Swarm)', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker Container', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Dockerize', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker Swarm', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker Compose', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker-compose', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker & Docker Compose', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Docker Containerization', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'linux', 'skill'] = 'Linux'
    df_.loc[df_['skill'] == 'asp.net', 'skill'] = 'ASP.NET'
    df_.loc[df_['skill'] == 'asp.net core', 'skill'] = 'ASP.NET'
    df_.loc[df_['skill'] == 'asp. net core', 'skill'] = 'ASP.NET'
    df_.loc[df_['skill'] == 'asp.net webapi', 'skill'] = 'ASP.NET'
    df_.loc[df_['skill'] == 'asp core', 'skill'] = 'ASP.NET'
    df_.loc[df_['skill'] == 'sql', 'skill'] = 'SQL'
    df_.loc[df_['skill'] == 'Mysql', 'skill'] = 'MySQL'
    df_.loc[df_['skill'] == 'mysql', 'skill'] = 'MySQL'
    df_.loc[df_['skill'] == 'ms sql', 'skill'] = 'MSSQL'
    df_.loc[df_['skill'] == 'Agile Metotları', 'skill'] = 'Agile Methodologies'
    df_.loc[df_['skill'] == 'Agile Methodolgy', 'skill'] = 'Agile Methodologies'
    df_.loc[df_['skill'] == 'agile methodologies', 'skill'] = 'Agile Methodologies'
    df_.loc[df_['skill'] == 'agile methodology', 'skill'] = 'Agile Methodologies'
    df_.loc[df_['skill'] == 'agile methodoligies', 'skill'] = 'Agile Methodologies'
    df_.loc[df_['skill'] == 'AGILE METHODOLOGIES', 'skill'] = 'Agile Methodologies'
    df_.loc[df_['skill'] == 'Agile Proje Yönetimi', 'skill'] = 'Agile Project Management'
    df_.loc[df_['skill'] == 'MYSQL', 'skill'] = 'MySQL'
    df_.loc[df_['skill'] == 'SQL MYSQL', 'skill'] = 'MySQL'
    df_.loc[df_['skill'] == 'mysqli', 'skill'] = 'MySQL'
    df_.loc[df_['skill'] == 'JQUERY', 'skill'] = 'jQuery'
    df_.loc[df_['skill'] == 'Microsoft SQL', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Microsoft Sql Server', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Microsoft Sql', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Microsoft SQL Server 2016 and T-SQL', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'microsoft sql server', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'React.js', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'ReactJS', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'React JS', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'Reactjs', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'ReactJs', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'React.Js', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'React js', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'React Js', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'Html', 'skill'] = 'HTML'
    df_.loc[df_['skill'] == 'scrum', 'skill'] = 'Scrum'
    df_.loc[df_['skill'] == 'programlama', 'skill'] = 'Programming'
    df_.loc[df_['skill'] == 'excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'Mühendislik', 'skill'] = 'Engineering'
    df_.loc[df_['skill'] == 'mühendislik', 'skill'] = 'Engineering'
    df_.loc[df_['skill'] == 'yazılım mühendisliği', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'bilgisayar mühendisliği', 'skill'] = 'Computer Engineering'
    df_.loc[df_['skill'] == 'Yazılım mühendisliği', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Software Engineering Practices', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Software Engineers', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Software Engineer', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Software Enginering', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Software Engine', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Object Oriented Software Engineering', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Embeded Software Engineering', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'software engineering', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'software engineer', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'Microcontroller 68hc811 hardware and software engineer', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'x86 based microcontroller hardware and software engineer', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'model based software engineering', 'skill'] = 'Software Engineering'
    df_.loc[df_['skill'] == 'software development', 'skill'] = 'Software Development'
    df_.loc[df_['skill'] == 'software developer', 'skill'] = 'Software Development'
    df_.loc[df_['skill'] == 'Javascript', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'Javscript', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'oop', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'ms office', 'skill'] = 'Microsoft Office'
    df_.loc[df_['skill'] == 'css', 'skill'] = 'CSS'
    df_.loc[df_['skill'] == 'Css', 'skill'] = 'CSS'
    df_.loc[df_['skill'] == 'advance in english', 'skill'] = 'English'
    df_.loc[df_['skill'] == 'ASP.Net', 'skill'] = 'ASP.NET'
    df_.loc[df_['skill'] == 'ORACLE', 'skill'] = 'Oracle'
    df_.loc[df_['skill'] == 'php', 'skill'] = 'PHP'
    df_.loc[df_['skill'] == 'jira', 'skill'] = 'JIRA'
    df_.loc[df_['skill'] == 'Jira', 'skill'] = 'JIRA'
    df_.loc[df_['skill'] == 'Atlassian Jira', 'skill'] = 'JIRA'
    df_.loc[df_['skill'] == '.net', 'skill'] = '.NET'
    df_.loc[df_['skill'] == '.Net', 'skill'] = '.NET'
    df_.loc[df_['skill'] == 'Asp.Net', 'skill'] = 'ASP.NET'
    df_.loc[df_['skill'] == 'object oriented programming', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'json', 'skill'] = 'JSON'
    df_.loc[df_['skill'] == 'html5', 'skill'] = 'HTML5'
    df_.loc[df_['skill'] == 'html 5', 'skill'] = 'HTML5'
    df_.loc[df_['skill'] == 'android', 'skill'] = 'Android'
    df_.loc[df_['skill'] == 'Swift (Programming Language)', 'skill'] = 'Swift'
    df_.loc[df_['skill'] == 'Go (Programming Language)', 'skill'] = 'Go'
    df_.loc[df_['skill'] == 'C (Programming Language)', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'R (Programming Language)', 'skill'] = 'R'
    df_.loc[df_['skill'] == 'c', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'c#', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'c++', 'skill'] = 'C++'
    df_.loc[df_['skill'] == 'C programming', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C programlama', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C Programlama', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'R Programlama', 'skill'] = 'R'
    df_.loc[df_['skill'] == 'R programming', 'skill'] = 'R'
    df_.loc[df_['skill'] == 'R programlama', 'skill'] = 'R'
    df_.loc[df_['skill'] == 'c programlama', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C# programlama', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'C programlama dili', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C++ ile programlama', 'skill'] = 'C++'
    df_.loc[df_['skill'] == 'C# ile programlama', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'c sharp', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'C #', 'skill'] = 'C#'
    df_.loc[df_['skill'] == '•C #', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'c #', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'WinForms application developer using C #', 'skill'] = 'C# Programming'
    df_.loc[df_['skill'] == 'Object oriented programming', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Java programming language', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'SAS programming', 'skill'] = 'SAS'
    df_.loc[df_['skill'] == 'computer programming', 'skill'] = 'Computer Programming'
    df_.loc[df_['skill'] == 'programming', 'skill'] = 'Programming'
    df_.loc[df_['skill'] == 'Nesne yönelimli programlama', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Swift programming language', 'skill'] = 'Swift'
    df_.loc[df_['skill'] == 'Software programming Fundamentals', 'skill'] = 'Software Programming'
    df_.loc[df_['skill'] == 'Introduction to JAVA programming', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'bilgisayar programlama', 'skill'] = 'Computer Programming' 
    df_.loc[df_['skill'] == 'Bilgisayar programlama', 'skill'] = 'Computer Programming'
    df_.loc[df_['skill'] == 'nesneye yönelik programlama', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Nesne tabanlı programlama', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Nesneye yönelik programlama', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'SAS Programlama', 'skill'] = 'SAS'
    df_.loc[df_['skill'] == 'microsoft excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'excel eğitmeni', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'excel makro', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'macro excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'ileri excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'Web geliştirme', 'skill'] = 'Web Development'
    df_.loc[df_['skill'] == 'SCRUM', 'skill'] = 'Scrum'
    df_.loc[df_['skill'] == 'scrump', 'skill'] = 'Scrum'
    df_.loc[df_['skill'] == 'reactjs', 'skill'] = 'React.js'
    df_.loc[df_['skill'] == 'DOCKER', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'DOCKERS', 'skill'] = 'Docker'
    df_.loc[df_['skill'] == 'Postgresql', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'Postgre SQL', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'Postgres', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'PostgreSql', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'Postgre', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'PostgreSQL 9.6', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'PostgreSQL = 5/10', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'PostgreSQL Server', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'Postgre sql', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'Postgresql Administration', 'skill'] = 'PostgreSQL'
    df_.loc[df_['skill'] == 'JAVA', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'JAVASCRIPT', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'JAVASCRİPT', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'MSSQL', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MSSQL Server', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MSSQL SERVER', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MSSQL Server 2012', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MSSQL DATABASE MANAGEMENT', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MSSQL DB', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MSSQL Server and SQL Coding', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MSSQL SERVER 2000/2005/2008', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Microsoft mssql', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Mssql Server', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Ms sql', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Ms Sql', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Ms Sql server', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Ms Sql Server Managemant Studio', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Ms Sql 2000', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Ms Sql 2008', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Ms SQL', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Ms SQL Server', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'Nesne Tabanlı Programlama', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'Oop', 'skill'] = 'Object Oriented Programming'
    df_.loc[df_['skill'] == 'GİT', 'skill'] = 'Git'
    df_.loc[df_['skill'] == 'GİTHUB', 'skill'] = 'GitHub'
    df_.loc[df_['skill'] == 'MS office', 'skill'] = 'Microsoft Office'
    df_.loc[df_['skill'] == 'MS Office', 'skill'] = 'Microsoft Office'
    df_.loc[df_['skill'] == 'MS Office Programları (Word, Excell, Powerpoint)', 'skill'] = 'Microsoft Office'
    df_.loc[df_['skill'] == 'MS Office Programları', 'skill'] = 'Microsoft Office'
    df_.loc[df_['skill'] == 'MS Office Applications', 'skill'] = 'Microsoft Office'
    df_.loc[df_['skill'] == 'MS Office tools', 'skill'] = 'Microsoft Office'
    df_.loc[df_['skill'] == 'MS Office Tools', 'skill'] = 'Microsoft Office'
    df_.loc[df_['skill'] == 'My SQL', 'skill'] = 'MySQL'
    df_.loc[df_['skill'] == 'Python 3', 'skill'] = 'Python'
    df_.loc[df_['skill'] == 'Python3', 'skill'] = 'Python'
    df_.loc[df_['skill'] == 'HTML 5', 'skill'] = 'HTML5'
    df_.loc[df_['skill'] == 'CSS 3', 'skill'] = 'CSS3'
    df_.loc[df_['skill'] == 'Cascading Style Sheets (CSS)', 'skill'] = 'CSS'
    df_.loc[df_['skill'] == 'Machine Learning Algorithms', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Machine Learning: Python', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Machine Learninig', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'machine learninig', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Ms Excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'Microsoft Office Excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'Excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'Excell', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'Ms Excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'MS Excel', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'MS Excell', 'skill'] = 'Microsoft Excel'
    df_.loc[df_['skill'] == 'Web Hizmetleri', 'skill'] = 'Web Services'
    df_.loc[df_['skill'] == 'Web Tasarımı', 'skill'] = 'Web Design'
    df_.loc[df_['skill'] == 'Web design', 'skill'] = 'Web Design'
    df_.loc[df_['skill'] == 'web design', 'skill'] = 'Web Design'
    df_.loc[df_['skill'] == 'MongoDb', 'skill'] = 'MongoDB'
    df_.loc[df_['skill'] == 'Mongodb', 'skill'] = 'MongoDB'
    df_.loc[df_['skill'] == 'Mongo DB', 'skill'] = 'MongoDB'
    df_.loc[df_['skill'] == 'Mongo Db', 'skill'] = 'MongoDB'
    df_.loc[df_['skill'] == 'Mongo db', 'skill'] = 'MongoDB'
    df_.loc[df_['skill'] == 'nodejs', 'skill'] = 'Node.js'
    df_.loc[df_['skill'] == 'Nodejs', 'skill'] = 'Node.js'
    df_.loc[df_['skill'] == 'Node js', 'skill'] = 'Node.js'
    df_.loc[df_['skill'] == 'TSQL', 'skill'] = 'T-SQL'
    df_.loc[df_['skill'] == 'Software Design Patterns', 'skill'] = 'Software Design'
    df_.loc[df_['skill'] == 'Software Development Methodologies', 'skill'] = 'Software Development'
    df_.loc[df_['skill'] == 'Software Development Life Cycle (SDLC)', 'skill'] = 'Software Development'
    df_.loc[df_['skill'] == 'Teamworking', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'teamwork', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'Ekip Çalışması', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'Ekip Liderliği', 'skill'] = 'Team Leadership'
    df_.loc[df_['skill'] == 'Team Work', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'ekip çalışması', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'ekip çakışması', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'Analitik Beceriler', 'skill'] = 'Analytical Skills'
    df_.loc[df_['skill'] == 'Analatik Beceriler', 'skill'] = 'Analytical Skills'
    df_.loc[df_['skill'] == 'analitik beceriler', 'skill'] = 'Analytical Skills'
    df_.loc[df_['skill'] == 'Sunum Becerileri', 'skill'] = 'Presentation Skills'
    df_.loc[df_['skill'] == 'Yönetim', 'skill'] = 'Management'
    df_.loc[df_['skill'] == 'Yönetim', 'skill'] = 'Management'
    df_.loc[df_['skill'] == 'Ürün Yönetimi', 'skill'] = 'Product Management'
    df_.loc[df_['skill'] == 'Ekip Yönetimi', 'skill'] = 'Team Management'
    df_.loc[df_['skill'] == 'Android Geliştirme', 'skill'] = 'Android Development'
    df_.loc[df_['skill'] == 'Front-end', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front-End Development', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front-end Coding', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front-end Design', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Frontend Development', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front End Development', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front End Developer', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front End Developers', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front-End', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Frontend', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Frontend Developer', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front End', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Front-end Developer', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'İş Analizi', 'skill'] = 'Business Analysis'
    df_.loc[df_['skill'] == 'Veri Analizi', 'skill'] = 'Data Analysis'
    df_.loc[df_['skill'] == 'Analizler', 'skill'] = 'Analysis'
    df_.loc[df_['skill'] == 'Analiz', 'skill'] = 'Analysis'
    df_.loc[df_['skill'] == 'Gereksinim Analizi', 'skill'] = 'Requirements Analysis'
    df_.loc[df_['skill'] == 'Gereksinim Analizi', 'skill'] = 'Requirements Analysis'
    df_.loc[df_['skill'] == 'FMEA (Hata Türleri ve Etkileri Analizi)', 'skill'] = 'Failure Modes and Effects Analysis'
    df_.loc[df_['skill'] == 'Failure Mode and Effects Analysis (FMEA)', 'skill'] = 'Failure Modes and Effects Analysis'
    df_.loc[df_['skill'] == 'FMEA ( Failure Modes and Effects Analysis)', 'skill'] = 'Failure Modes and Effects Analysis'
    df_.loc[df_['skill'] == '(FMEA) Failure Mode and Effects Analysis', 'skill'] = 'Failure Modes and Effects Analysis'
    df_.loc[df_['skill'] == 'FMEA (Failure Mode Effects Analysis)', 'skill'] = 'Failure Modes and Effects Analysis'
    df_.loc[df_['skill'] == 'Teknik Analiz', 'skill'] = 'Technical Analysis'
    df_.loc[df_['skill'] == 'Nümerik Analiz', 'skill'] = 'Numerical Analysis'
    df_.loc[df_['skill'] == 'Raporlama ve Analiz', 'skill'] = 'Reporting & Analysis'
    df_.loc[df_['skill'] == 'Raporlama', 'skill'] = 'Reporting'
    df_.loc[df_['skill'] == 'İş Geliştirme', 'skill'] = 'Business Development'
    df_.loc[df_['skill'] == 'İş Zekası', 'skill'] = 'Business Intelligence'
    df_.loc[df_['skill'] == 'Business Intelligence (BI)', 'skill'] = 'Business Intelligence'
    df_.loc[df_['skill'] == 'İş Süreçlerini İyileştirme', 'skill'] = 'Business Process Improvement'
    df_.loc[df_['skill'] == 'İş Planı', 'skill'] = 'Business Planning'
    df_.loc[df_['skill'] == 'İş İngilizcesi', 'skill'] = 'Business English'
    df_.loc[df_['skill'] == 'Yeni İş Geliştirme', 'skill'] = 'New Business Development'
    df_.loc[df_['skill'] == 'İş Analitiği', 'skill'] = 'Business Analytics'
    df_.loc[df_['skill'] == 'Bilgisayar Bilimleri', 'skill'] = 'Computer Science'
    df_.loc[df_['skill'] == 'Bilgisayar Mühendisliği', 'skill'] = 'Computer Engineering'
    df_.loc[df_['skill'] == 'Bilgisayar Güvenliği', 'skill'] = 'Computer Security'
    df_.loc[df_['skill'] == 'Bilgisayarla Görme', 'skill'] = 'Computer Vision'
    df_.loc[df_['skill'] == 'Bilgisayar Ağları', 'skill'] = 'Computer Networking'
    df_.loc[df_['skill'] == 'Bilgisayar Grafiği', 'skill'] = 'Computer Graphics'
    df_.loc[df_['skill'] == 'Bilgisayar Programcılığı', 'skill'] = 'Computer Programming'
    df_.loc[df_['skill'] == 'Bilgisayar Donanımı', 'skill'] = 'Computer Hardware'
    df_.loc[df_['skill'] == 'Bilgisayar Onarımı', 'skill'] = 'Computer Repair'
    df_.loc[df_['skill'] == 'Bilgisayar Tamiri', 'skill'] = 'Computer Repair'
    df_.loc[df_['skill'] == 'CAD (Bilgisayar Destekli Tasarım)', 'skill'] = 'Computer-Aided Design (CAD)'
    df_.loc[df_['skill'] == 'Computer vision', 'skill'] = 'Computer Vision'
    df_.loc[df_['skill'] == 'Veritabanları', 'skill'] = 'Databases'
    df_.loc[df_['skill'] == 'Büyük Veri', 'skill'] = 'Big Data'
    df_.loc[df_['skill'] == 'Oracle Veritabanı', 'skill'] = 'Oracle Database'
    df_.loc[df_['skill'] == 'Veri Yapıları', 'skill'] = 'Data Structures'
    df_.loc[df_['skill'] == 'Veri Madenciliği', 'skill'] = 'Data Mining'
    df_.loc[df_['skill'] == 'Veri Bilimi', 'skill'] = 'Data Science'
    df_.loc[df_['skill'] == 'Veritabanı Tasarımı', 'skill'] = 'Database Design'
    df_.loc[df_['skill'] == 'Veritabanı Yönetimi', 'skill'] = 'Database Administration'
    df_.loc[df_['skill'] == 'Veritabanı Geliştirme', 'skill'] = 'Database Development'
    df_.loc[df_['skill'] == 'Müşteri Hizmetleri', 'skill'] = 'Customer Service'
    df_.loc[df_['skill'] == 'Müşteri Memnuniyeti', 'skill'] = 'Customer Satisfaction'
    df_.loc[df_['skill'] == 'Customer Experience', 'skill'] = 'Müşteri Deneyimi'
    df_.loc[df_['skill'] == 'Müşteri Desteği', 'skill'] = 'Customer Support'
    df_.loc[df_['skill'] == 'Nesne Yönelimli Tasarım', 'skill'] = 'Object Oriented Design'
    df_.loc[df_['skill'] == 'Tasarım Örüntüleri', 'skill'] = 'Design Patterns'
    df_.loc[df_['skill'] == 'Tasarım', 'skill'] = 'Design'
    df_.loc[df_['skill'] == 'Ürün Geliştirme', 'skill'] = 'Product Development'
    df_.loc[df_['skill'] == 'Stratejik Planlama', 'skill'] = 'Strategic Planning'
    df_.loc[df_['skill'] == 'Planlama', 'skill'] = 'Planning'
    df_.loc[df_['skill'] == 'ERP (Kurumsal Kaynak Planlaması)', 'skill'] = 'Enterprise Resource Planning (ERP)'
    df_.loc[df_['skill'] == 'ERP', 'skill'] = 'Enterprise Resource Planning (ERP)'
    df_.loc[df_['skill'] == 'Yazılım Ürün Yönetimi', 'skill'] = 'Software Project Management'
    df_.loc[df_['skill'] == 'Linux Sistem Yönetimi', 'skill'] = 'Linux System Administration'
    df_.loc[df_['skill'] == 'Sistem Yönetimi', 'skill'] = 'System Administration'
    df_.loc[df_['skill'] == 'Sistem Analizi', 'skill'] = 'Systems Analysis'
    df_.loc[df_['skill'] == 'Sistem Mühendisliği', 'skill'] = 'Systems Engineering'
    df_.loc[df_['skill'] == 'Kalite Sistemi', 'skill'] = 'Quality System'
    df_.loc[df_['skill'] == 'Sistem Testi', 'skill'] = 'System Testing'
    df_.loc[df_['skill'] == 'Systems Design', 'skill'] = 'System Design'
    df_.loc[df_['skill'] == 'Sistem Mimarisi', 'skill'] = 'System Architecture'
    df_.loc[df_['skill'] == 'Veri Görselleştirme', 'skill'] = 'Data Visualization'
    df_.loc[df_['skill'] == 'İstatistiksel Veri Analizi', 'skill'] = 'Statistical Data Analysis'
    df_.loc[df_['skill'] == 'İlişkisel Veritabanları', 'skill'] = 'Relational Databases'
    df_.loc[df_['skill'] == 'Veri Merkezi', 'skill'] = 'Data Center'
    df_.loc[df_['skill'] == 'Veri Modelleme', 'skill'] = 'Data Modeling'
    df_.loc[df_['skill'] == 'Data Modelling', 'skill'] = 'Data Modeling'
    df_.loc[df_['skill'] == 'Veri Entegrasyonu', 'skill'] = 'Data Integration'
    df_.loc[df_['skill'] == 'Mobil Uygulamalar', 'skill'] = 'Mobile Applications'
    df_.loc[df_['skill'] == 'Mobil Uygulama Geliştirme', 'skill'] = 'Mobile Application Development'
    df_.loc[df_['skill'] == 'Android Uygulama Geliştirme', 'skill'] = 'Android Application Development'
    df_.loc[df_['skill'] == 'Android Application Developer', 'skill'] = 'Android Application Development'
    df_.loc[df_['skill'] == 'Takım Çalışması', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'Takım çalışması', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'Takım Oluşturma', 'skill'] = 'Team Building'
    df_.loc[df_['skill'] == 'Takim calismasi', 'skill'] = 'Teamwork'
    df_.loc[df_['skill'] == 'Veri Tabanı', 'skill'] = 'Databases'
    df_.loc[df_['skill'] == 'Veri Tabanları', 'skill'] = 'Databases'
    df_.loc[df_['skill'] == 'Veri tabanı', 'skill'] = 'Databases'
    df_.loc[df_['skill'] == 'Veri tabanları', 'skill'] = 'Databases'
    df_.loc[df_['skill'] == 'Veri tabanı yönetimi', 'skill'] = 'Database Administration'
    df_.loc[df_['skill'] == 'Satış Yönetimi', 'skill'] = 'Sales Management'
    df_.loc[df_['skill'] == 'BT Hizmeti Yönetimi', 'skill'] = 'IT Service Management'
    df_.loc[df_['skill'] == 'Operasyon Yönetimi', 'skill'] = 'Operations Management'
    df_.loc[df_['skill'] == 'Değişiklik Yönetimi', 'skill'] = 'Change Management'
    df_.loc[df_['skill'] == 'Zaman Yönetimi', 'skill'] = 'Time Management'
    df_.loc[df_['skill'] == 'Risk Yönetimi', 'skill'] = 'Risk Management'
    df_.loc[df_['skill'] == 'Risk Yönetimi', 'skill'] = 'Risk Management'
    df_.loc[df_['skill'] == 'Risk Analizi', 'skill'] = 'Risk Analysis'
    df_.loc[df_['skill'] == 'Risk Analizleri', 'skill'] = 'Risk Analysis'
    df_.loc[df_['skill'] == 'Finansal Analizler', 'skill'] = 'Financial Analysis'
    df_.loc[df_['skill'] == 'Yazılım Test Yaşam Döngüsü (STLC)', 'skill'] = 'Software Testing Life Cycle (STLC)'
    df_.loc[df_['skill'] == 'Yazılım Sektörü', 'skill'] = 'Software'
    df_.loc[df_['skill'] == 'Yazılım Testi', 'skill'] = 'Software Testing'
    df_.loc[df_['skill'] == 'Kriz Yönetimi', 'skill'] = 'Crisis Management'
    df_.loc[df_['skill'] == 'Kriz anlarında karar verebilme.', 'skill'] = 'Crisis Management'
    df_.loc[df_['skill'] == 'HACCP', 'skill'] = 'Hazard Analysis and Critical Control Points'
    df_.loc[df_['skill'] == 'HACCP (Tehlike Analizleri ve Kritik Kontrol Noktaları)', 'skill'] = 'Hazard Analysis and Critical Control Points'
    df_.loc[df_['skill'] == 'Hazard Analysis and Critical Control Points (HACCP)', 'skill'] = 'Hazard Analysis and Critical Control Points'
    df_.loc[df_['skill'] == 'Pazarlama', 'skill'] = 'Marketing'
    df_.loc[df_['skill'] == 'Pazarlama Stratejisi', 'skill'] = 'Marketing Strategy'
    df_.loc[df_['skill'] == 'Dijital Pazarlama', 'skill'] = 'Digital Marketing'
    df_.loc[df_['skill'] == 'Sosyal Medya Pazarlaması', 'skill'] = 'Social Media Marketing'
    df_.loc[df_['skill'] == 'Satış', 'skill'] = 'Sales'
    df_.loc[df_['skill'] == 'Satış Operasyonları', 'skill'] = 'Sales Operations'
    df_.loc[df_['skill'] == 'Oyun Geliştirme', 'skill'] = 'Game Development'
    df_.loc[df_['skill'] == 'Arka Plan Web Geliştirmesi', 'skill'] = 'Back-End Web Development'
    df_.loc[df_['skill'] == 'Ön Uç Geliştirme', 'skill'] = 'Front-end Development'
    df_.loc[df_['skill'] == 'Backend Development', 'skill'] = 'Back-End Web Development'
    df_.loc[df_['skill'] == 'Backend Developer', 'skill'] = 'Back-End Web Development'
    df_.loc[df_['skill'] == 'Backend Develepment', 'skill'] = 'Back-End Web Development'
    df_.loc[df_['skill'] == 'Makine Öğrenimi', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Makine Mühendisliği', 'skill'] = 'Mechanical Engineering'
    df_.loc[df_['skill'] == 'Makine Öğrenmesi', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Makine öğrenmesi', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Makine ögrenmesi', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Makine öğrenme', 'skill'] = 'Machine Learning'
    df_.loc[df_['skill'] == 'Go (Programming Language)', 'skill'] = 'Go'
    df_.loc[df_["skill"] == "Teknik Destek", "skill"] = "Technical Support"
    df_.loc[df_["skill"] == "Teknik Liderlik", "skill"] = "Technical Leadership"
    df_.loc[df_["skill"] == "Teknik Servisler", "skill"] = "Technical Services"
    df_.loc[df_["skill"] == "Teknik Sunumlar", "skill"] = "Technical Presentations"
    df_.loc[df_["skill"] == "Teknik Tasarım", "skill"] = "Technical Design"
    df_.loc[df_["skill"] == "Teknik İşe Alım", "skill"] = "Technical Recruitment"
    df_.loc[df_["skill"] == "Teknik Raporlar", "skill"] = "Technical Reports"
    df_.loc[df_["skill"] == "Teknik Mimari", "skill"] = "Technical Architecture"
    df_.loc[df_["skill"] == "Teknik Yazı", "skill"] = "Technical Writing"
    df_.loc[df_["skill"] == "Teknik Resim", "skill"] = "Technical Drawing"
    df_.loc[df_["skill"] == "Teknik Yazım", "skill"] = "Technical Writing"
    df_.loc[df_["skill"] == "Teknik Çeviri", "skill"] = "Technical Translation"
    df_.loc[df_["skill"] == "Teknik İletişim", "skill"] = "Technical Communication"
    df_.loc[df_["skill"] == "Bilgisayar Yeterliliği", "skill"] = "Computer Proficiency"
    df_.loc[df_["skill"] == "Etik Bilgisayar Korsanlığı", "skill"] = "Ethical Hacking"
    df_.loc[df_["skill"] == "Bilgisayar Ağı Operasyonları", "skill"] = "Computer Network Operations"
    df_.loc[df_["skill"] == "Bilgisayar Mimarisi", "skill"] = "Computer Architecture"
    df_.loc[df_["skill"] == "Bilgisayar Bakımı", "skill"] = "Computer Maintenance"
    df_.loc[df_["skill"] == "Bilgisayar Donanımı Sorun Giderme", "skill"] = "Computer Hardware Troubleshooting"
    df_.loc[df_["skill"] == "İnsan Bilgisayar Etkileşimi", "skill"] = "Human Computer Interaction"
    df_.loc[df_["skill"] == "Bilgisayar Simülasyonu", "skill"] = "Computer Simulation"
    df_.loc[df_["skill"] == "Bilgisayar Yazılımı", "skill"] = "Computer Software"
    df_.loc[df_["skill"] == "Bilgisayar", "skill"] = "Computer"
    df_.loc[df_["skill"] == "Dizüstü Bilgisayarlar", "skill"] = "Laptops"
    df_.loc[df_["skill"] == "Masaüstü Bilgisayarlar", "skill"] = "Desktop Computers"
    df_.loc[df_["skill"] == "Bilgisayarla görme", "skill"] = "Computer Vision"
    df_.loc[df_["skill"] == "Bilgisayar programcılığı", "skill"] = "Computer programming"
    df_.loc[df_["skill"] == "Bilgisayar yazılım", "skill"] = "computer software"
    df_.loc[df_["skill"] == "CAM (Bilgisayar Destekli İmalat)", "skill"] = "CAM (Computer Aided Manufacturing)"
    df_.loc[df_["skill"] == "Bilgisayarlı Görme", "skill"] = "Computer Vision"
    df_.loc[df_["skill"] == "Bilgisayar Grafikleri", "skill"] = "Computer Graphics"
    df_.loc[df_["skill"] == "Bilgisayarlı Görü", "skill"] = "Computer Vision"
    df_.loc[df_["skill"] == "Bilgisayarlı Sistem Validasyonu", "skill"] = "Computerized System Validation"
    df_.loc[df_["skill"] == "Bilgisayar Donanım", "skill"] = "Computer Hardware"
    df_.loc[df_["skill"] == "Görüntü İşleme", "skill"] = "Image Processing"
    df_.loc[df_["skill"] == "İş Stratejisi", "skill"] = "Business Strategy"
    df_.loc[df_["skill"] == "İşletim Sistemleri", "skill"] = "Operating Systems"
    df_.loc[df_["skill"] == "Bulut Bilgi İşlem", "skill"] = "Cloud Computing"
    df_.loc[df_["skill"] == "Doğal Dil İşleme", "skill"] = "Natural Language Processing"
    df_.loc[df_["skill"] == "Gerçek Zamanlı İşletim Sistemleri (RTOS)", "skill"] = "Real Time Operating Systems (RTOS)"
    df_.loc[df_["skill"] == "RTOS", "skill"] = "Real Time Operating Systems (RTOS)"
    df_.loc[df_["skill"] == "Real-Time Operating Systems (RTOS)", "skill"] = "Real Time Operating Systems (RTOS)"
    df_.loc[df_["skill"] == "Sinyal İşleme", "skill"] = "Signal Processing"
    df_.loc[df_["skill"] == "Mikro İşlemciler", "skill"] = "Microprocessors"
    df_.loc[df_["skill"] == "Bilgi İşlem", "skill"] = "Computing"
    df_.loc[df_["skill"] == "İşletme", "skill"] = "Business"
    df_.loc[df_["skill"] == "Dijital Sinyal İşleme", "skill"] = "Digital Signal Processing"
    df_.loc[df_["skill"] == "Dijital Görüntü İşleme", "skill"] = "Digital Image Processing"
    df_.loc[df_["skill"] == "İş Süreci", "skill"] = "Business Process"
    df_.loc[df_["skill"] == "İş Zekası Araçları", "skill"] = "Business Intelligence Tools"
    df_.loc[df_["skill"] == "İşe Alma", "skill"] = "Recruitment"
    df_.loc[df_["skill"] == "İş Denetim Dili (JCL)", "skill"] = "Job Control Language (JCL)"
    df_.loc[df_["skill"] == "VMware İş İstasyonu", "skill"] = "Vmware Workstation"
    df_.loc[df_["skill"] == "İş Süreci Yönetimi", "skill"] = "Business Process Management"
    df_.loc[df_["skill"] == "Gıda İşleme", "skill"] = "Food Processing"
    df_.loc[df_["skill"] == "Liman İşçisi", "skill"] = "Docker"
    df_.loc[df_["skill"] == "İş Süreci Tasarımı", "skill"] = "Business Process Design"
    df_.loc[df_["skill"] == "İşlemciler", "skill"] = "Processors"
    df_.loc[df_["skill"] == "İş İlişkisi Yönetimi", "skill"] = "Business Relationship Management"
    df_.loc[df_["skill"] == "İş Modellemesi", "skill"] = "Business Modeling"
    df_.loc[df_["skill"] == "Isıl İşlem", "skill"] = "Heat Treatment"
    df_.loc[df_["skill"] == "E-İş", "skill"] = "E-Business"
    df_.loc[df_["skill"] == "İş Sürekliliği Planlaması", "skill"] = "Business Continuity Planning"
    df_.loc[df_["skill"] == "Uluslararası İşletme", "skill"] = "International Business"
    df_.loc[df_["skill"] == "Kanal İş Ortakları", "skill"] = "Channel Partners"
    df_.loc[df_["skill"] == "İşaret Dili", "skill"] = "Sign Language"
    df_.loc[df_["skill"] == "İş Uygulamaları", "skill"] = "Business Applications"
    df_.loc[df_["skill"] == "İş Gücü Planlaması", "skill"] = "Workforce Planning"
    df_.loc[df_["skill"] == "Uluslararası İş Geliştirme", "skill"] = "International Business Development"
    df_.loc[df_["skill"] == "Toplu İşlem", "skill"] = "Batch Processing"
    df_.loc[df_["skill"] == "İş Ağı Oluşturma", "skill"] = "Business Networking"
    df_.loc[df_["skill"] == "Ödeme Kartı İşleme", "skill"] = "Payment Card Processing"
    df_.loc[df_["skill"] == "İş İçgörüleri", "skill"] = "Business Insights"
    df_.loc[df_["skill"] == "Küçük İşletmeler", "skill"] = "Small Business"
    df_.loc[df_["skill"] == "İşaret dili", "skill"] = "Sign Language"
    df_.loc[df_["skill"] == "İş Değerlendirmesi", "skill"] = "Job Evaluation"
    df_.loc[df_["skill"] == "Küçük İşletme Yönetimi", "skill"] = "Small Business Management"
    df_.loc[df_["skill"] == "İş analizi", "skill"] = "Business Analysis"
    df_.loc[df_["skill"] == "Ses İşleme", "skill"] = "Audio Processing"
    df_.loc[df_["skill"] == "BT İşe Alımı", "skill"] = "It Recruitment"
    df_.loc[df_["skill"] == "İş Gücü Geliştirme", "skill"] = "Workforce Development"
    df_.loc[df_["skill"] == "Paralel Bilgi İşlem", "skill"] = "Parallel Computing"
    df_.loc[df_["skill"] == "İş Devamlılığı", "skill"] = "Business Continuity"
    df_.loc[df_["skill"] == "İş Analisti", "skill"] = "Business Analyst"
    df_.loc[df_["skill"] == "İş Akış Yönetimi", "skill"] = "Workflow Management"
    df_.loc[df_["skill"] == "Ağ Yönetimi", "skill"] = "Network Management"
    df_.loc[df_["skill"] == "Kalite Yönetimi", "skill"] = "Quality Management"
    df_.loc[df_["skill"] == "Test Yönetimi", "skill"] = "Test Management"
    df_.loc[df_["skill"] == "Sunucu Yönetimi", "skill"] = "Server Management"
    df_.loc[df_["skill"] == "BT Yönetimi", "skill"] = "IT Management"
    df_.loc[df_["skill"] == "Veritabanı Yönetimi Sistemi (DBMS)", "skill"] = "Database Management System (DBMS)"
    df_.loc[df_["skill"] == "Bilgi Güvenliği Yönetimi", "skill"] = "Information Security Management"
    df_.loc[df_["skill"] == "Etkinlik Yönetimi", "skill"] = "Event Management"
    df_.loc[df_["skill"] == "Web Projesi Yönetimi", "skill"] = "Web Project Management"
    df_.loc[df_["skill"] == "Program Yönetimi", "skill"] = "Program Management"
    df_.loc[df_["skill"] == "İçerik Yönetimi", "skill"] = "Content Management"
    df_.loc[df_["skill"] == "Mühendislik Yönetimi", "skill"] = "Engineering Management"
    df_.loc[df_["skill"] == "Lojistik Yönetimi", "skill"] = "Logistics Management"
    df_.loc[df_["skill"] == "Veri Yönetimi", "skill"] = "Data Management"
    df_.loc[df_["skill"] == "Üretim Yönetimi", "skill"] = "Production Management"
    df_.loc[df_["skill"] == "Problem Yönetimi", "skill"] = "Problem Management"
    df_.loc[df_["skill"] == "Tedarikçi Yönetimi", "skill"] = "Supplier Management"
    df_.loc[df_["skill"] == "Stres Yönetimi", "skill"] = "Stress Management"
    df_.loc[df_["skill"] == "Gereksinim Yönetimi", "skill"] = "Requirements Management"
    df_.loc[df_["skill"] == "Sözleşme Yönetimi", "skill"] = "Contract Management"
    df_.loc[df_["skill"] == "Hesap Yönetimi", "skill"] = "Account Management"
    df_.loc[df_["skill"] == "Konfigürasyon Yönetimi", "skill"] = "Configuration Management"
    df_.loc[df_["skill"] == "Yönetim Kurulu", "skill"] = "Board Of Directors"
    df_.loc[df_["skill"] == "BT Altyapı Yönetimi", "skill"] = "IT Infrastructure Management"
    df_.loc[df_["skill"] == "Bulut Yönetimi", "skill"] = "Cloud Management"
    df_.loc[df_["skill"] == "İnsan Yönetimi", "skill"] = "People Management"
    df_.loc[df_["skill"] == "Marka Yönetimi", "skill"] = "Brand Management"
    df_.loc[df_["skill"] == "Güvenlik Yönetimi", "skill"] = "Security Management"
    df_.loc[df_["skill"] == "Ürün Yaşam Döngüsü Yönetimi", "skill"] = "Product Lifecycle Management"
    df_.loc[df_["skill"] == "Pazarlama Yönetimi", "skill"] = "Marketing Management"
    df_.loc[df_["skill"] == "İnşaat Yönetimi", "skill"] = "Construction Management"
    df_.loc[df_["skill"] == "Müşteri Yönetimi", "skill"] = "Customer Management"
    df_.loc[df_["skill"] == "Yönetim Danışmanlığı", "skill"] = "Management Consulting"
    df_.loc[df_["skill"] == "Stratejik Yönetim", "skill"] = "Strategic Management"
    df_.loc[df_["skill"] == "Bilgi Yönetimi", "skill"] = "Information Management"
    df_.loc[df_["skill"] == "Kurumsal İçerik Yönetimi", "skill"] = "Enterprise Content Management"
    df_.loc[df_["skill"] == "Sanat Yönetimi", "skill"] = "Art Direction"
    df_.loc[df_["skill"] == "Kaynak Yönetimi", "skill"] = "Resource Management"
    df_.loc[df_["skill"] == "Gönüllü Yönetimi", "skill"] = "Volunteer Management"
    df_.loc[df_["skill"] == "Depo Yönetimi", "skill"] = "Warehouse Management"
    df_.loc[df_["skill"] == "Reklam Yönetimi", "skill"] = "Advertising Management"
    df_.loc[df_["skill"] == "Çevre Yönetimi Sistemleri", "skill"] = "Environmental Management Systems"
    df_.loc[df_["skill"] == "Enerji Yönetimi", "skill"] = "Energy Management"
    df_.loc[df_["skill"] == "Sürüm Yönetimi", "skill"] = "Release Management"
    df_.loc[df_["skill"] == "Maliyet Yönetimi", "skill"] = "Cost Management"
    df_.loc[df_["skill"] == "BT Yönetim", "skill"] = "It Management"
    df_.loc[df_["skill"] == "Proje Yönetimi Bilgi Tabanı (PMBOK)", "skill"] = "Project Management Knowledge Base (PMBOK)"
    df_.loc[df_["skill"] == "Acil Durum Yönetimi", "skill"] = "Emergency Management"
    df_.loc[df_["skill"] == "Yalın Yönetim", "skill"] = "Lean Management"
    df_.loc[df_["skill"] == "Paydaş Yönetimi", "skill"] = "Stakeholder Management"
    df_.loc[df_["skill"] == "Şikayet Yönetimi", "skill"] = "Complaint Management"
    df_.loc[df_["skill"] == "Olay Yönetimi", "skill"] = "Incident Management"
    df_.loc[df_["skill"] == "Çatışma Yönetimi", "skill"] = "Conflict Management"
    df_.loc[df_["skill"] == "Topluluk Yönetimi", "skill"] = "Community Management"
    df_.loc[df_["skill"] == "Mobil Cihaz Yönetimi", "skill"] = "Mobile Device Management"
    df_.loc[df_["skill"] == "Uluslararası Proje Yönetimi", "skill"] = "International Project Management"
    df_.loc[df_["skill"] == "Kurumsal Risk Yönetimi", "skill"] = "Enterprise Risk Management"
    df_.loc[df_["skill"] == "Grafik Tasarımı", "skill"] = "Graphic Design"
    df_.loc[df_["skill"] == "PCB Tasarım", "skill"] = "PCB Design"
    df_.loc[df_["skill"] == "Oyun Tasarımı", "skill"] = "Game Design"
    df_.loc[df_["skill"] == "Adobe Tasarım Programları", "skill"] = "Adobe Design Programs"
    df_.loc[df_["skill"] == "Kullanıcı Arabirimi Tasarımı", "skill"] = "User Interface Design"
    df_.loc[df_["skill"] == "Algoritma Tasarımı", "skill"] = "Algorithm Design"
    df_.loc[df_["skill"] == "Ağ Tasarımı", "skill"] = "Network Design"
    df_.loc[df_["skill"] == "Devre Tasarımı", "skill"] = "Circuit Design"
    df_.loc[df_["skill"] == "Kontrol Sistemleri Tasarımı", "skill"] = "Control Systems Design"
    df_.loc[df_["skill"] == "Sistem Tasarımı", "skill"] = "System Design"
    df_.loc[df_["skill"] == "Kullanıcı Deneyimi Tasarımı (UED)", "skill"] = "User Experience Design (UED)"
    df_.loc[df_["skill"] == "WordPress Tasarımı", "skill"] = "Wordpress Design"
    df_.loc[df_["skill"] == "Analog Devre Tasarımı", "skill"] = "Analog Circuit Design"
    df_.loc[df_["skill"] == "Ürün Tasarımı", "skill"] = "Product Design"
    df_.loc[df_["skill"] == "Web Uygulama Tasarımı", "skill"] = "Web Application Design"
    df_.loc[df_["skill"] == "Tasarım Mühendisliği", "skill"] = "Design Engineering"
    df_.loc[df_["skill"] == "Tasarımcı Düşünce", "skill"] = "Design Thinking"
    df_.loc[df_["skill"] == "Kullanıcı Deneyimi Tasarımı", "skill"] = "User Experience Design"
    df_.loc[df_["skill"] == "3D Tasarımı", "skill"] = "3D Design"
    df_.loc[df_["skill"] == "Mobil Tasarım", "skill"] = "Mobile Design"
    df_.loc[df_["skill"] == "Öğretim Tasarımı", "skill"] = "Instructional Design"
    df_.loc[df_["skill"] == "Elektriksel Tasarım", "skill"] = "Electrical Design"
    df_.loc[df_["skill"] == "Tümleşik Devre Tasarımı", "skill"] = "Integrated Circuit Design"
    df_.loc[df_["skill"] == "Güvenlik Mimarisi Tasarımı", "skill"] = "Security Architecture Design"
    df_.loc[df_["skill"] == "Tasarım Araştırması", "skill"] = "Design Research"
    df_.loc[df_["skill"] == "Proje Tasarımı", "skill"] = "Project Design"
    df_.loc[df_["skill"] == "Proses Tasarımı", "skill"] = "Process Design"
    df_.loc[df_["skill"] == "Makine Tasarımı", "skill"] = "Machine Design"
    df_.loc[df_["skill"] == "Mühendislik Tasarımı", "skill"] = "Engineering Design"
    df_.loc[df_["skill"] == "Ses Tasarımı", "skill"] = "Sound Design"
    df_.loc[df_["skill"] == "İmalat İçin Tasarım", "skill"] = "Design For Manufacturing"
    df_.loc[df_["skill"] == "Seviye Tasarımı", "skill"] = "Level Design"
    df_.loc[df_["skill"] == "Kavramsal Tasarım", "skill"] = "Conceptual Design"
    df_.loc[df_["skill"] == "Tasarım Yönetimi", "skill"] = "Design Management"
    df_.loc[df_["skill"] == "Etkileşim Tasarımı", "skill"] = "Interaction Design"
    df_.loc[df_["skill"] == "Grafik Tasarım", "skill"] = "Graphic Design"
    df_.loc[df_["skill"] == "Uçak Tasarımı", "skill"] = "Aircraft Design"
    df_.loc[df_["skill"] == "Kentsel Tasarım", "skill"] = "Urban Design"
    df_.loc[df_["skill"] == "Tasarım Gözden Geçirme", "skill"] = "Design Review"
    df_.loc[df_["skill"] == "Kurumsal Yazılım", "skill"] = "Enterprise Software"
    df_.loc[df_["skill"] == "ERP Yazılımı", "skill"] = "Erp Software"
    df_.loc[df_["skill"] == "Tümleşik Yazılım", "skill"] = "Embedded Software"
    df_.loc[df_["skill"] == "Yazılım Uygulama", "skill"] = "Software Application"
    df_.loc[df_["skill"] == "Arena Simülasyon Yazılımı", "skill"] = "Arena Simulation Software"
    df_.loc[df_["skill"] == "Kötü Amaçlı Yazılım Analizi", "skill"] = "Malware Analysis"
    df_.loc[df_["skill"] == "Yazılım Projeleri", "skill"] = "Software Projects"
    df_.loc[df_["skill"] == "Gömülü Yazılım", "skill"] = "Embedded Software"
    df_.loc[df_["skill"] == "Yazılım Yaşam Döngüsü", "skill"] = "Software Lifecycle"
    df_.loc[df_["skill"] == "Yazılım Gereksinimleri", "skill"] = "Software Requirements"
    df_.loc[df_["skill"] == "Muhasebe Yazılımı", "skill"] = "Accounting Software"
    df_.loc[df_["skill"] == "Proje Yönetimi Yazılımı", "skill"] = "Project Management Software"
    df_.loc[df_["skill"] == "Autodesk Yazılımı", "skill"] = "Autodesk Software"
    df_.loc[df_["skill"] == "Gömülü Sistem Yazılımı", "skill"] = "Embedded System Software"
    df_.loc[df_["skill"] == "Yazılım Test", "skill"] = "Software Testing"
    df_.loc[df_["skill"] == "Kurumsal Yazılım Geliştirme", "skill"] = "Enterprise Software Development"
    df_.loc[df_["skill"] == "Yazılım Mimarisi", "skill"] = "Software Architecture"
    df_.loc[df_["skill"] == "Yazılım Destek", "skill"] = "Software Support"
    df_.loc[df_["skill"] == "Kurumsal Yazılım Mimarisi", "skill"] = "Enterprise Software Architecture"
    df_.loc[df_["skill"] == "Yazılım Analizi", "skill"] = "Software Analysis"
    df_.loc[df_["skill"] == "Yazılım test", "skill"] = "Software Testing"
    df_.loc[df_["skill"] == "Yazılım Test Otomasyon", "skill"] = "Software Test Automation"
    df_.loc[df_["skill"] == "Yazılımcı", "skill"] = "Programmer"
    df_.loc[df_["skill"] == "Emniyet Kritik Yazılım Geliştirme", "skill"] = "Safety Critical Software Development"
    df_.loc[df_["skill"] == "Özgür Yazılım", "skill"] = "Free Software"
    df_.loc[df_["skill"] == "Yazılım Varlık Yönetimi", "skill"] = "Software Asset Management"
    df_.loc[df_["skill"] == "Yazılım Test Mühendisi", "skill"] = "Software Test Engineer"
    df_.loc[df_["skill"] == "Kontrol Yazılımı", "skill"] = "Control Software"
    df_.loc[df_["skill"] == "Kurumsal Yazılım", "skill"] = "Enterprise Software"
    df_.loc[df_["skill"] == "ERP Yazılımı", "skill"] = "Erp Software"
    df_.loc[df_["skill"] == "Tümleşik Yazılım", "skill"] = "Embedded Software"
    df_.loc[df_["skill"] == "Yazılım Uygulama", "skill"] = "Software Application"
    df_.loc[df_["skill"] == "Arena Simülasyon Yazılımı", "skill"] = "Arena Simulation Software"
    df_.loc[df_["skill"] == "Kötü Amaçlı Yazılım Analizi", "skill"] = "Malware Analysis"
    df_.loc[df_["skill"] == "Yazılım Projeleri", "skill"] = "Software Projects"
    df_.loc[df_["skill"] == "Gömülü Yazılım", "skill"] = "Embedded Software"
    df_.loc[df_["skill"] == "Yazılım Yaşam Döngüsü", "skill"] = "Software Lifecycle"
    df_.loc[df_["skill"] == "Yazılım Gereksinimleri", "skill"] = "Software Requirements"
    df_.loc[df_["skill"] == "Muhasebe Yazılımı", "skill"] = "Accounting Software"
    df_.loc[df_["skill"] == "Proje Yönetimi Yazılımı", "skill"] = "Project Management Software"
    df_.loc[df_["skill"] == "Autodesk Yazılımı", "skill"] = "Autodesk Software"
    df_.loc[df_["skill"] == "Gömülü Sistem Yazılımı", "skill"] = "Embedded System Software"
    df_.loc[df_["skill"] == "Yazılım Test", "skill"] = "Software Testing"
    df_.loc[df_["skill"] == "Kurumsal Yazılım Geliştirme", "skill"] = "Enterprise Software Development"
    df_.loc[df_["skill"] == "Yazılım Mimarisi", "skill"] = "Software Architecture"
    df_.loc[df_["skill"] == "Yazılım Destek", "skill"] = "Software Support"
    df_.loc[df_["skill"] == "Kurumsal Yazılım Mimarisi", "skill"] = "Enterprise Software Architecture"
    df_.loc[df_["skill"] == "Yazılım Analizi", "skill"] = "Software Analysis"
    df_.loc[df_["skill"] == "Yazılım test", "skill"] = "Software Testing"
    df_.loc[df_["skill"] == "Yazılım Test Otomasyon", "skill"] = "Software Test Automation"
    df_.loc[df_["skill"] == "Yazılımcı", "skill"] = "Programmer"
    df_.loc[df_["skill"] == "Emniyet Kritik Yazılım Geliştirme", "skill"] = "Safety Critical Software Development"
    df_.loc[df_["skill"] == "Özgür Yazılım", "skill"] = "Free Software"
    df_.loc[df_["skill"] == "Yazılım Varlık Yönetimi", "skill"] = "Software Asset Management"
    df_.loc[df_["skill"] == "Yazılım Test Mühendisi", "skill"] = "Software Test Engineer"
    df_.loc[df_["skill"] == "Kontrol Yazılımı", "skill"] = "Control Software"
    df_.loc[df_["skill"] == "Proje Mühendisliği", "skill"] = "Project Engineering"
    df_.loc[df_["skill"] == "Proje Koordinasyonu", "skill"] = "Project Coordination"
    df_.loc[df_["skill"] == "Proje Kontrolü", "skill"] = "Project Control"
    df_.loc[df_["skill"] == "Proje Teslimi", "skill"] = "Project Delivery"
    df_.loc[df_["skill"] == "Proje Tahmini", "skill"] = "Project Estimation"
    df_.loc[df_["skill"] == "Proje Uygulaması", "skill"] = "Project Implementation"
    df_.loc[df_["skill"] == "Proje Ekipleri", "skill"] = "Project Teams"
    df_.loc[df_["skill"] == "Proje Portföy Yönetimi", "skill"] = "Project Portfolio Management"
    df_.loc[df_["skill"] == "Proje Planları", "skill"] = "Project Plans"
    df_.loc[df_["skill"] == "Proje Yöneticileri", "skill"] = "Project Management"
    df_.loc[df_["skill"] == "Proje Geliştirme", "skill"] = "Project Development"
    df_.loc[df_["skill"] == "Dijital Proje Yönetimi", "skill"] = "Digital Project Management"
    df_.loc[df_["skill"] == "Proje Yazımı", "skill"] = "Project Writing"
    df_.loc[df_["skill"] == "Proje Dökümantasyon", "skill"] = "Project Documentation"
    df_.loc[df_["skill"] == "Kuruluş Proje Yönetimi (EPM)", "skill"] = "Enterprise Project Management (EPM)"
    df_.loc[df_["skill"] == "Proje Takibi", "skill"] = "Project Tracking"
    df_.loc[df_["skill"] == "Agile Proje Management", "skill"] = "Agile Project Management"
    df_.loc[df_["skill"] == "Proje Yöneticiliği", "skill"] = "Project Management"
    df_.loc[df_["skill"] == "Proje Analizi", "skill"] = "Project Analysis"
    df_.loc[df_["skill"] == "Problem Analizi", "skill"] = "Problem Analysis"
    df_.loc[df_["skill"] == "Kök Neden Analizi", "skill"] = "Root Cause Analysis"
    df_.loc[df_["skill"] == "Sonlu Elemanlar Analizi", "skill"] = "Finite Element Analysis"
    df_.loc[df_["skill"] == "Hata Analizi", "skill"] = "Error Analysis"
    df_.loc[df_["skill"] == "Yapısal Analiz", "skill"] = "Structural Analysis"
    df_.loc[df_["skill"] == "Regresyon Analizi", "skill"] = "Regression Analysis"
    df_.loc[df_["skill"] == "Piyasa Analizi", "skill"] = "Market Analysis"
    df_.loc[df_["skill"] == "Zaman Serisi Analizi", "skill"] = "Time Series Analysis"
    df_.loc[df_["skill"] == "Pazarlama Analizleri", "skill"] = "Marketing Analytics"
    df_.loc[df_["skill"] == "Trend Analizi", "skill"] = "Trend Analysis"
    df_.loc[df_["skill"] == "Algoritma Analizi", "skill"] = "Algorithm Analysis"
    df_.loc[df_["skill"] == "Proses Analizi", "skill"] = "Process Analysis"
    df_.loc[df_["skill"] == "İhtiyaç Analizi", "skill"] = "Needs Analysis"
    df_.loc[df_["skill"] == "Görüntü Analizi", "skill"] = "Image Analysis"
    df_.loc[df_["skill"] == "Analiz Hizmetleri", "skill"] = "Analysis Services"
    df_.loc[df_["skill"] == "Fonksiyonel Analiz", "skill"] = "Functional Analysis"
    df_.loc[df_["skill"] == "Müşteri Analizi", "skill"] = "Customer Analysis"
    df_.loc[df_["skill"] == "Tehlike Analizi", "skill"] = "Hazard Analysis"
    df_.loc[df_["skill"] == "Sayısal Analiz", "skill"] = "Numerical Analysis"
    df_.loc[df_["skill"] == "Devre Analizi", "skill"] = "Circuit Analysis"
    df_.loc[df_["skill"] == "Paydaş Analizi", "skill"] = "Stakeholder Analysis"
    df_.loc[df_["skill"] == "İstihbarat Analizi", "skill"] = "Intelligence Analysis"
    df_.loc[df_["skill"] == "Uzamsal Analiz", "skill"] = "Spatial Analysis"
    df_.loc[df_["skill"] == "Stres Analizi", "skill"] = "Stress Analysis"
    df_.loc[df_["skill"] == "Kredi Analizi", "skill"] = "Credit Analysis"
    df_.loc[df_["skill"] == "Zafiyet Analizi", "skill"] = "Vulnerability Analysis"
    df_.loc[df_["skill"] == "Duygu Analizi", "skill"] = "Sentiment Analysis"
    df_.loc[df_["skill"] == "Teknoloji İhtiyaçları Analizi", "skill"] = "Technology Needs Analysis"
    df_.loc[df_["skill"] == "Temel Analiz", "skill"] = "Fundamental Analysis"
    df_.loc[df_["skill"] == "Fiyatlandırma Analizi", "skill"] = "Pricing Analysis"
    df_.loc[df_["skill"] == "Eğitim", "skill"] = "Education"
    df_.loc[df_["skill"] == "Profesyonel Eğitim", "skill"] = "Professional Education"
    df_.loc[df_["skill"] == "Çalışan Eğitimi", "skill"] = "Employee Training"
    df_.loc[df_["skill"] == "Kişisel Eğitim", "skill"] = "Personal Training"
    df_.loc[df_["skill"] == "Uzaktan Eğitim", "skill"] = "Distance Learning"
    df_.loc[df_["skill"] == "Uçuş Eğitimi", "skill"] = "Flight Training"
    df_.loc[df_["skill"] == "Matematik Eğitimi", "skill"] = "Mathematics Education"
    df_.loc[df_["skill"] == "Yetişkin Eğitimi", "skill"] = "Adult Education"
    df_.loc[df_["skill"] == "Son Kullanıcı Eğitimi", "skill"] = "End User Training"
    df_.loc[df_["skill"] == "Fitness Eğitimi", "skill"] = "Fitness Training"
    df_.loc[df_["skill"] == "Müzik Eğitimi", "skill"] = "Music Education"
    df_.loc[df_["skill"] == "Eğitimci", "skill"] = "Trainer"
    df_.loc[df_["skill"] == "Mesleki Eğitim", "skill"] = "Vocational Education"
    df_.loc[df_["skill"] == "Güvenlik Eğitimi", "skill"] = "Safety Training"
    df_.loc[df_["skill"] == "K-12 Eğitimi", "skill"] = "K-12 Education"
    df_.loc[df_["skill"] == "Sağlık Eğitimi", "skill"] = "Health Education"
    df_.loc[df_["skill"] == "Anket Tasarımı", "skill"] = "Survey Design"
    df_.loc[df_["skill"] == "Basım Tasarımı", "skill"] = "Print Design"
    df_.loc[df_["skill"] == "Sayısal Tasarım", "skill"] = "Digital Design"
    df_.loc[df_["skill"] == "Elektronik Devre Tasarımı", "skill"] = "Electronic Circuit Design"
    df_.loc[df_["skill"] == "Tasarım Odaklı Düşünme", "skill"] = "Design Thinking"
    df_.loc[df_["skill"] == "Elektronik Donanım Tasarımı", "skill"] = "Electronic Hardware Design"
    df_.loc[df_["skill"] == "Aydınlatma Tasarımı", "skill"] = "Lighting Design"
    df_.loc[df_["skill"] == "Gömülü Sistem Tasarımı", "skill"] = "Embedded System Design"
    df_.loc[df_["skill"] == "Optik Tasarım", "skill"] = "Optical Design"
    df_.loc[df_["skill"] == "Hareket Tasarımı", "skill"] = "Motion Design"
    df_.loc[df_["skill"] == "Tasarım Düşüncesi", "skill"] = "Design Thinking"
    df_.loc[df_["skill"] == "Müfredat Tasarımı", "skill"] = "Curriculum Design"
    df_.loc[df_["skill"] == "Tasarım Stratejisi", "skill"] = "Design Strategy"
    df_.loc[df_["skill"] == "Elektronik Tasarım", "skill"] = "Electronic Design"
    df_.loc[df_["skill"] == "Dergi Tasarım", "skill"] = "Magazine Design"
    df_.loc[df_["skill"] == "Tasarım Desenleri", "skill"] = "Design Patterns"
    df_.loc[df_["skill"] == "Responsive Web Tasarım", "skill"] = "Responsive Web Design"
    df_.loc[df_["skill"] == "Marka Tasarımı", "skill"] = "Brand Design"
    df_.loc[df_["skill"] == "İç Mekan Tasarımı", "skill"] = "Interior Design"
    df_.loc[df_["skill"] == "Veri Tabanı Tasarım", "skill"] = "Database Design"
    df_.loc[df_["skill"] == "Tekstil Tasarımı", "skill"] = "Textile Design"
    df_.loc[df_["skill"] == "Ön Uç Mühendislik Tasarımı (FEED)", "skill"] = "Front End Engineering Design (FEED)"
    df_.loc[df_["skill"] == "Mekanik Tasarım", "skill"] = "Mechanical Design"
    df_.loc[df_["skill"] == "Organizasyonel Tasarım", "skill"] = "Organizational Design"
    df_.loc[df_["skill"] == "Deneyim Tasarımı", "skill"] = "Experience Design"
    df_.loc[df_["skill"] == "Çelik Tasarım", "skill"] = "Steel Design"
    df_.loc[df_["skill"] == "Fonksiyonel Tasarım", "skill"] = "Functional Design"
    df_.loc[df_["skill"] == "Donanım Tasarımı", "skill"] = "Hardware Design"
    df_.loc[df_["skill"] == "RTL Tasarımı", "skill"] = "RTL Design"
    df_.loc[df_["skill"] == "Uygulama Geliştirme", "skill"] = "Application Development"
    df_.loc[df_["skill"] == "iOS Uygulaması Geliştirme", "skill"] = "IOS App Development"
    df_.loc[df_["skill"] == "Test Odaklı Geliştirme", "skill"] = "Test Driven Development"
    df_.loc[df_["skill"] == "Program Geliştirme", "skill"] = "Program Development"
    df_.loc[df_["skill"] == "İçerik Geliştirme", "skill"] = "Content Development"
    df_.loc[df_["skill"] == "iPhone Uygulama Geliştirme", "skill"] = "Iphone Application Development"
    df_.loc[df_["skill"] == "Marka Geliştirme", "skill"] = "Brand Development"
    df_.loc[df_["skill"] == "Müfredat Geliştirme", "skill"] = "Curriculum Development"
    df_.loc[df_["skill"] == "Platformlar Arası Geliştirme", "skill"] = "Cross-Platform Development"
    df_.loc[df_["skill"] == "Bulut Uygulama Geliştirme", "skill"] = "Cloud Application Development"
    df_.loc[df_["skill"] == "Tedarikçi Geliştirme", "skill"] = "Supplier Development"
    df_.loc[df_["skill"] == "Sistem Geliştirme", "skill"] = "System Development"
    df_.loc[df_["skill"] == "Geliştirme Projeleri", "skill"] = "Development Projects"
    df_.loc[df_["skill"] == "Çözüm Geliştirme", "skill"] = "Solution Development"
    df_.loc[df_["skill"] == "Web Sitesi Geliştirme", "skill"] = "Web Development"
    df_.loc[df_["skill"] == "Yenilik Geliştirme", "skill"] = "Innovation Development"
    df_.loc[df_["skill"] == "SaaS Geliştirme", "skill"] = "Saas Development"
    df_.loc[df_["skill"] == "Yazlım Geliştirme", "skill"] = "Software Development"
    df_.loc[df_["skill"] == "BT Geliştirme", "skill"] = "IT Development"
    df_.loc[df_["skill"] == "Prosedür Geliştirme", "skill"] = "Procedure Development"
    df_.loc[df_["skill"] == "Kariyer Geliştirme", "skill"] = "Career Development"
    df_.loc[df_["skill"] == "Geliştirme Araçları", "skill"] = "Development Tools"
    df_.loc[df_["skill"] == "Yemek Tarifi Geliştirme", "skill"] = "Recipe Development"
    df_.loc[df_["skill"] == "Menü Geliştirme", "skill"] = "Menu Development"
    df_.loc[df_["skill"] == "İlaç Geliştirme", "skill"] = "Drug Development"
    df_.loc[df_["skill"] == "Hızlı Uygulama Geliştirme (RAD)", "skill"] = "Rapid Application Development (RAD)"
    df_.loc[df_["skill"] == "Sunum Geliştirme", "skill"] = "Presentation Development"
    df_.loc[df_["skill"] == "Masaüstü Uygulama Geliştirme", "skill"] = "Desktop Application Development"
    df_.loc[df_["skill"] == "Algoritma Geliştirme", "skill"] = "Algorithm Development"
    df_.loc[df_["skill"] == "Arduino Geliştirme", "skill"] = "Arduino Development"
    df_.loc[df_["skill"] == "Piyasa Geliştirme", "skill"] = "Market Development"
    df_.loc[df_["skill"] == "Web Uygulama Geliştirme", "skill"] = "Web Application Development"
    df_.loc[df_["skill"] == "Yöntem Geliştirme", "skill"] = "Method Development"
    df_.loc[df_["skill"] == "Use Case Analizi", "skill"] = "Use Case Analizi"
    df_.loc[df_["skill"] == "Algoritma Tasarımı ve Analizi", "skill"] = "Algorithm Design And Analysis"
    df_.loc[df_["skill"] == "Arıza Analizleri", "skill"] = "Fault Analysis"
    df_.loc[df_["skill"] == "Kümeleme Analizi", "skill"] = "Cluster Analysis"
    df_.loc[df_["skill"] == "Faktör Analizi", "skill"] = "Factor Analysis"
    df_.loc[df_["skill"] == "ANSYS Yapısal Analiz", "skill"] = "ANSYS Structural Analysis"
    df_.loc[df_["skill"] == "Tolerans Analizi", "skill"] = "Tolerance Analysis"
    df_.loc[df_["skill"] == "Test Analiz", "skill"] = "Test Analysis"
    df_.loc[df_["skill"] == "Veri Tabanı Analizi", "skill"] = "Database Analysis"
    df_.loc[df_["skill"] == "Adli Bilişim Analizi", "skill"] = "Forensic Analysis"
    df_.loc[df_["skill"] == "Malware Analizi", "skill"] = "Malware Analizi"
    df_.loc[df_["skill"] == "Ağ Analizi", "skill"] = "Network Analysis"
    df_.loc[df_["skill"] == "Uygulama Programlaması Arayüzleri", "skill"] = "Application Programming Interfaces"
    df_.loc[df_["skill"] == "Grafik Kullanıcı Arayüzü (GUI)", "skill"] = "Graphical User Interface (GUI)"
    df_.loc[df_["skill"] == "İnsan Makine Arayüzü", "skill"] = "Human Machine Interface"
    df_.loc[df_["skill"] == "Arayüzler", "skill"] = "Interfaces"
    df_.loc[df_["skill"] == "Uygulama Programlama Arayüzü", "skill"] = "Application Programming Interface"
    df_.loc[df_["skill"] == "Kullanıcı Arayüzü Tasarımı", "skill"] = "User Interface Design"
    df_.loc[df_["skill"] == "Talep Yönetimi", "skill"] = "Demand Management"
    df_.loc[df_["skill"] == "Kalite Yönetim", "skill"] = "Quality Management"
    df_.loc[df_["skill"] == "Altyapı Yönetimi", "skill"] = "Infrastructure Management"
    df_.loc[df_["skill"] == "Tesis Yönetimi (FM)", "skill"] = "Facility Management (FM)"
    df_.loc[df_["skill"] == "Müşteri Hizmet Yönetimi", "skill"] = "Customer Service Management"
    df_.loc[df_["skill"] == "Mağaza Yönetimi", "skill"] = "Store Management"
    df_.loc[df_["skill"] == "Teknoloji Yönetimi", "skill"] = "Technology Management"
    df_.loc[df_["skill"] == "Bayi Yönetimi", "skill"] = "Dealer Management"
    df_.loc[df_["skill"] == "Sistemler Yönetimi", "skill"] = "Systems Management"
    df_.loc[df_["skill"] == "Dosya Yönetimi", "skill"] = "File Management"
    df_.loc[df_["skill"] == "İhale Yönetimi", "skill"] = "Tender Management"
    df_.loc[df_["skill"] == "Depolama Yönetimi", "skill"] = "Storage Management"
    df_.loc[df_["skill"] == "Uygulama Yaşam Döngüsü Yönetimi", "skill"] = "Application Lifecycle Management"
    df_.loc[df_["skill"] == "Ulaştırma Yönetimi", "skill"] = "Transportation Management"
    df_.loc[df_["skill"] == "Finansal Risk Yönetimi", "skill"] = "Financial Risk Management"
    df_.loc[df_["skill"] == "Kalite Yönetim Sistemleri", "skill"] = "Quality Management Systems"
    df_.loc[df_["skill"] == "Hastane Bilgi Yönetim Sistemi", "skill"] = "Hospital Information Management System"
    df_.loc[df_["skill"] == "Veri Tabanı Yönetimi", "skill"] = "Database Management"
    df_.loc[df_["skill"] == "Laboratuvar Bilgi Yönetim Sistemi (LIMS)", "skill"] = "Laboratory Information Management System (LIMS)"
    df_.loc[df_["skill"] == "Mali Raporlama", "skill"] = "Financial Reporting"
    df_.loc[df_["skill"] == "Performans Raporlama", "skill"] = "Performance Reporting"
    df_.loc[df_["skill"] == "Finansal Raporlama", "skill"] = "Financial Reporting"
    df_.loc[df_["skill"] == "Veri Raporlama", "skill"] = "Data Reporting"
    df_.loc[df_["skill"] == "Araştırıcı Raporlama", "skill"] = "Investigative Reporting"
    df_.loc[df_["skill"] == "Raporlama Gereksinimleri", "skill"] = "Reporting Requirements"
    df_.loc[df_["skill"] == "Raporlama Aracı", "skill"] = "Reporting Tool"
    df_.loc[df_["skill"] == "A3 Raporlama", "skill"] = "A3 Raporlama"
    df_.loc[df_["skill"] == "Finansal Raporlamalar", "skill"] = "Financial Reporting"
    df_.loc[df_["skill"] == "Maliyet Raporlama", "skill"] = "Cost Reporting"
    df_.loc[df_["skill"] == "Topluluk Önünde Konuşma", "skill"] = "Public Speaking"
    df_.loc[df_["skill"] == "Satış Öncesi", "skill"] = "Pre-Sales"
    df_.loc[df_["skill"] == "Önleyici Bakım", "skill"] = "Preventive Maintenance"
    df_.loc[df_["skill"] == "Ön muhasebe", "skill"] = "Accounting"
    df_.loc[df_["skill"] == "Önleme", "skill"] = "Prevention"
    df_.loc[df_["skill"] == "Veri Ön İşleme", "skill"] = "Data Preprocessing"
    df_.loc[df_["skill"] == "Mobil Oyunlar", "skill"] = "Mobile Games"
    df_.loc[df_["skill"] == "Mobil Cihazlar", "skill"] = "Mobile Devices"
    df_.loc[df_["skill"] == "Mobil Platformlar", "skill"] = "Mobile Platforms"
    df_.loc[df_["skill"] == "Mobil İletişimler", "skill"] = "Mobile Communications"
    df_.loc[df_["skill"] == "Mobil Reklamcılık", "skill"] = "Mobile Advertising"
    df_.loc[df_["skill"] == "Mobil Ödemeler", "skill"] = "Mobile Payments"
    df_.loc[df_["skill"] == "Mobil Programlama", "skill"] = "Mobile Programming"
    df_.loc[df_["skill"] == "Mobil Pazarlama", "skill"] = "Mobile Marketing"
    df_.loc[df_["skill"] == "Mobil Uygulama Testi", "skill"] = "Mobile Application Testing"
    df_.loc[df_["skill"] == "Mobil Uygulama Tasarımı", "skill"] = "Mobile Application Design"
    df_.loc[df_["skill"] == "Android Mobil Uygulama Geliştirme", "skill"] = "Android Mobile Application Development"
    df_.loc[df_["skill"] == "Mobil Güvenlik", "skill"] = "Mobile Security"
    df_.loc[df_["skill"] == "Mobil Uygulama", "skill"] = "Mobile Application"
    df_.loc[df_["skill"] == "Mobil Yazılım", "skill"] = "Mobile Software"
    df_.loc[df_["skill"] == "Mobil Uygulama Test", "skill"] = "Mobile Application Testing"
    df_.loc[df_["skill"] == "Mobil Uygulama Gelistirme", "skill"] = "Mobile Application Development"
    df_.loc[df_["skill"] == "Mobil Yazılm", "skill"] = "Mobile Software"
    df_.loc[df_['skill'] == 'C++ Language', 'skill'] = "C++"
    df_.loc[df_['skill'] == 'Dev C++', 'skill'] = "C++"
    df_.loc[df_['skill'] == 'C/C++', 'skill'] = "C, C++"
    df_.loc[df_['skill'] == 'Jquery', 'skill'] = "jQuery"
    df_.loc[df_['skill'] == 'JS', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'JScript', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'js', 'skill'] = 'JavaScript'
    df_.loc[df_['skill'] == 'MS-SQL', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MS SQL', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MS SQL SERVER', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'SQL Server Management Studio', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MSQL', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'MS SQL Server', 'skill'] = 'Microsoft SQL Server'
    df_.loc[df_['skill'] == 'NOSQL', 'skill'] = 'NoSQL'
    df_.loc[df_['skill'] == 'NOSQL', 'skill'] = 'NoSQL'
    df_.loc[df_['skill'] == 'Sun Certified Java Programmer', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Programming', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Programming Language', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Sun Sertifikalı Java Programcısı', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Programlama', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Programlama 1 (SE) (İsmek- 09.2017  12.2017)', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Programlama 3 (Spring Core & MVC) (İsmek- 12.2017 04.2018)', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Programlama 2 (EE) (İsmek- 12.2017  04.2018)', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Programlama Dili İle Uygulama Geliştirme', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Program Geliştirme', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'Java Programmer', 'skill'] = 'Java'
    df_.loc[df_['skill'] == 'C# Programming', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'C# Programlama Dili', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'C# Programming Language', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'C# Programlamaya Giriş', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'C# Programlama,', 'skill'] = 'C#'
    df_.loc[df_['skill'] == 'C Programming', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C Programming Language', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C Programing Language', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C Programing', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C Programmin Language', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'Advanced C Programming', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'Embedded C Programming', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C Progamming', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'C Programming Languages', 'skill'] = 'C'
    df_.loc[df_['skill'] == 'Python Programming', 'skill'] = 'Python'
    df_.loc[df_['skill'] == 'Python Programlama', 'skill'] = 'Python'

    return df_

def fix_studies(dataframe: pd.DataFrame) -> pd.DataFrame:

    df_ = dataframe.copy()
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Mühendisliği', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektrik ve Elektronik Mühendisliği', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Programlama', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'bilgisayar programcılığı', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Programcılığı', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'bilgilsayar programcılığı', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Matematik', 'fields_of_study'] = 'Mathematics'
    df_.loc[df_['fields_of_study'] == 'İşletme ve Yönetim, Genel', 'fields_of_study'] = 'Business Administration and Management, General'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Yazılımı Mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Makine Mühendisliği', 'fields_of_study'] = 'Mechanical Engineering'
    df_.loc[df_['fields_of_study'] == 'Fizik', 'fields_of_study'] = 'Physics'
    df_.loc[df_['fields_of_study'] == 'Ekonomi', 'fields_of_study'] = 'Economics'
    df_.loc[df_['fields_of_study'] == 'İstatistik', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'Kimya', 'fields_of_study'] = 'Chemistry'
    df_.loc[df_['fields_of_study'] == 'Elektrik Mühendisliği', 'fields_of_study'] = 'Electrical Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektrik mühendisliği', 'fields_of_study'] = 'Electrical Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektrik Mühendisliği', 'fields_of_study'] = 'Electrical Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektrik Mühendisi', 'fields_of_study'] = 'Electrical Engineering'
    df_.loc[df_['fields_of_study'] == 'elektrik mühendisliği', 'fields_of_study'] = 'Electrical Engineering'
    df_.loc[df_['fields_of_study'] == 'bilgisayar mühendisliği', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'bilgisayar mühendisliği ', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'bilgisayar mühendisligi', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'bilgisayar müh.', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'bilgisayar mühendisi', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'yönetim bilişim sistemleri', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yönetim Bilişim Sistemleri', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Computer Engineering BSC', 'fields_of_study'] = 'Computer Engineering' 
    df_.loc[df_['fields_of_study'] == 'Computer Engineering, BE', 'fields_of_study'] = 'Computer Engineering' 
    df_.loc[df_['fields_of_study'] == ' Computer Engineering', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'computer engineering', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'computer engineer', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'computer engineerig', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'computer engeneering', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'computer science', 'fields_of_study'] = 'Computer Science'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Bilimleri', 'fields_of_study'] = 'Computer Science'
    df_.loc[df_['fields_of_study'] == 'computer sciences', 'fields_of_study'] = 'Computer Science'
    df_.loc[df_['fields_of_study'] == 'computer scientist', 'fields_of_study'] = 'Computer Science'
    df_.loc[df_['fields_of_study'] == 'computer Engineering', 'fields_of_study'] = 'Computer Engineering' 
    df_.loc[df_['fields_of_study'] == 'computer Engineer', 'fields_of_study'] = 'Computer Engineering' 
    df_.loc[df_['fields_of_study'] == 'computer programming', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'computer programmer', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Programlama/Programcı, Genel', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'elektrik elektronik mühendisliği', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'elektrik elektronik mühendisi', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'elektrik elektronik', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'endüstri mühendisliği', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği/Industrial Engineering', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği / Industrial Engineering', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği, Mühendislik Yönetimi ', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisi', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği Yüksek Lisans', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği-Mühendislik Yönetimi', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği (Tezli)', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği / Müh. Yönetimi ', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği Ana Bilim Dalı', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği Lisans', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği Yöneylem Araştırması Anabilim Dalı', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği (EN)', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği - Yan Dal Programı', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Lisans-Endüstri Mühendisliği', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği, Tam Burslu', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği (Minor)', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Industrial Engineering (Endüstri Mühendisliği)', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Sistem Mühendisliği (Endüstri Mühendisliği)', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Industrial Engineering / Endüstri Mühendisliği', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği (Industrial Engineer)', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'Endüstri Mühendisliği lisans', 'fields_of_study'] = 'Industrial Engineering'
    df_.loc[df_['fields_of_study'] == 'matematik', 'fields_of_study'] = 'Mathematics' 
    df_.loc[df_['fields_of_study'] == 'matematik(ingilizce)', 'fields_of_study'] = 'Mathematics'
    df_.loc[df_['fields_of_study'] == 'kimya', 'fields_of_study'] = 'Chemistry'
    df_.loc[df_['fields_of_study'] == 'KİMYA', 'fields_of_study'] = 'Chemistry'
    df_.loc[df_['fields_of_study'] == 'KİMYAGER', 'fields_of_study'] = 'Chemistry'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management, General', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management (English)', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management, Technology Track', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management, MBA', 'fields_of_study'] = 'MBA'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management, Executive', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management (Master)', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management (German)', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management.Open University', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Faculty of Business Administration and Management', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Bachelor of Business Administration, Business Administration and Management', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Istanbul Master degree in Business Administration and Management, General', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management, Toronto ON ', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'Business Administration and Management, Marketing', 'fields_of_study'] = 'Business Administration and Management'
    df_.loc[df_['fields_of_study'] == 'MAKİNE MÜHENDİSLİĞİ', 'fields_of_study'] = 'Mechanical Engineering'
    df_.loc[df_['fields_of_study'] == 'BİLGİSAYAR PROGRAMCILIĞI', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'BİLGİSAYAR MÜHENDİSLİĞİ', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'BİLGİSAYAR PROGRAMCILIĞI - ÖN LİSANS', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'yazılım mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'yazılım mühendisliği ', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'yazılım mühendiliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'yazılım mühendisligi', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'yazılım mühendisi', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım mühendisliği ', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım mühendisi', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisi', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği ', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendislik', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == ' Yazılım Mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği / Software Engineering', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği (Çift Anadal)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği Yüksek Lisans Programı', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği ve Veri Bilimi', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği(İngilizce)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği Software Engineering', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği MS', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği (UOLP)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği (%100 Burslu)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği(Software Engineering)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Bilimleri Fakültesi, Yazılım Mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Software Engineering - Yazılım Mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Müh.', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği (Uzaktan)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği , Ortalama :  3.1 / 4', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği ( Ingilizce )', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği U.O.L.P (ingilizce)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği (EN/TR)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği(EN/TR), Comprehensive Scholarship (%100)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği(English)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Yazılım Mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisligi', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği; Bilgisayar Tek. ve Prog', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar ve Yazılım Mühendisliği', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliğ', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Yazılım Mühendisliği (Software Engineering)', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Sofware Developer', 'fields_of_study'] = 'Software Developer'
    df_.loc[df_['fields_of_study'] == 'Sofware Engineer', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Sofware Engineering', 'fields_of_study'] = 'Software Engineering'
    df_.loc[df_['fields_of_study'] == 'Computer science', 'fields_of_study'] = 'Computer Science'
    df_.loc[df_['fields_of_study'] == 'computer science', 'fields_of_study'] = 'Computer Science'
    df_.loc[df_['fields_of_study'] == 'Computer engineering', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Computer engineer', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Computer engineering ', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Computer engineerig', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Computer engineering and natural sciences faculty', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'COMPUTER ENGINEERING', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == '( ENG. FAC.) / COMPUTER ENGINEERING', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'COMPUTER ENGINEER', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'COMPUTER ENGİNEER', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'COMPUTER ENGİNEERİNG', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'business administration', 'fields_of_study'] = 'Business Administration'
    df_.loc[df_['fields_of_study'] == 'School of business administration', 'fields_of_study'] = 'Business Administration'
    df_.loc[df_['fields_of_study'] == 'school of business administration', 'fields_of_study'] = 'Business Administration'
    df_.loc[df_['fields_of_study'] == 'MBA(Master of business administration)', 'fields_of_study'] = 'Business Administration'
    df_.loc[df_['fields_of_study'] == 'business administrative  (english)', 'fields_of_study'] = 'Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of business administration (MBA)', 'fields_of_study'] = 'Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of business administration', 'fields_of_study'] = 'Business Administration'
    df_.loc[df_['fields_of_study'] == 'Statistic', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'Statictics', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'BSc, Statistics', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'B.Sc., Statistics', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'Statistic ', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'Statictics', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'Statistics (English)', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'Department of Statistics', 'fields_of_study'] = 'Statistics'
    df_.loc[df_['fields_of_study'] == 'Electronics and Communications Engineering', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'elektronik ve haberleşme mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'elektronik ve haberleşme ', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'elektronik ve haberlesme muhendisligi', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'elektronik ve haberleșme mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'elektronik ve haberleşme mühendisi', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'elektronik ve haberleşme mühendisi ', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'elektronik ve haberleşme mühendisligi', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'ELEKTRONİK VE HABERLEŞME MÜHENDİSLİĞİ', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'ELEKTRONİK VE HABERLEŞME', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'ELEKTRONİK HABERLEŞME', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'ELEKTRONİK HABERLEŞME MÜHENDİSLİĞİ', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'ELEKTRONİK HABERLEŞME TEKNOLOJİSİ', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (MBA)', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration - MBA', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Adminstration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (M.B.A.)', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration(MBA)', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Executive Master of Business Administration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'MBA (Master of Business Administration)', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Institute of Social Sciences, Master of Business Administration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administrator (with Thesis)', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administrator', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Executive Master of Business Adminstrations', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration MBA', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (MBA) (Without Thesis)', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (MBA) , Executive MBA', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (e-MBA)', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Information Systems', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (MBA) - English', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'MBA, Master of Business Administration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (MBA), Master degree', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business and Administration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration MBA - thesis program', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration - MBA, Social Sciences', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (MBA), Hospital and Health Institutions Management', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master’s degree • Master of Business Administration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (M.B.A.), Executive M.B.A', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration ', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration,MBA', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'MBA - Master of Business Administration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration (M.B.A), Marketing/Marketing Management, General', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration ( MBA )', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration, MBA', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Management and Strategy - Master of Business Administration', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration, Executive MBA', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Master of Business Administration in Finance', 'fields_of_study'] = 'Master of Business Administration'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği ', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisi', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisligi', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği (İngilizce)', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği(İngilizce)', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği, 3.39', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği (İng)', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği/ Mathematical Engineering', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği 3,22', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisligi (%100 Ingilizce)', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Department of Mathematical Engineering / Matematik Mühendisliği', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği (%100 İngilizce)', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği (Ingilizce) ', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Kimya Metalurji fakültesi, Matematik Mühendisliği', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik Mühendisliği Yüksek Lisans', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik mühendisliği', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik mühendisliği ', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'Matematik mühendisliği 3.33', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'matematik mühendisliği', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'matematik mühendisi', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'matematik mühendisliği / mathematical engineering', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'mathematical enginering', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'mathematical Engineering', 'fields_of_study'] = 'Mathematical Engineering'
    df_.loc[df_['fields_of_study'] == 'mathematical engineering', 'fields_of_study'] = 'Mathematical Engineering'
    
    df_.loc[df_['fields_of_study'] == 'Kimya Mühendisliği', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineer', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical and Biological Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical engineer', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Department of Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical and Biological Engineering (English)', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Process Technology', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering ', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Food Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering, Polymer', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering-Transition to Computer Engineering/Drop out', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering/Chemical Technologics', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering, 3.55/4, Graduated with third degree', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering, 3.97', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering (%100 English)', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering, 3.64', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Technologies', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering Department', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering / Master', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering Process and Reactor Design', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Engineering and Applied Chemistry %100 Scholarship', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Master Student (thesis stage), Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'B.S., Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'B.Sc Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical and Biochemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == "Chemical Engineering Master's Programme", 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical and Bioprocess Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'B.Sc.,Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'M.Sc. Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'M.Sc., Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'BSc, Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'MSc, Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Kimya Mühendisliği - Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical, Biological, Radiological and Nuclear Defense', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical and Process Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'B.Sc. Chemical Engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'Chemical Enginnering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'chemical engineering', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'chemical engineer', 'fields_of_study'] = 'Chemical Engineering'
    df_.loc[df_['fields_of_study'] == 'chemical', 'fields_of_study'] = 'Chemical Engineering'

    df_.loc[df_['fields_of_study'] == 'Computer Engineer', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrical and Electronic Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrical & Electronics Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrical-Electronics Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrical Electronics Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrical&Electronics Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electric and Electronic Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrical and Electronics Engineer', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrical - Electronics Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electric and Electronics Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrical & Electronics Engineer', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Electrics and Electronics Engineering', 'fields_of_study'] = 'Electrical and Electronics Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme Mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme Mühendisi', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme Mühendisliği ', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme Müh.', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme Teknolojisi', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektrik, Elektronik ve Haberleşme Mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == ' Elektronik ve Haberleşme Mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Habeleşme Mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme Mühendisliği (İngilizce)', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == ' Elektronik ve Haberleşme Mühendisliği, Bilgisayar Mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberlesme Muhendisligi', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme Mühendisligi', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberlesme Muh.', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Lisans (BSc), Elektronik ve Haberleşme Mühendisliği', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Yüksek Lisans (MSc), Elektronik ve Haberleşme Mühendisliği, Elektronik Programı', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronik ve Haberleşme Mühendisliği.', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronics and Communication Engineering', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronics and Communication Enginnering', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronics and Telecommunication Engineering', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronics and Communications Engineering', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'Elektronics and Communication engineering', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'electronics and communication engineering', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'electronics and Communication Engineer', 'fields_of_study'] = 'Electronics and Communication Engineering'
    df_.loc[df_['fields_of_study'] == 'electronics and commnication engineering', 'fields_of_study'] = 'Electronics and Communication Engineering'

    df_.loc[df_['fields_of_study'] == 'Bilgisayar Mühendisliği (İngilizce)', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar programcılığı', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Teknolojisi ve Programlama', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Programlama, Özel Uygulamalar', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Teknolojileri ve Programlama', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Mühendisi', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Mühendisliği Yüksek Lisans', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Programcılığı / Tekniker', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'B.Sc., Faculty of Computer and Information Science, Computer Engineering', 'fields_of_study'] = 'Computer Engineering'
    df_.loc[df_['fields_of_study'] == 'bilgisayar programcısı', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'bilgisayar programcılığı', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'bilgisayar programlama,bilgisayar teknoljileri', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'bilgisayar teknolojisi ve programlama', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'bilgisayar programcılığı ', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'bilgisayar programlama', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Bilgisayar Teknolojileri ve bilgisayar programcılığı', 'fields_of_study'] = 'Computer Programming'
    df_.loc[df_['fields_of_study'] == 'Yönetim Bilgi Sistemleri, Genel', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems, General', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information System', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (MIS)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems and Engineering', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems and Services', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yönetim Bilişim Sistemleri - Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems ', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems-MBA', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Sciences', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Informations System', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information System (MIS)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Auzef • Yönetim Bilişim Sistemleri / Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems(MIS)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == ' Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems - MIS', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems Engineering', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'MBA-  Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (English) ', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yönetim Bilişim Sistemleri / Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (Ph.D.)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'IT Institute/Management Information Sytem', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'MIS - Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == "Bachelor's Degree Management Information Systems", 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Master of Management Information Systems ( M.I.S. )', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (Master)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems, Full Scholarship', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems Master’s Program (with thesis)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Sys. And Eng.', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems and Engineering ', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Informations System (MIS)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems MIS', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'M.S, Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems & Engineering', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'MIS(Management Information Systems)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == ' Management Informaton Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems and Engineering (MIS)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems Technologies ', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == "Master's degree, Management Information Systems", 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yönetim Bilişim Sistemleri ( Management Information Systems )', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'MIS Management Information Systems - Yönetim Bilişi', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems, Honours Degree', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yüksek Lisans (Master) / Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (Wirtschaftsinformatik)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == "Master's Degree, Management Information Systems (MIS).", 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'MIS (Management Information Systems)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Astronomy , Computer Programming, Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (Graduate School)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (Success Scholarship)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == "Faculty of Engineering Management Information Systems Master's Degree", 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (MIS) Master', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == "Management Information Systems / Master's Degree", 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Master of Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == '(MIS) Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yönetim Bilişim Sistemleri (Management Information Systems)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == "Bachelor's Degree, Management Information Systems ", 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Technology', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Engineer Faculty - Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems in German (Wirtschaftsinformatik)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == ' Faculty of Commercial Sciences, Department of Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Bachelor’s Degree in Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems, General (Yönetim Bilişim Sistemleri)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (M.I.S)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (MSc)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (3.41/4)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems, Full Scholarship Student', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Informatics Systems,Undergradute Program', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Informatics Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (M.I.S.)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yönetim Bilişim Sistemleri (MIS - Management Information Systems)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'MBA - Management Information Systems and Services', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yönetim Bilişim Sistemleri, Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Sytems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems,MIS', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (Yönetim Bilişim Sistemleri)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'MBA / Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == ' Management Information System (MIS)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems / Master', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information System (MIS) - Yönetim Bilişim Sistemleri', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == ' Management Information Systems ( Yönetim Bilişim Sistemleri )', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems Specialist ', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == "bachelor's degree • Management Information Systems", 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems, General (in German)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == ' Management Information Systems and Engineering ', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems (German)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information System and Engineering', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Information Systems, 3.26', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management information systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management information systems ', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management information system/MBA', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Yönetim bilişim sistemleri mühendisliği /Management information systems engineering (%100 İngilizce)', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management information System', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Infırmation Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'Management Infomations Systems', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'yönetim bilişim ve sistemleri', 'fields_of_study'] = 'Management Information Systems'
    df_.loc[df_['fields_of_study'] == 'işletme', 'fields_of_study'] = 'İşletme'
    df_.loc[df_['fields_of_study'] == 'işletme ', 'fields_of_study'] = 'İşletme'
    df_.loc[df_['fields_of_study'] == 'işletme bölümü', 'fields_of_study'] = 'İşletme'
    df_.loc[df_['fields_of_study'] == 'işletme/işletmecilik', 'fields_of_study'] = 'İşletme'
    df_.loc[df_['fields_of_study'] == 'işletmecilik', 'fields_of_study'] = 'İşletme'
    df_.loc[df_['fields_of_study'] == 'İngilizce işletme', 'fields_of_study'] = 'İşletme'

    return df_

def fix_school_names(dataframe: pd.DataFrame) -> pd.DataFrame:

    df_ = dataframe.copy()
    df_.loc[df_['school_name'] == 'Anadolu University', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Eskişehir Anadolu University ', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu University (www.anadolu.edu.tr)', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Eskişehir Anadolu University', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu University;', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Eskisehir Anadolu University', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu University ', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu University, Eskisehir, Turkey', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi/ Anadolu University', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu University (Open)', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu University, Faculty of Engineering and Architecture', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'University 2\t\tAnadolu University- Isletme Faculty / Eskisehir (Correspondence\tSchool)\t\t\t\t\tDepartment of Business', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi / Anatolian University', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Eskişehir Anadolu Üniversitesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Eskişehir Anadolu Üniversitesi İktisat Fakültesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi İşletme Fakültesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Eskişehir Anadolu Üniversitesi Açık öğretim Fakultesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi / Açıköğretim Fakültesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi (Eskişehir)', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi Açık Öğretim Fakültesi Sigortacılık ve Bankacılık', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi sosyal hizmetler', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi AÖF / İşletme', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi (AÖF)', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi Açik Öğretim', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi Açık Öğretim Fakültesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi İsletme Fakultesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi Açıköğretim Fakültesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi Açık Öğretim', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi ', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Karadeniz Teknik Üniversitesi, Anadolu Üniversitesi', 'school_name'] = 'Karadeniz Technical University, Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu Üniversitesi Bilgisayar Mühendisliği', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Anadolu üniversitesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Eskişehir anadolu üniversitesi', 'school_name'] = 'Anadolu University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi / Yildiz Technical University', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Technical University', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi Fen Bilimleri Enstitüsü', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Technical University ', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik University', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi Kimya Mühendisliği Bölümü', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi Meslek Yüksek Okulu', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi Matematik Mühendisi', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Technical University Institute of Science and Technology', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Tek', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi / Yıldız Technical University', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi (Yildiz Technical University)', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi Elektrik-Elektronik Fakültesi Elektronik Ve Haberleşme Mühendisliği Bölüm', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Tehcnical University', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Tecnical University', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Universitesi', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üniversitesi Kimya Mühendisliği', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız Teknik Üni', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yıldız teknik üniversitesi', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yildiz Teknik Üniversitesi', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yildiz Technical University &  Istanbul Institute', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yildiz Teknik Universitesi', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yildiz Technical University ', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yildiz Technical University Physics Department', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'University\tYildiz Teknik Üniversitesi - Istanbul', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yildiz Technical University, Graduate School of Science and Engineering', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'University\tYildiz Technical  University - Istanbul Naval Architecture and Marine', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'Yildiz Technical University- Graduate School of Natural and Applied Sciences', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'yıldız teknik üniversitesi', 'school_name'] = 'Yildiz Technical University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi-Cerrahpaşa', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi / Istanbul University', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi İstanbul Tıp Fakültesi İç Hastalıkları ABD Beslenme Doktora Programı', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi İstanbul Tıp Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi ', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi AUZEF', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi / İşletme', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi Devlet Konservatuari', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi Cerrahpaşa', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi İstanbul/Türkiye', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi Teknik Bilimler MYO', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi,İstanbul Tıp Fakültesi Temel Bilimler Anabilim Dalı-Mikrobiyoloji', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi-Hasan Ali Yücel Eğitim Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi-Fen-Edebiyat Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi / Economics', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi Açık ve Uzaktan Eğitim Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi Veterinerlik Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi (AUZEF)', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi Mühendislik Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == ' İstanbul Üniversitesi ', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi İşletme Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi MYO', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi  Mühendislik Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi / İletişim Fakültesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'İstanbul Üniversitesi Biyoloji / İstanbul Teknik Üniversitesi Yazılım Uzmanlığı', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul Üniversitesi', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University-Cerrahpasa', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul Univertisy Faculty of Medicine', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University, Istanbul, Turkey', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University Social Sciences Institute', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University  İF İİE', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University - Cerrahpasa', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University Language Center', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul Univercity', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University,MBA', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul Universty', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University -Cerrahpasa', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University AUZEF', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University State Conservatory', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Current\tIstanbul University', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Sisli Vocational High School, Department of Telecommunication, IstanbulIstanbul University', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'University\t\t\tIstanbul University  / Istanbul', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University - Cerrahpaşa', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Preperation School in Istanbul University', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Istanbul University Faculty of  Literature', 'school_name'] = 'Istanbul University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üni', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniveristesi', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi/University of Sakarya', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi/Yönetim Bilişim Sistemleri', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversity', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi(lisans)', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi Karasu Meslek Yüksek Okulu', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi,Bilgisayar ve Bilişim Fakültesi,Bilgisayar Mühendisliği', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi Yabancı Diller Bölümü', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniverstesi', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi / Karasu Meslek Yüksekokulu', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi (Endüstri Mühendisliği)', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi Fen Bilimleri Enstitüsü', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversty', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi Biyomedikal Mühendisliği', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi ', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi (2015 - 2020)', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi Ka', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi Hendek Meslek Yüksek Okulu', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Üniversitesi Bilgisayar ve Bilişim Bilimleri Fakültesi', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'University of Sakarya', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya University', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya University (www.sakarya.edu.tr)', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya University of Applied Sciences', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya University, Faculty of Technical Education', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Universitesi', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya University, Turkey', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'Sakarya Universty', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'sakarya üniversitesi', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'sakarya ünivesitesi', 'school_name'] = 'Sakarya University'
    df_.loc[df_['school_name'] == 'İstanbul Teknik Üniversitesi / Istanbul Technical University', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technical Uni', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'University (Postgraduate Degree) Istanbul Technical University', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technical University (ITU)', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Faculty of Electronical and Communication Engineering, Istanbul Technical University', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technical Univercity', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technical University, Istanbul', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technical University, Computer Engineering', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technical University, Institue of Science and Technology', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technic University', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technical University ', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == ' Istanbul Technical University', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Technical University, Faculty of Mechanical Engineering', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Teknik Üniversitesi', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Teknik Universitesi', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Istanbul Teknik University', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'İstanbul Teknik Üniversitesi', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'İstanbul Teknik Üniversitesi ', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'İstanbul Teknik Üniversitesi & Mindset Institute', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'İstanbul Teknik Üniversitesi / Bilgisayar Mühendisliği', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'İstanbul Teknik Üniversitesi Bilişim Teknolojileri CCNA eğitimi sertifikası', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'İstanbul Teknik Üniversitesi Meslek Yüksekokulu Elektronik Ve Haberleşme Mühendisliği Bölümü', 'school_name'] = 'Istanbul Technical University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversitesi', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversity', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversitesi / University of Kocaeli', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversitesi Köseköy Meslek Yüksek Okulu', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversitesi MYO', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversitesi ', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniver', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversitesi elektronik ve haberleşme mühendisliği', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversitesi Hazırlık Okulu (İngilizce)', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üni', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Üniversitesi"', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli University ', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Universitesi', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Universty', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli University Faculty Of Education', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli University English Preparatory School', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli üniversitesi', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli universitesi', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'kocaeli Üniversitesi Mühendislik Fakültesi', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'kocaeli üniversitesi', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'University of Kocaeli', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Kocaeli Ã\x9cniversitesi', 'school_name'] = 'Kocaeli University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi İşletme Fakültesi', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi-Marmara University', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi Iktisadi ve İdari Bilimler', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi (İstanbul Şehir Üniversitesi) ', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi Teknik Eğitim Fakültesi mezunu', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi Fen Bilimleri Enstitüsü', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi İletişim Fakültesi', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Univeristy', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University-Contemporaray Business Management', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University English Preparation School', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == '2006-2010          Marmara University', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University,Turkey', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University TMYO', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University ', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == ' Marmara University', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Üniversitesi-Marmara University', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University \xad Istanbul', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University, Istanbul/Turkey', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Universitesi', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Computer Engineering - Marmara University', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Universty', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara Universitesi-Turkey', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara University, İstanbul', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Wirtschaftsinformatik-Marmara University', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Marmara üniversitesi', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] ==  'Marmara üni', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] ==  'marmara üniversitesi', 'school_name'] = 'Marmara University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi Bilgisayar Mühendisliği', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi / Hacettepe University', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'University of Hacettepe', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi Bilgisayar Öğretmenliği', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Universitesi', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi ', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Univercity', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi, Near East University', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi Engineering Faculty', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi-Chemical Engineer', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'B.S.Food Engineering , 1997: Hacettepe University, Ankara', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi Computer Engineering', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi Tıp Fakültesi', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe University, Computer Science and Engineering (BSc)', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversitesi Elektrik ve Elektronik Mühendisliği Bölümü', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Universitesi - Elektrik Elektronik Mühendisligi', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Üniversity', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacettepe Universitesi Elektrik-Elektronik Muh.', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Hacetepe University', 'school_name'] = 'Hacettepe University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi Yabancı Diller Yüksekokulu', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi ', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi Uluslararası Bilgisayar Enstitüsü', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi İzmir/Türkiye', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi Bilgisayar Mühendisliği', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi Meslek Yüksekokulu', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi - International Computer Institute ', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi Elektrik-Elektronik Mühendisliği', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi Ege Meslek Yüksekokulu', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Üniversitesi, ', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege University International Computer Institute', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == "Ege University, Int'l Computer Institute", 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege University, Computer Engineering Dep.', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege University Department of Stem Cell', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Univercity', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege University Turkey', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Univeristy', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Universiy Computer Engineering ', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'International Computer Institute Ege University', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege University, İzmir', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'IEEE Ege University Student Branch', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege Universty', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege University - Izmir/TURKEY', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege univeristy', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Ege university', 'school_name'] = 'Ege University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi İktisadi İdari Bilimler Fakültesi İşletme Bölümü', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi  (Ön Lisans)', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi Teknoloji Fakültesi', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi(Gazi University)', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi Vakfı Özel Anadolu Lisesi', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi Gazi Meslek Yüksek Okulu', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniveristesi ', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi Teknik Eğitim Fakültesi', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi S.M.Y.O', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi ', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi / Gazi University', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Ankara Gazi Üniversitesi ', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi üniversitesi', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'gazi üniversitesi', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi University (www.gazi.edu.tr)', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Universitesi', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi University Foundation Private Science High School', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi University, Computer Engineering Department', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi(Gazi University)', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Universty', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi University Foundation Science High School', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi University Department of Computer Engineering', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi University, Ankara', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi Üniversitesi / Gazi University', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'Gazi University ', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'gazi university', 'school_name'] = 'Gazi University'
    df_.loc[df_['school_name'] == 'GFN & Bahcesehir University Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahcesehir University, Istanbul', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahcesehir University Rome Campus', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahcesehir University | Wissen Academie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahcesehir University & Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahcesehir University Berlin Campus', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahcesehir University -', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Turk Telekom Academy and COOP(Bahcesehir Uni)', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Turkcell Academy and COOP(Bahcesehir Uni)', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir University', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'GFN & Bahçeşehir University Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir University Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Berlin International Bahçeşehir University', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Wissen Bahçeşehir Universitesi', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'BAU Bahçeşehir Universität Berlin', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'GFN & Bahçeşehir Üniversitesi Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi Wissen Academie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi & Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi (Wissen Akademi)', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi - Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'GFN & Bahçeşehir Üniversitesi Wissen Akademi', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Wissen Akademie GFN & Bahçeşehir Üniversitesi', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi-Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'GFN & Bahçeşehir Üniversitesi Wissen Akademie  ', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi | Wissen Akademi', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi/Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Meslek Yüksek Okulu/Bahçeşehir Üniversitesi', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi SEM Wissen Akademie', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi Silicon Valley Campus', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahçeşehir Üniversitesi Wissen Akademi', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Bahcesehir Üniversitesi', 'school_name'] = 'Bahcesehir University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül University', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi Sosyal Bilimler Enstitüsü', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi Biyotıp ve Genom Enstitüsü', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversite', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'İzmir Üniversitesi - Dokuz Eylül Üniversitesi ', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül University Faculty of Engineering Geophysical Engineering', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül University, İzmir', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi - iBG Izmir', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'İzmir Konak Dokuz Eylül Anadolu Lisesi', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi Özel 75. Yıl İ.Ö.O', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversites', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül University Business Administration', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi İMYO', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'İzmir Dokuz Eylül Anadolu Lisesi', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi Mühendislik Fakültesi', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül University Institute of Social Sciences', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'İzmir Dokuz Eylül Üniversitesi', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi / Dokuz Eylul University', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi Yabancı Diller Yüksekokulu', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul University ', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul University Izmir Vocational School', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul Univercity', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul ', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul Unıversıty', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul Univesity İzmir Vocational School', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul University,', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi / Dokuz Eylul University', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul University / Computer Engineering', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul University , Faculty of Engineering ,  Izmir', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul University - The Graduate School of Natural and Applied Sciences', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylul University - Faculty of Economics and Administrative Sciences', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Izmir Dokuz Eylul University', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eyl&#xfc;l &#xdc;niversitesi', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz eylül üniversitesi', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül University - Department of Mechanical Engineering', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül University - Department of Thermodynamics', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz EYlul University', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi - İzmir Meslek Yüksekokulu', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül University, Graduate School of Natural and Applied Sciences', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Dokuz Eylül Üniversitesi Sürekli Eğitim MErkezi', 'school_name'] = 'Dokuz Eylul University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi Ereğli Meslek Yüksekokulu', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Konya Selçuk Üniversitesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Konya Teknik Üniversitesi (Selçuk Üniversitesi)', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi Beyşehir MYO', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Konya Selçuk Üniversitesi ', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi Sivil Havacılık Yüksekokulu', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi Beyşehir Ali Akkanat Meslek Yüksekokulu', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi Huğlu M.Y.O', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi Eğitim Fakültesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi ( Vocational School of Higher Education )', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üniversitesi Teknik Bilimler Meslek Yüksekokulu', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk Üni', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selcuk University | Turkey', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selcuk Universitesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selcuk Universty', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == ' Selcuk University ', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk University', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk universıty', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Selçuk üniversitesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'selçuk üniversitesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'konya selçuk üniversitesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'bilgisayar mühendisliği selçuk üniversitesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Konya selcuk universitesi eregli meslek yuksek okulu', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'konya selcuk universitesi', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'selcuk university', 'school_name'] = 'Selcuk University'
    df_.loc[df_['school_name'] == 'Beykent University ', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'T.C. Beykent University', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'Postgraduate\tBeykent University', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'Beykent Universitesi', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'Beykent Univercity', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'Beykent Universty', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'Beykent Üniversitesi', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'İstanbul Beykent Üniversitesi Yönetim Bilşim Sistemleri Y.Lisans (MIS) Tezli', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'T.C. Beykent Üniversitesi', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'Beykent Üniversitesi Yazılım Mühendisliği', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'İstanbul Beykent Üniversitesi', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'Beykent üniversity', 'school_name'] = 'Beykent University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Fen Fakültesi', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi (Tömer) - 세종학당', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi ', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Geliştirme Vakfı Özel Okulları', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Geliştirme Vakfı Özel İlköğretim Okulu', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Siyasal Bilgiler Fakültesi', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi / EMYO', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Physics', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Bilgisayar Mühendisliği', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Bilgisayar Muhendisliği', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Avrupa Birliği Araştırma ve Uygulama Merkezi', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi, Avrupa Topluluğu Araştırma Uygulama Merkezi', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi/Y.Lisans', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Elektrik-Elektronik Mühendisliği', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Üniversitesi Mühendislik Fakültesi', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University, Ankara, Turkey', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Faculty of Law', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara Univercity', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Computer Education & Instructional Tecnologies', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Biotechnology Institute', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Engineering Faculty (Electronic Eng.)', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Institute of Science ', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Development Foundation Private Anatolian High School', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University, Institue of Nuclear Sciences', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Development Foundation Primary School', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Centre for Continuing Education', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University - Computer Programming', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University Computer Engineering', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'The Rectors Office Ankara Universitesi Rektorlugu', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Ankara University,  TÖMER', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'ankara university faculty of electronic engineer', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'ankara üniversitesi kastamonu m.y.o', 'school_name'] = 'Ankara University'
    df_.loc[df_['school_name'] == 'Trakya Üniversitesi', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Üniversitesi Teknik Bilimler MYO', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'T.C. Trakya Üniversitesi Keşan Yusuf Çapraz Uygulamalı Bilimler Yüksekokulu', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Üniversitesi Bilgisayar Mühendisliği', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Üniversitesi / Trakya University', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Üniversitesi - Computer Technology and Programming', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Üniversitesi vize meslek yüksek okulu pazarlama bölümü', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Üniversites', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Üniversitesi, Fen Bilimleri Enstitüsü', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya University (www.trakya.edu.tr)', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Universty', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya University, Edirne', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya University, Turkey', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya Üniversitesi / Trakya University', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya University, Faculty of Engineering and Architecture', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Trakya University - Faculty of Engineering', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Kirklareli / Trakya Universitesi', 'school_name'] = 'Trakya University'
    df_.loc[df_['school_name'] == 'Suleyman Demirel Universty', 'school_name'] = 'Suleyman Demirel University'
    df_.loc[df_['school_name'] == 'Suleyman Demirel Univercity', 'school_name'] = 'Suleyman Demirel University'
    df_.loc[df_['school_name'] == 'Suleyman Demirel Universirty', 'school_name'] = 'Suleyman Demirel University'
    df_.loc[df_['school_name'] == 'Suleyman Demirel University ', 'school_name'] = 'Suleyman Demirel University'
    df_.loc[df_['school_name'] == 'Süleyman Demirel Üniversitesi', 'school_name'] = 'Suleyman Demirel University'
    df_.loc[df_['school_name'] == 'Süleyman Demirel Üniversitesi Fenbilimleri Enstitüsü', 'school_name'] = 'Suleyman Demirel University'
    df_.loc[df_['school_name'] == 'Isparta Süleyman Demirel Üniversitesi', 'school_name'] = 'Suleyman Demirel University'
    df_.loc[df_['school_name'] == 'Süleyman Demirel Üniversitesi | Bilgisayar ve Kontrol Öğretmeni', 'school_name'] = 'Suleyman Demirel University'
    df_.loc[df_['school_name'] == 'Eskişehir Osmangazi Üniversitesi', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskişehir Osmangazi University', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskişehir Osmangazi Üniversitesi - Metallurgical Engineering', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskişehir Osmangazi Universty', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskişehir Osmangazi  University', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskişehir Osmangazi University, Faculty of Engineering & Architecture', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskisehir Osmangazi University', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskisehir Osmangazi University (Turkey)', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskisehir Osmangazi University GPA: 3,06', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Eskisehir Osmangazi Universitesi', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Osmangazi Universitesi', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Osmangazi Üniversitesi', 'school_name'] = 'Osmangazi University'
    df_.loc[df_['school_name'] == 'Karadeniz Teknik Üniversitesi', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == 'Karadeniz Technical Universty', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == 'Karadeniz Teknik Universitesi', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == 'Karadeniz Teknik', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == 'Karadeniz Technical University, Anadolu University', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == 'Karadeniz Teknik Üniversites', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == 'Karadeniz Teknik Üniversitesi,Bilgisayar Mühendisliği Bölümü', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == 'Karadeniz Teknik Üniversitesi / Karadeniz Technical University', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == ' Karadeniz Technical University', 'school_name'] = 'Karadeniz Technical University'
    df_.loc[df_['school_name'] == 'Boğaziçi Üniversitesi', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Boğaziçi Üniversitesi / Bogazici University', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Boğaziçi Üniversitesi / Bosphorus University', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Boğaziçi Üniversitesi-Kalder', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Limak Enerji & Boğaziçi Üniversitesi ', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Boğaziçi Üniversitesi Yaşam Boyu Eğitim Merkezi', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Boğaziçi Üniversitesi BÜEM', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Boğaziçi Üniversitesi Yabanci Diller Yuksek Okulu', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Bogazici University Department of Economics', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Bogazici University Lifelong Learning Center', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Üniversite\t\tBogaziçi Üniversitesi - Istanbul', 'school_name'] = 'Bogazici University'
    df_.loc[df_['school_name'] == 'Fırat Üniversitesi', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Elazığ Fırat Üniversitesi ', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Fırat Üniversitesi ', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Fırat Üniversitesi Su Ürünleri Fakültesi', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Fırat Üniversite', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Fırat Üniversitesi Teknoloji Fakültesi Yazılım Mühendisliği ', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Teknoloji Fakültesi/Fırat Üniversitesi', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Fırat University Mechatronics Engineer', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Fırat University', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Fırat University  Elazığ, Turkey', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Firat Universty', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Firat Universitesi', 'school_name'] = 'Firat University'
    df_.loc[df_['school_name'] == 'Bilkent Üniversitesi / Bilkent University', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == 'Bilkent University, Ankara, Industrial Engineering', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == 'Bilkent University Faculty of Business Administration', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == 'Bilkent University and Preparatory School, BUPS', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == 'Bilkent University, 2008', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == '2005 -2011 Bilkent University', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == 'Bilkent Üniversitesi', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == 'İhsan Doğramacı Bilkent Üniversitesi', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == 'Bilkent university', 'school_name'] = 'Bilkent University'
    df_.loc[df_['school_name'] == 'Erciyes Üniversitesi', 'school_name'] = 'Erciyes University'
    df_.loc[df_['school_name'] == 'Erciyes Üniversitesi ', 'school_name'] = 'Erciyes University'
    df_.loc[df_['school_name'] == 'Erciyes Üniversitesi Yabancı Diller Yüksekokulu (İngilizce)', 'school_name'] = 'Erciyes University'
    df_.loc[df_['school_name'] == 'Erciyes Universty', 'school_name'] = 'Erciyes University'
    df_.loc[df_['school_name'] == 'Kayseri Erciyes University', 'school_name'] = 'Erciyes University'
    df_.loc[df_['school_name'] == 'Erciyes', 'school_name'] = 'Erciyes University'
    df_.loc[df_['school_name'] == 'Erciyes Unıversity', 'school_name'] = 'Erciyes University'
    df_.loc[df_['school_name'] == 'University of Erciyes', 'school_name'] = 'Erciyes University'
    df_.loc[df_['school_name'] == 'Çukurova Üniversitesi myo', 'school_name'] = 'Cukurova University'
    df_.loc[df_['school_name'] == 'Çukurova Üniversty', 'school_name'] = 'Cukurova University'
    df_.loc[df_['school_name'] == 'Çukurova Üniversitesi Adana myo', 'school_name'] = 'Cukurova University'
    df_.loc[df_['school_name'] == 'Çukurova University', 'school_name'] = 'Cukurova University'
    df_.loc[df_['school_name'] == 'Çukurova Universty', 'school_name'] = 'Cukurova University'
    df_.loc[df_['school_name'] == 'Cukurova Universitesi', 'school_name'] = 'Cukurova University'
    df_.loc[df_['school_name'] == 'Cukurova Universty', 'school_name'] = 'Cukurova University'
    df_.loc[df_['school_name'] == 'Çukurova Üniversitesi', 'school_name'] = 'Cukurova University'
    df_.loc[df_['school_name'] == 'TOBB Ekonomi ve Teknoloji Üniversitesi', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB Ekonomi ve Teknoloji Universitesi', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB University of Economics &Technology', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB Economy and Technology University', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB Economics and Technology University', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB Ekonomi ve Teknoloji Universitesi', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB University of Economics & Technology', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB ETÜ, Electrical and Electronic Engineering', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB ETU - University of Economics & Technology', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB ETU University of Economics and Technology', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB Economics and Technolgy University', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'TOBB ETU', 'school_name'] = 'TOBB University of Economics and Technology'
    df_.loc[df_['school_name'] == 'Gebze Teknik Üniversitesi', 'school_name'] = 'Gebze Technical University'
    df_.loc[df_['school_name'] == 'Gebze Technical Universityy', 'school_name'] = 'Gebze Technical University'
    df_.loc[df_['school_name'] == 'Gebze Tecnical University', 'school_name'] = 'Gebze Technical University'
    df_.loc[df_['school_name'] == 'Gebze Yüksek Teknoloji Üniversitesi', 'school_name'] = 'Gebze Technical University'
    df_.loc[df_['school_name'] == 'Gebze Instude of Technology', 'school_name'] = 'Gebze Institute of Technology'
    df_.loc[df_['school_name'] == 'Gebze Enstitute of Techonology', 'school_name'] = 'Gebze Institute of Technology'
    df_.loc[df_['school_name'] == 'Gebze Institute Of Technology', 'school_name'] = 'Gebze Institute of Technology'
    df_.loc[df_['school_name'] == 'Gebze yuksek teknoloji enstitusu', 'school_name'] = 'Gebze Institute of Technology'
    df_.loc[df_['school_name'] == 'Gebze Yüksek Teknoloji Enstitüsü', 'school_name'] = 'Gebze Institute of Technology'
    df_.loc[df_['school_name'] == 'Istanbul Bilgi Üniversitesi', 'school_name'] = 'Istanbul Bilgi University'
    df_.loc[df_['school_name'] == 'Istanbul Bilgi', 'school_name'] = 'Istanbul Bilgi University'
    df_.loc[df_['school_name'] == 'İstanbul Bilgi Üniversitesi', 'school_name'] = 'Istanbul Bilgi University'
    df_.loc[df_['school_name'] == 'İstanbul Bilgi University', 'school_name'] = 'Istanbul Bilgi University'
    df_.loc[df_['school_name'] == 'İstanbul Bilgi Üniversitesi - Laureate International Universities', 'school_name'] = 'Istanbul Bilgi University'
    df_.loc[df_['school_name'] == 'İstanbul Bilgi Üniversity', 'school_name'] = 'Istanbul Bilgi University'
    df_.loc[df_['school_name'] == 'İstanbul Bilgi Üniversitesi, Tasarim Kulturu ve Yonetimi', 'school_name'] = 'Istanbul Bilgi University'
    df_.loc[df_['school_name'] == 'Yeditepe Üniversitesi', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Yeditepe Üniversitesi / Yeditepe University', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Yeditepe university', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Yeditepe University (Department of Computer Engineering', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Yeditepe Üniversitesi ', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Yeditepe University English Preparatory School', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Yeditepe Ü', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Yeditepe Univercity', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == '2005\tYeditepe University', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Yeditepe Üniversitesi Mimarlık Fakültesi ', 'school_name'] = 'Yeditepe University'
    df_.loc[df_['school_name'] == 'Orta Doğu Teknik Üniversitesi / Middle East Technical University', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Orta Doğu Teknik Üniversitesi', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Orta Dogu Teknik Üniversitesi', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Türkiye ve Orta Doğu Amme İdaresi Enstitüsü (TODAİE)', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Orta Doğu Teknik Üniversitesi ', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Orta Doğu Teknik Üniversitesi Kuzey Kıbrıs Kampüsü', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Orta Doğu Teknik Üniversitesi / Middle East Technical University English Prep. School', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Middle East Technical University Northern Cyprus Campus', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'TODAİE, Institute of Public Administration for Turkey and Middle East', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Private Middle East College', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Middle East Technical University Development Foundation High School (METU Collage)', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'METU-Middle East Technical University (Turkey)', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Middle East Techincal University', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Middle East Technical University (METU)', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == ' Middle East Technical University', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'METU', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'ODTÜ (METU)', 'school_name'] = 'Middle East Technical University'
    df_.loc[df_['school_name'] == 'Akdeniz Üniversitesi', 'school_name'] = 'Mediterranean University'
    df_.loc[df_['school_name'] == 'Doğu Akdeniz Üniversitesi', 'school_name'] = 'Eastern Mediterranean University'
    df_.loc[df_['school_name'] == 'Doğu Akdeniz Üniversitesi / Eastern Mediterranean University', 'school_name'] = 'Eastern Mediterranean University'
    df_.loc[df_['school_name'] == 'Akdeniz Üniversitesi Teknik Bilimler Myo', 'school_name'] = 'Mediterranean University'
    df_.loc[df_['school_name'] == 'Akdeniz Üniversitesi ', 'school_name'] = 'Mediterranean University'
    df_.loc[df_['school_name'] == 'Adana Özel Yeni Lise, Doğu Akdeniz Üniversitesi', 'school_name'] = 'Adana Özel Yeni Lise, Eastern Mediterranean University'
    df_.loc[df_['school_name'] == 'Akdeniz Üniversitesi - Computer Engineering', 'school_name'] = 'Mediterranean University'
    df_.loc[df_['school_name'] == 'Akdeniz Üniversitesi kamu yönetimi', 'school_name'] = 'Mediterranean University'
    df_.loc[df_['school_name'] == 'University 1\t\tAkdeniz University  - Akseki Vocational Schools /Antalya', 'school_name'] = 'Mediterranean University'
    df_.loc[df_['school_name'] == 'Akdeniz University Computer Engineering ', 'school_name'] = 'Mediterranean University'
    df_.loc[df_['school_name'] == 'Akdeniz Universtity', 'school_name'] = 'Mediterranean University'
    df_.loc[df_['school_name'] == 'Mugla Sıtkı Kocman University', 'school_name'] = 'Mugla Sitki Kocman University'
    df_.loc[df_['school_name'] == 'Muğla Sıtkı Koçman Üniversitesi', 'school_name'] = 'Mugla Sitki Kocman University'
    df_.loc[df_['school_name'] == 'Muğla Sıtkı Koçman University', 'school_name'] = 'Mugla Sitki Kocman University'
    df_.loc[df_['school_name'] == 'Muğla Sıtkı Koçman Üniversitesi English Preparatory Year', 'school_name'] = 'Mugla Sitki Kocman University'
    df_.loc[df_['school_name'] == 'Muğla Sıktı Koçman University', 'school_name'] = 'Mugla Sitki Kocman University'
    df_.loc[df_['school_name'] == 'School of Foreign Languages-Muğla Sıtkı Koçman University', 'school_name'] = 'Mugla Sitki Kocman University'
    df_.loc[df_['school_name'] == 'Muğla Sıtkı Koçman University School of Foreign Languages', 'school_name'] = 'Mugla Sitki Kocman University'
    df_.loc[df_['school_name'] == 'Muğla Science High School', 'school_name'] = 'Mugla Sitki Kocman University'
    df_.loc[df_['school_name'] == 'Koç University', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == 'Koç University, College of Engineering, Istanbul', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == '2003-2008 Fall \t\tKoç University', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == 'Koç University, College of Engineering', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == 'Koç University - College of Administrative Science and Economics', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == 'English Language Center, Koç University', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == 'Koç Univeristy', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == 'Koç Üniversitesi', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == 'Koç Üni', 'school_name'] = 'Koc University'
    df_.loc[df_['school_name'] == 'Sabancı University', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == 'Sabancı Universitesi', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == 'Sabancı Univeristy', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == 'Sabancı Üniversitesi', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == 'Sabancı Üniversitesi / Sabanci University', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == 'Sabancı Üniversitesi, Istanbul', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == 'Sabanci University, Istanbul', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == '2005 - 2010           Sabanci University, Istanbul', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == 'Sabanci University Summer School', 'school_name'] = 'Sabanci University'
    df_.loc[df_['school_name'] == 'Galatasaray Üniversitesi', 'school_name'] = 'Galatasaray University'
    df_.loc[df_['school_name'] == 'GalaGalatasaray Üniversitesi', 'school_name'] = 'Galatasaray University'
    df_.loc[df_['school_name'] == 'Ahmet Yesevi Üniversitesi', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Hoca Ahmet Yesevi Üniversitesi', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'H. Ahmet Yesevi Üniversitesi', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Ahmet Yesevi Üniversitesi ', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == "Ahmet Yesevi Üniversitesi(master's degree)", 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Ahmet Yesevi Üniversitesi Mühendislik Fakültesi', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Ahmet Yesevi Üniversitesi - Uzaktan Eğitim', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Ahmet Yesevi Üniversity', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Ahmet Yesevi Universitesi', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Ahmet Yesevi Universitesi Yuksek Lisans', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Hoca Ahmet Yesevi University', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Ahmet Yesevi Universty', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Hoca Ahmet Yesevi Universty', 'school_name'] = 'Ahmet Yesevi University'
    df_.loc[df_['school_name'] == 'Atatürk Üniversitesi', 'school_name'] = 'Ataturk University'
    df_.loc[df_['school_name'] == 'Atatürk Üniversitesi Resmi', 'school_name'] = 'Ataturk University'
    df_.loc[df_['school_name'] == 'Erzurum Atatürk Üniversitesi', 'school_name'] = 'Ataturk University'
    df_.loc[df_['school_name'] == 'Erzurum Atatürk Üniversitesi Açıköğretim Fakültesi', 'school_name'] = 'Ataturk University'
    df_.loc[df_['school_name'] == 'Atatürk University', 'school_name'] = 'Ataturk University'
    df_.loc[df_['school_name'] == 'Ataturk University ( İkinci Universite )', 'school_name'] = 'Ataturk University'
    df_.loc[df_['school_name'] == 'Uluslararasi Ataturk Universitesi', 'school_name'] = 'Ataturk University'
    df_.loc[df_['school_name'] == 'Ataturk Universitesi', 'school_name'] = 'Ataturk University'
    df_.loc[df_['school_name'] == 'Eskişehir Teknik Üniversitesi', 'school_name'] = 'Eskisehir Technical University'
    df_.loc[df_['school_name'] == 'Eskişehir Technical University', 'school_name'] = 'Eskisehir Technical University'
    df_.loc[df_['school_name'] == 'Eskişehir Teknik Üniversitesi ', 'school_name'] = 'Eskisehir Technical University'
    df_.loc[df_['school_name'] == 'Eskişehir Teknik University', 'school_name'] = 'Eskisehir Technical University'
    df_.loc[df_['school_name'] == 'Baskent Üniversitesi', 'school_name'] = 'Baskent University'
    df_.loc[df_['school_name'] == 'Başkent Üniversitesi', 'school_name'] = 'Baskent University'
    df_.loc[df_['school_name'] == 'Başkent University', 'school_name'] = 'Baskent University'
    df_.loc[df_['school_name'] == 'Başkent Üniversitesi, Sosyal Bilimler Meslek Yüksekokulu', 'school_name'] = 'Baskent University'
    df_.loc[df_['school_name'] == 'Başkent Üniversitesi Özel Ayşeabla Okulları Fen Lisesi', 'school_name'] = 'Baskent University Özel Ayşeabla Okulları Fen Lisesi'
    df_.loc[df_['school_name'] == 'Başkent Üniversitesi Kolej Ayşeabla', 'school_name'] = 'Baskent University'
    df_.loc[df_['school_name'] == 'Başkent ', 'school_name'] = 'Baskent University'
    df_.loc[df_['school_name'] == 'Uludağ Üniversitesi', 'school_name'] = 'Uludag University'
    df_.loc[df_['school_name'] == 'Uludağ University', 'school_name'] = 'Uludag University'
    df_.loc[df_['school_name'] == 'Uludağ Üniversitesi Bilgisayar Programcılığı', 'school_name'] = 'Uludag University'
    df_.loc[df_['school_name'] == 'Uludağ Üniversitesi Teknik Bilimler Meslek Yüksek Okulu', 'school_name'] = 'Uludag University'
    df_.loc[df_['school_name'] == 'Uludag Universitesi (Lisans)', 'school_name'] = 'Uludag University'
    df_.loc[df_['school_name'] == 'Atılım Üniversitesi', 'school_name'] = 'Atilim University'
    df_.loc[df_['school_name'] == 'Celal Bayar Üniversitesi', 'school_name'] = 'Celal Bayar University'
    df_.loc[df_['school_name'] == 'Celal Bayar Üniversitesi Kırkağaç Meslek Yüksekokulu ', 'school_name'] = 'Celal Bayar University'
    df_.loc[df_['school_name'] == 'Manisa Celal Bayar University', 'school_name'] = 'Celal Bayar University'
    df_.loc[df_['school_name'] == 'Pamukkale Üniversitesi', 'school_name'] = 'Pamukkale University'
    df_.loc[df_['school_name'] == 'İstanbul Commerce Univercity', 'school_name'] = 'Istanbul Commerce University'
    df_.loc[df_['school_name'] == 'İstanbul Ticaret Üniversitesi', 'school_name'] = 'Istanbul Commerce University'
    df_.loc[df_['school_name'] == 'Çankaya Üniversitesi', 'school_name'] = 'Cankaya University'
    df_.loc[df_['school_name'] == 'Çankaya University', 'school_name'] = 'Cankaya University'
    df_.loc[df_['school_name'] == 'Maltepe Üniversitesi', 'school_name'] = 'Maltepe University'
    df_.loc[df_['school_name'] == 'Maltepe Üniversitesi Yazılım Mühendisliği', 'school_name'] = 'Maltepe University'
    df_.loc[df_['school_name'] == 'T.C. Maltepe Üniversitesi', 'school_name'] = 'Maltepe University'
    df_.loc[df_['school_name'] == 'Ondokuz Mayıs Üniversitesi', 'school_name'] = 'Ondokuz Mayis University'
    df_.loc[df_['school_name'] == 'Samsun Ondokuz Mayıs Anadolu Lisesi', 'school_name'] = 'Ondokuz Mayis University'
    df_.loc[df_['school_name'] == 'Ondokuz Mayıs Üniversitesi', 'school_name'] = 'Ondokuz Mayis University'
    df_.loc[df_['school_name'] == 'Ondokuz Mayıs University', 'school_name'] = 'Ondokuz Mayis University'
    df_.loc[df_['school_name'] == 'Samsun Ondokuz Mayıs Anadolu Lisesi', 'school_name'] = 'Ondokuz Mayis University'
    df_.loc[df_['school_name'] == 'Samsun Ondokuz Mayis University', 'school_name'] = 'Ondokuz Mayis University'
    df_.loc[df_['school_name'] == 'Kirikkale Üniversitesi', 'school_name'] = 'Kirikkale University'
    df_.loc[df_['school_name'] == 'Kırıkkale Üniversitesi', 'school_name'] = 'Kirikkale University'
    df_.loc[df_['school_name'] == 'Kırıkkale University', 'school_name'] = 'Kirikkale University'
    df_.loc[df_['school_name'] == 'Istanbul Kültür University', 'school_name'] = 'Istanbul Kultur University'
    df_.loc[df_['school_name'] == 'Istanbul Kültür Üniversitesi', 'school_name'] = 'Istanbul Kultur University'
    df_.loc[df_['school_name'] == 'İstanbul Kültür Üniversitesi', 'school_name'] = 'Istanbul Kultur University'
    df_.loc[df_['school_name'] == 'İzmir Ekonomi Üniversitesi', 'school_name'] = 'Izmir University of Economics'
    df_.loc[df_['school_name'] == 'Izmir Ekonomi Universitesi', 'school_name'] = 'Izmir University of Economics'
    df_.loc[df_['school_name'] == 'İzmir University of Economics', 'school_name'] = 'Izmir University of Economics'
    df_.loc[df_['school_name'] == 'Izmir University of Economics Graduate School of Social Sciences', 'school_name'] = 'Izmir University of Economics'
    df_.loc[df_['school_name'] == 'Gaziantep Üniversitesi', 'school_name'] = 'Gaziantep University'
    df_.loc[df_['school_name'] == 'Gaziantep Üniversitesi MYO', 'school_name'] = 'Gaziantep University'
    df_.loc[df_['school_name'] == 'Mersin Üniversitesi', 'school_name'] = 'Mersin University'
    df_.loc[df_['school_name'] == 'Mersin Üniversitesi / Mersin University', 'school_name'] = 'Mersin University'
    df_.loc[df_['school_name'] == 'Işık Üniversitesi', 'school_name'] = 'Isik University'
    df_.loc[df_['school_name'] == 'FMV Işık Üniversitesi', 'school_name'] = 'Isik University'
    df_.loc[df_['school_name'] == 'Işık Üniversitesi / Işık University (Feyziye Mektepleri Vakfı, 1885 / Feyziye Schools Foundation, 1885)', 'school_name'] = 'Isik University'
    df_.loc[df_['school_name'] == 'Isik Üniversitesi', 'school_name'] = 'Isik University'
    df_.loc[df_['school_name'] == 'Doğuş Üniversitesi', 'school_name'] = 'Dogus University'
    df_.loc[df_['school_name'] == 'Doğuş University', 'school_name'] = 'Dogus University'
    df_.loc[df_['school_name'] == 'Dogus Üniversitesi', 'school_name'] = 'Dogus University'
    df_.loc[df_['school_name'] == 'Kadir Has Üniversitesi', 'school_name'] = 'Kadir Has University'
    df_.loc[df_['school_name'] == 'Okan Üniversitesi', 'school_name'] = 'Okan University'
    df_.loc[df_['school_name'] == 'Abant İzzet Baysal Üniversitesi', 'school_name'] = 'Abant Izzet Baysal University'
    df_.loc[df_['school_name'] == 'Bolu Abant İzzet Baysal University', 'school_name'] = 'Abant Izzet Baysal University'
    df_.loc[df_['school_name'] == 'Abant İzzet Baysal University', 'school_name'] = 'Abant Izzet Baysal University'
    df_.loc[df_['school_name'] == 'Abant İzzet Baysal Üniversitesi / Abant Izzet Baysal University', 'school_name'] = 'Abant Izzet Baysal University'
    df_.loc[df_['school_name'] == 'Afyon Kocatepe Üniversitesi', 'school_name'] = 'Afyon Kocatepe University'
    df_.loc[df_['school_name'] == 'Afyon Kocatepe Üniversitesi Fen Bilimleri Enstitüsü', 'school_name'] = 'Afyon Kocatepe University'
    df_.loc[df_['school_name'] == 'Afyon Kocatepe Üniversitesi Bolvadin Meslek Yüksek Okulu', 'school_name'] = 'Afyon Kocatepe University'
    df_.loc[df_['school_name'] == 'İnönü Üniversitesi', 'school_name'] = 'Inonu University'
    df_.loc[df_['school_name'] == 'Malatya İnönü Üniversitesi', 'school_name'] = 'Inonu University'
    df_.loc[df_['school_name'] == 'İnönü University', 'school_name'] = 'Inonu University'
    df_.loc[df_['school_name'] == 'Inönü University', 'school_name'] = 'Inonu University'
    df_.loc[df_['school_name'] == 'Fatih Üniversitesi', 'school_name'] = 'Fatih University'
    df_.loc[df_['school_name'] == 'Çanakkale Onsekiz Mart Üniversitesi', 'school_name'] = 'Canakkale Onsekiz Mart University'
    df_.loc[df_['school_name'] == 'Çanakkale Onsekiz Mart University', 'school_name'] = 'Canakkale Onsekiz Mart University'
    df_.loc[df_['school_name'] == 'Halic Universitesi', 'school_name'] = 'Halic University'
    df_.loc[df_['school_name'] == 'Haliç Üniversitesi', 'school_name'] = 'Halic University'
    df_.loc[df_['school_name'] == ' Haliç Üniversitesi', 'school_name'] = 'Halic University'
    df_.loc[df_['school_name'] == 'Balikesir Üniversitesi', 'school_name'] = 'Balikesir University'
    df_.loc[df_['school_name'] == 'Balıkesir Üniversitesi', 'school_name'] = 'Balikesir University'
    df_.loc[df_['school_name'] == 'Balıkesir University', 'school_name'] = 'Balikesir University'
    df_.loc[df_['school_name'] == 'Mugla Üniversitesi', 'school_name'] = 'Mugla University'
    df_.loc[df_['school_name'] == 'Muğla Üniversitesi', 'school_name'] = 'Mugla University'
    df_.loc[df_['school_name'] == 'Muğla University', 'school_name'] = 'Mugla University'
    df_.loc[df_['school_name'] == 'Ankara Yıldırım Beyazıt Üniversitesi', 'school_name'] = 'Ankara Yildirim Beyazit University'
    df_.loc[df_['school_name'] == 'Ankara Yıldırım Beyazıt University', 'school_name'] = 'Ankara Yildirim Beyazit University'
    df_.loc[df_['school_name'] == 'Ankara Yıldırım Beyazit University', 'school_name'] = 'Ankara Yildirim Beyazit University'
    df_.loc[df_['school_name'] == 'Yaşar Üniversitesi', 'school_name'] = 'Yasar University'
    df_.loc[df_['school_name'] == 'Yaşar Üniversitesi (Yaşar University)', 'school_name'] = 'Yasar University'
    df_.loc[df_['school_name'] == 'Yaşar University', 'school_name'] = 'Yasar University'
    df_.loc[df_['school_name'] == 'Altınbaş Üniversitesi', 'school_name'] = 'Altinbas University'
    df_.loc[df_['school_name'] == 'Nişantaşı Üniversitesi', 'school_name'] = 'Nisantasi University'
    df_.loc[df_['school_name'] == 'Nişantaşı University', 'school_name'] = 'Nisantasi University'
    df_.loc[df_['school_name'] == 'Nişantaşı üniversitesi', 'school_name'] = 'Nisantasi University'
    df_.loc[df_['school_name'] == 'Nişantaşı Üniversitesi ', 'school_name'] = 'Nisantasi University'
    df_.loc[df_['school_name'] == 'Nişantaşı Ünversitesi', 'school_name'] = 'Nisantasi University'
    df_.loc[df_['school_name'] == 'Dumlupınar Üniversitesi', 'school_name'] = 'Dumlupinar University'
    df_.loc[df_['school_name'] == 'Kütahya Dumlupınar Üniversitesi', 'school_name'] = 'Dumlupinar University'
    df_.loc[df_['school_name'] == 'Dumlupınar University', 'school_name'] = 'Dumlupinar University'
    df_.loc[df_['school_name'] == 'College\tDumlupinar University - Kütahya', 'school_name'] = 'Dumlupinar University'
    df_.loc[df_['school_name'] == 'Dumlupinar Üniversitesi', 'school_name'] = 'Dumlupinar University'
    df_.loc[df_['school_name'] == 'Kütahya Dumlupınar University', 'school_name'] = 'Dumlupinar University'
    df_.loc[df_['school_name'] == 'Kutahya Dumlupinar University-Turkey', 'school_name'] = 'Dumlupinar University'
    df_.loc[df_['school_name'] == 'Konya Teknik Üniversitesi', 'school_name'] = 'Konya Technical University'
    df_.loc[df_['school_name'] == 'Konya Teknik Üniversitesi ', 'school_name'] = 'Konya Technical University'
    df_.loc[df_['school_name'] == 'Konya Teknik Universitesi', 'school_name'] = 'Konya Technical University'
    df_.loc[df_['school_name'] == 'İstanbul Gelişim Üniversitesi', 'school_name'] = 'Istanbul Gelisim University'
    df_.loc[df_['school_name'] == 'Izmir Katip Celebi University, Izmir, Turkey', 'school_name'] = 'Izmir Katip Celebi University'
    df_.loc[df_['school_name'] == 'Izmir Katip Celebi Üniversitesi', 'school_name'] = 'Izmir Katip Celebi University'
    df_.loc[df_['school_name'] == 'Izmir Katip Çelebi University', 'school_name'] = 'Izmir Katip Celebi University'
    df_.loc[df_['school_name'] == 'İzmir Katip Çelebi Üniversitesi', 'school_name'] = 'Izmir Katip Celebi University'
    df_.loc[df_['school_name'] == 'İzmir Katip Çelebi University', 'school_name'] = 'Izmir Katip Celebi University'
    df_.loc[df_['school_name'] == 'İstanbul Aydın Universitesi', 'school_name'] = 'Istanbul Aydin University'
    df_.loc[df_['school_name'] == 'Namık Kemal Üniversitesi', 'school_name'] = 'Namik Kemal University'
    df_.loc[df_['school_name'] == 'Namık Kemal Üniversitesi', 'school_name'] = 'Namik Kemal University'
    df_.loc[df_['school_name'] == 'Namık Kemal University', 'school_name'] = 'Namik Kemal University'
    df_.loc[df_['school_name'] == 'Tekirdağ Namık Kemal Üniversitesi', 'school_name'] = 'Namik Kemal University'
    df_.loc[df_['school_name'] == 'University of Namik Kemal', 'school_name'] = 'Namik Kemal University'
    df_.loc[df_["school_name"] == "Cumhuriyet Üniversitesi", "school_name"] = "Cumhuriyet University"
    df_.loc[df_["school_name"] == "Dicle Üniversitesi", "school_name"] = "Dicle University"
    df_.loc[df_["school_name"] == "Bursa Teknik Üniversitesi", "school_name"] = "Bursa Technical University"
    df_.loc[df_["school_name"] == "İstanbul Sabahattin Zaim Üniversitesi", "school_name"] = "Istanbul Sabahattin Zaim University"
    df_.loc[df_["school_name"] == "İstanbul Medeniyet Üniversitesi", "school_name"] = "Istanbul Medeniyet University"
    df_.loc[df_["school_name"] == "Üsküdar Üniversitesi", "school_name"] = "Uskudar University"
    df_.loc[df_["school_name"] == "Mimar Sinan Güzel Sanatlar Üniversitesi", "school_name"] = "Mimar Sinan Fine Arts University"
    df_.loc[df_["school_name"] == "Türk Hava Kurumu Üniversitesi", "school_name"] = "Turkish Aeronautical Association University"
    df_.loc[df_["school_name"] == "Düzce Üniversitesi", "school_name"] = "Duzce University"
    df_.loc[df_["school_name"] == "Mustafa Kemal Üniversitesi", "school_name"] = "Mustafa Kemal University"
    df_.loc[df_["school_name"] == "İzmir Üniversitesi", "school_name"] = "Izmir University"
    df_.loc[df_["school_name"] == "Ufuk Üniversitesi", "school_name"] = "Ufuk University"
    df_.loc[df_["school_name"] == "Bogaziçi Üniversitesi", "school_name"] = "Bogazici University"
    df_.loc[df_["school_name"] == "Netkent Akdeniz Araştırma ve Bilim Üniversitesi", "school_name"] = "Netkent Mediterranean Research And Science University"
    df_.loc[df_["school_name"] == "Medipol Üniversitesi", "school_name"] = "Medipol University"
    df_.loc[df_["school_name"] == "Yakın Doğu Üniversitesi", "school_name"] = "Near East University"
    df_.loc[df_["school_name"] == "TED Üniversitesi", "school_name"] = "Ted University"
    df_.loc[df_["school_name"] == "Yalova Üniversitesi", "school_name"] = "Yalova University"
    df_.loc[df_["school_name"] == "Karabük Üniversitesi", "school_name"] = "Karabuk University"
    df_.loc[df_["school_name"] == "İstanbul Aydın Üniversitesi", "school_name"] = "Istanbul Aydin University"
    df_.loc[df_["school_name"] == "Adnan Menderes Üniversitesi", "school_name"] = "Adnan Menderes University"
    df_.loc[df_["school_name"] == "9 Eylül Üniversitesi", "school_name"] = "9 Eylul University"
    df_.loc[df_["school_name"] == "Kahramanmaraş Sütçüimam Üniversitesi", "school_name"] = "Kahramanmaras Sutcuimam University"
    df_.loc[df_["school_name"] == "Mehmet Akif Ersoy Üniversitesi", "school_name"] = "Mehmet Akif Ersoy University"
    df_.loc[df_["school_name"] == "Uluslararası Kıbrıs Üniversitesi", "school_name"] = "Cyprus International University"
    df_.loc[df_["school_name"] == "Toros Üniversitesi", "school_name"] = "Toros University"
    df_.loc[df_["school_name"] == "İzmir Bakırçay Üniversitesi", "school_name"] = "Izmir Bakircay University"
    df_.loc[df_["school_name"] == "Kastamonu Üniversitesi", "school_name"] = "Kastamonu University"
    df_.loc[df_["school_name"] == "Erzurum Teknik Üniversitesi", "school_name"] = "Erzurum Technical University"
    df_.loc[df_["school_name"] == "Giresun Üniversitesi", "school_name"] = "Giresun University"
    df_.loc[df_["school_name"] == "Harran Üniversitesi", "school_name"] = "Harran University"
    df_.loc[df_["school_name"] == "Sabanci Üniversitesi", "school_name"] = "Sabanci University"
    df_.loc[df_["school_name"] == "Özyeğin Üniversitesi", "school_name"] = "Ozyegin University"
    df_.loc[df_["school_name"] == "Bahçesehir Üniversitesi", "school_name"] = "Bahcesehir University"
    df_.loc[df_["school_name"] == "İstinye Üniversitesi", "school_name"] = "Istinye University"
    df_.loc[df_["school_name"] == "Kırklareli Üniversitesi", "school_name"] = "Kirklareli University"
    df_.loc[df_["school_name"] == "İstanbul Arel Üniversitesi", "school_name"] = "Istanbul Arel University"
    df_.loc[df_["school_name"] == "Necmettin Erbakan Üniversitesi", "school_name"] = "Necmettin Erbakan University"
    df_.loc[df_["school_name"] == "İstanbul Esenyurt Üniversitesi", "school_name"] = "Istanbul Esenyurt University"
    df_.loc[df_["school_name"] == "Samsun Üniversitesi", "school_name"] = "Samsun University"
    df_.loc[df_["school_name"] == "Turgut Özal Üniversitesi", "school_name"] = "Turgut Ozal University"
    df_.loc[df_["school_name"] == "İskenderun Teknik Üniversitesi", "school_name"] = "Iskenderun Technical University"
    df_.loc[df_["school_name"] == "İstanbul Medipol Üniversitesi", "school_name"] = "Istanbul Medipol University"
    df_.loc[df_["school_name"] == "Türk-Alman Üniversitesi", "school_name"] = "Turkish-German University"
    df_.loc[df_["school_name"] == "Lefke Avrupa Üniversitesi", "school_name"] = "European University Of Lefke"
    df_.loc[df_["school_name"] == "Necmettin Erbakan Üniversitesi ", "school_name"] = "Necmettin Erbakan University"
    df_.loc[df_["school_name"] == "Ortadoğu Teknik Üniversitesi ", "school_name"] = "Middle East Technical University"
    df_.loc[df_["school_name"] == "Gedik Üniversitesi", "school_name"] = "Gedik University"
    df_.loc[df_["school_name"] == "selcuk üniversitesi", "school_name"] = "Selcuk University"
    df_.loc[df_["school_name"] == "Kırgızistan türkiye manas üniversitesi ", "school_name"] = "Kyrgyzstan Turkey Manas University"
    df_.loc[df_["school_name"] == "istanbul üniversitesi", "school_name"] = "Istanbul University"
    df_.loc[df_["school_name"] == "Düzce üniversitesi", "school_name"] = "Duzce University"
    df_.loc[df_["school_name"] == "Firat üniversitesi", "school_name"] = "Firat University"
    df_.loc[df_["school_name"] == "nişantaşı üniversitesi", "school_name"] = "Nisantasi University"
    df_.loc[df_["school_name"] == "Karabük üniversitesi", "school_name"] = "Karabuk University"
    df_.loc[df_["school_name"] == "düzce üniversitesi", "school_name"] = "Duzce University"
    df_.loc[df_["school_name"] == "Ahmet Yesevi üniversitesi", "school_name"] = "Ahmet Yesevi University"
    df_.loc[df_["school_name"] == "Bahçeşehir üniversitesi", "school_name"] = "Bahcesehir University"
    df_.loc[df_["school_name"] == "İstanbul esenyurt üniversitesi ", "school_name"] = "Istanbul Esenyurt University"
    df_.loc[df_["school_name"] == "Necmettin Erbakan üniversitesi", "school_name"] = "Necmettin Erbakan University"
    df_.loc[df_["school_name"] == "Biruni Üniversitesi", "school_name"] = "Biruni University"
    df_.loc[df_["school_name"] == "Gediz Üniversitesi", "school_name"] = "Gediz University"
    df_.loc[df_["school_name"] == "Adıyaman Üniversitesi", "school_name"] = "Adiyaman University"

    ################################################################################################################

    df_.loc[df_['school_name'] == 'İzmir Atatürk Lisesi', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'İzmir Atatürk High School', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'İzmir Atatürk Anadolu Lisesi', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'İzmir Atatürk Anadolu Teknik Lisesi', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'İzmir Atatürk Highschool', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'İzmir Atatürk Lisesi,', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'Izmir Ataturk Lisesi', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'Izmir Atatürk Lisesi', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'Izmir Atatürk High School', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == '2003 \xad 2007 Izmir Atatürk High School', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'İzmir Ataturk Lisesi', 'school_name'] = 'Izmir Ataturk High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Anadolu Lisesi', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Lisesi', 'school_name'] = 'Ankara Ataturk High School'
    df_.loc[df_['school_name'] == 'Ankara Ataturk Anadolu Lisesi', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Anatolian High School', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Ataturk Anatolian High School', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Ataturk Anadolu High School', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Lisesi ', 'school_name'] = 'Ankara Ataturk High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Anatolian Highschool', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk High school', 'school_name'] = 'Ankara Ataturk High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Anadolu Lisesi (AAAL)', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Anadolu High School', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatük Lisesi', 'school_name'] = 'Ankara Ataturk High School'
    df_.loc[df_['school_name'] == 'Ankara Ataturk Lisesi', 'school_name'] = 'Ankara Ataturk High School'
    df_.loc[df_['school_name'] == 'Çankaya Ankara Atatürk Anatolian High Scool', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk High School ', 'school_name'] = 'Ankara Ataturk High School'
    df_.loc[df_['school_name'] == 'Ankara Ataturk Anatolian Highschool', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Anadolu Lisesi - AAAL', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Atatürk Anadolu Lİsesi', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_['school_name'] == 'Ankara Ataturk Anatolian High School ', 'school_name'] = 'Ankara Ataturk Anatolian High School'
    df_.loc[df_["school_name"] == "Bornova Anadolu Lisesi", "school_name"] = "Bornova Anatolian High School"
    df_.loc[df_["school_name"] == "Pertevniyal Anadolu Lisesi", "school_name"] = "Pertevniyal Anatolian High School"
    df_.loc[df_["school_name"] == "Beşiktaş Atatürk Anadolu Lisesi", "school_name"] = "Besiktas Ataturk Anatolian High School"
    df_.loc[df_["school_name"] == "Burak Bora Anadolu Lisesi", "school_name"] = "Burak Bora Anatolian High School"
    df_.loc[df_["school_name"] == "Haydarpaşa Lisesi", "school_name"] = "Haydarpasa High School"
    df_.loc[df_["school_name"] == "Kadıköy Anadolu Lisesi", "school_name"] = "Kadıköy Anatolian High School"
    df_.loc[df_["school_name"] == "Karşıyaka Anadolu Lisesi", "school_name"] = "Karsiyaka Anatolian High School"
    df_.loc[df_["school_name"] == "Bursa Anadolu Lisesi", "school_name"] = "Bursa Anatolian High School"
    df_.loc[df_["school_name"] == "Galatasaray Lisesi", "school_name"] = "Galatasaray High School"
    df_.loc[df_["school_name"] == "Denizli Anadolu Lisesi", "school_name"] = "Denizli Anatolian High School"
    df_.loc[df_["school_name"] == "Şehremini Anadolu Lisesi", "school_name"] = "Şehremini Anatolian High School"
    df_.loc[df_["school_name"] == "Adana Anadolu Lisesi", "school_name"] = "Adana Anatolian High School"
    df_.loc[df_["school_name"] == "İstanbul Atatürk Fen Lisesi", "school_name"] = "Istanbul Ataturk Science High School"
    df_.loc[df_["school_name"] == "Istanbul Cagaloglu Anadolu Lisesi", "school_name"] = "Istanbul Cagaloglu Anatolian High School"
    df_.loc[df_["school_name"] == "Bahçelievler Anadolu Lisesi", "school_name"] = "Bahcelievler Anatolian High School"
    df_.loc[df_["school_name"] == "Samsun Anadolu Lisesi", "school_name"] = "Samsun Anatolian High School"
    df_.loc[df_["school_name"] == "Gazi Anadolu Lisesi", "school_name"] = "Gazi Anatolian High School"
    df_.loc[df_["school_name"] == "Antalya Anadolu Lisesi", "school_name"] = "Antalya Anatolian High School"
    df_.loc[df_["school_name"] == "Pertevniyal Lisesi", "school_name"] = "Pertevniyal High School"
    df_.loc[df_["school_name"] == "Kartal Anadolu Lisesi", "school_name"] = "Kartal Anatolian High School"
    df_.loc[df_["school_name"] == "Kayseri Fen Lisesi", "school_name"] = "Kayseri Science High School"
    df_.loc[df_["school_name"] == "Tekirdağ Anadolu Lisesi", "school_name"] = "Tekirdag Anatolian High School"
    df_.loc[df_["school_name"] == "Kocaeli Anadolu Lisesi", "school_name"] = "Kocaeli Anatolian High School"
    df_.loc[df_["school_name"] == "Vefa Lisesi", "school_name"] = "Vefa High School"
    df_.loc[df_["school_name"] == "Ankara Fen Lisesi", "school_name"] = "Ankara Science High School"
    df_.loc[df_["school_name"] == "Hüseyin Avni Sözen Anadolu Lisesi", "school_name"] = "Hüseyin Avni Sözen Anatolian High School"
    df_.loc[df_["school_name"] == "Çağrıbey Anadolu Lisesi", "school_name"] = "Cagribey Anatolian High School"
    df_.loc[df_["school_name"] == "Adnan Menderes Anadolu Lisesi", "school_name"] = "Adnan Menderes Anatolian High School"
    df_.loc[df_["school_name"] == "Seyhan Rotary Anadolu Lisesi", "school_name"] = "Seyhan Rotary Anatolian High School"
    df_.loc[df_["school_name"] == "Florya Tevfik Ercan Anadolu Lisesi", "school_name"] = "Florya Tevfik Ercan Anatolian High School"
    df_.loc[df_["school_name"] == "İzmir Fen Lisesi", "school_name"] = "Izmir Science High School"
    df_.loc[df_["school_name"] == "Mehmet Emin Resulzade Anadolu Lisesi", "school_name"] = "Mehmet Emin Resulzade Anatolian High School"
    df_.loc[df_["school_name"] == "İçel Anadolu Lisesi", "school_name"] = "İçel Anatolian High School"
    df_.loc[df_["school_name"] == "Tuzla Anadolu Teknik Lisesi", "school_name"] = "Tuzla Anatolian Technical High School"
    df_.loc[df_["school_name"] == "Adile Mermerci Anadolu Lisesi", "school_name"] = "Adile Mermerci Anatolian High School"
    df_.loc[df_["school_name"] == "Izmir Fen Lisesi", "school_name"] = "Izmir Science High School"
    df_.loc[df_["school_name"] == "Sakarya Anadolu Lisesi", "school_name"] = "Sakarya Anatolian High School"
    df_.loc[df_["school_name"] == "Eskişehir Anadolu Lisesi", "school_name"] = "Eskisehir Anatolian High School"
    df_.loc[df_["school_name"] == "Çanakkale Fen Lisesi", "school_name"] = "Çanakkale Science High School"
    df_.loc[df_["school_name"] == "Bolu Fen Lisesi", "school_name"] = "Bolu Science High School"
    df_.loc[df_["school_name"] == "Süleyman Demirel Anadolu Lisesi", "school_name"] = "Süleyman Demirel Anatolian High School"
    df_.loc[df_["school_name"] == "Maltepe Anadolu Lisesi", "school_name"] = "Maltepe Anatolian High School"
    df_.loc[df_["school_name"] == "Dede Korkut Anadolu Lisesi", "school_name"] = "Dede Korkut Anatolian High School"
    df_.loc[df_["school_name"] == "Nazilli Anadolu Lisesi", "school_name"] = "Nazilli Anatolian High School"
    df_.loc[df_["school_name"] == "Trabzon Yomra Fen Lisesi", "school_name"] = "Trabzon Yomra Science High School"
    df_.loc[df_["school_name"] == "Şişli Anadolu Lisesi", "school_name"] = "Sisli Anatolian High School"
    df_.loc[df_["school_name"] == "Ümraniye Anadolu Lisesi", "school_name"] = "Umraniye Anatolian High School"
    df_.loc[df_["school_name"] == "Silivri Lisesi", "school_name"] = "Silivri High School"
    df_.loc[df_["school_name"] == "Kuleli Askeri Lisesi", "school_name"] = "Kuleli Military High School"
    df_.loc[df_["school_name"] == "Malatya Anadolu Lisesi", "school_name"] = "Malatya Anatolian High School"
    df_.loc[df_["school_name"] == "Nermin Mehmet Çekiç Anadolu Lisesi", "school_name"] = "Nermin Mehmet Çekiç Anatolian High School"
    df_.loc[df_["school_name"] == "Malatya Fen Lisesi", "school_name"] = "Malatya Science High School"
    df_.loc[df_["school_name"] == "Zonguldak Fen Lisesi", "school_name"] = "Zonguldak Science High School"
    df_.loc[df_["school_name"] == "Adana Fen Lisesi", "school_name"] = "Adana Science High School"
    df_.loc[df_["school_name"] == "İstanbul Anadolu Lisesi", "school_name"] = "Istanbul Anatolian High School"
    df_.loc[df_["school_name"] == "Sivas Fen Lisesi", "school_name"] = "Sivas Science High School"
    df_.loc[df_["school_name"] == "Meram Anadolu Lisesi", "school_name"] = "Meram Anatolian High School"
    df_.loc[df_["school_name"] == "Ankara Anadolu Lisesi", "school_name"] = "Ankara Anatolian High School"
    df_.loc[df_["school_name"] == "Eskişehir Fatih Fen Lisesi", "school_name"] = "Eskisehir Fatih Science High School"
    df_.loc[df_["school_name"] == "Hayrullah Kefoğlu Anadolu Lisesi", "school_name"] = "Hayrullah Kefoğlu Anatolian High School"
    df_.loc[df_["school_name"] == "Tuzla Teknik Lisesi", "school_name"] = "Tuzla Technical High School"
    df_.loc[df_["school_name"] == "İstiklal Makzume Anadolu Lisesi", "school_name"] = "Istiklal Makzume Anatolian High School"
    df_.loc[df_["school_name"] == "Gemlik Celal Bayar Anadolu Lisesi", "school_name"] = "Gemlik Celal Bayar Anatolian High School"
    df_.loc[df_["school_name"] == "Profilo Anadolu Teknik Lisesi", "school_name"] = "Profilo Anatolian Technical High School"
    df_.loc[df_["school_name"] == "Edirne Süleyman Demirel Fen Lisesi", "school_name"] = "Edirne Süleyman Demirel Science High School"
    df_.loc[df_["school_name"] == "Konya Meram Fen Lisesi", "school_name"] = "Konya Meram Science High School"
    df_.loc[df_["school_name"] == "Gebze Anadolu Lisesi", "school_name"] = "Gebze Anatolian High School"
    df_.loc[df_["school_name"] == "Erzurum Fen Lisesi", "school_name"] = "Erzurum Science High School"
    df_.loc[df_["school_name"] == "Ordu Anadolu Lisesi", "school_name"] = "Ordu Anatolian High School"
    df_.loc[df_["school_name"] == "Kocaeli Fen Lisesi", "school_name"] = "Kocaeli Science High School"
    df_.loc[df_["school_name"] == "Haydarpaşa Anadolu Teknik Lisesi", "school_name"] = "Haydarpasa Anatolian Technical High School"
    df_.loc[df_["school_name"] == "Beykoz Anadolu Lisesi", "school_name"] = "Beykoz Anatolian High School"
    df_.loc[df_["school_name"] == "Konya Meram Anadolu Lisesi", "school_name"] = "Konya Meram Anatolian High School"
    df_.loc[df_["school_name"] == "Haydarpaşa Teknik Lisesi", "school_name"] = "Haydarpasa Technical High School"
    df_.loc[df_["school_name"] == "Gaziosmanpaşa Anadolu Lisesi", "school_name"] = "Gaziosmanpasa Anatolian High School"
    df_.loc[df_["school_name"] == "Maltepe Askeri Lisesi", "school_name"] = "Maltepe Military High School"
    df_.loc[df_["school_name"] == "Hasan Polatkan Anadolu Lisesi", "school_name"] = "Hasan Polatkan Anatolian High School"
    df_.loc[df_["school_name"] == "Bursa Erkek Lisesi", "school_name"] = "Bursa Boys' High School"
    df_.loc[df_["school_name"] == "Maltepe Fen Lisesi", "school_name"] = "Maltepe Science High School"
    df_.loc[df_["school_name"] == "Şükrü Şankaya Anadolu Lisesi", "school_name"] = "Şükrü Şankaya Anatolian High School"
    df_.loc[df_["school_name"] == "Gaziantep Anadolu Lisesi", "school_name"] = "Gaziantep Anatolian High School"
    df_.loc[df_["school_name"] == "Konak Anadolu Lisesi", "school_name"] = "Konak Anatolian High School"
    df_.loc[df_["school_name"] == "Buca Anadolu Lisesi", "school_name"] = "Buca Anatolian High School"
    df_.loc[df_["school_name"] == "Atakent Anadolu Lisesi", "school_name"] = "Atakent Anatolian High School"
    df_.loc[df_["school_name"] == "Hüseyin Bürge Anadolu Lisesi", "school_name"] = "Hüseyin Bürge Anatolian High School"
    df_.loc[df_["school_name"] == "Bursa Fen Lisesi", "school_name"] = "Bursa Science High School"
    df_.loc[df_["school_name"] == "denizli anadolu lisesi", "school_name"] = "Denizli Anatolian High School"
    df_.loc[df_["school_name"] == "Adnan menderes anadolu lisesi", "school_name"] = "Adnan Menderes Anatolian High School"
    df_.loc[df_["school_name"] == "kenan evren anadolu lisesi", "school_name"] = "Kenan Evren Anatolian High School"
    df_.loc[df_["school_name"] == "Sakarya anadolu lisesi", "school_name"] = "Sakarya Anatolian High School"
    df_.loc[df_["school_name"] == "Sekine Evren Anadolu Lisesi", "school_name"] = "Sekine Evren Anatolian High School"
    df_.loc[df_["school_name"] == "Zeytinburnu Teknik Lisesi", "school_name"] = "Zeytinburnu Technical High School"
    df_.loc[df_["school_name"] == "İstanbul Ticaret Odası Anadolu Teknik Lisesi", "school_name"] = "Istanbul Chamber Of Commerce Anatolian Technical High School"
    df_.loc[df_["school_name"] == "Üsküdar Anadolu Lisesi", "school_name"] = "Uskudar Anatolian High School"
    df_.loc[df_["school_name"] == "Aydın Fen Lisesi", "school_name"] = "Aydın Science High School"

    return df_

def fix_languages(dataframe: pd.DataFrame) -> pd.DataFrame:

    ger = 'German'
    eng = 'English'
    tr = 'Turkish'
    spa = 'Spanish'
    jp = 'Japanese'
    rus = 'Russian'
    fr = 'French'
    ita = 'Italian'
    chi = 'Chinese'
    mak = 'Macedonian'
    ara = 'Arabic'

    df_ = dataframe.copy()
    df_ = df_.drop_duplicates()
    df_['language'] = df_['language'].apply(lambda x: str(x).strip())
    df_ = df_.loc[df_["language"] != "Ms Sql (Sorgu dili)"]
    df_ = df_.loc[df_["language"] != 'Assembly']
    df_ = df_.loc[df_["language"] != 'SQL']
    df_ = df_.loc[df_["language"] != 'oracle data integrator']

    df_.loc[df_['language'] == 'Turkısh', 'language'] = tr
    df_.loc[df_['language'] == 'türkçe', 'language'] = tr
    df_.loc[df_['language'] == 'turkish', 'language'] = tr
    df_.loc[df_['language'] == 'turkçe', 'language'] = tr
    df_.loc[df_['language'] == 'Turksih', 'language'] = tr
    df_.loc[df_['language'] == 'Türkisch', 'language'] = tr
    df_.loc[df_['language'] == 'Türkçe', 'language'] = tr
    df_.loc[df_['language'] == 'Turkce', 'language'] = tr
    df_.loc[df_['language'] == 'Türkçe', 'language'] = tr
    df_.loc[df_['language'] == 'Türkish', 'language'] = tr
    df_.loc[df_['language'] == 'Turkçe', 'language'] = tr
    df_.loc[df_['language'] == 'Türkce', 'language'] = tr
    df_.loc[df_['language'] == 'Turkish,', 'language'] = tr
    df_.loc[df_['language'] == '3- Turkish', 'language'] = tr
    df_.loc[df_['language'] == 'Türkce', 'language'] = tr
    df_.loc[df_['language'] == 'Türkçe,', 'language'] =  tr
    df_.loc[df_["language"] == "Tükçe", "language"] = tr
    df_.loc[df_["language"] == "TÜRKÇE", "language"] = tr
    df_.loc[df_["language"] == "Türk", "language"] = tr
    df_.loc[df_["language"] == "Türke", "language"] = tr
    df_.loc[df_['language'] == 'turkce', 'language'] = tr
    df_.loc[df_["language"] == "Türkçe (Turkish)", "language"] = tr
    df_.loc[df_["language"] == 'Türkçe / Turkish', "language"] = tr

    df_.loc[df_['language'] == 'İngilizce', 'language'] = eng
    df_.loc[df_['language'] == 'english', 'language'] = eng
    df_.loc[df_['language'] == 'Englisch', 'language'] = eng
    df_.loc[df_['language'] == 'İnglizce', 'language'] = eng
    df_.loc[df_['language'] == 'ENGLISH', 'language'] = eng
    df_.loc[df_['language'] == 'ingilizce', 'language'] = eng
    df_.loc[df_['language'] == 'inglizce', 'language'] = eng
    df_.loc[df_['language'] == 'İng', 'language'] = eng
    df_.loc[df_['language'] == 'İngilizce A2', 'language'] = eng
    df_.loc[df_['language'] == 'Engish', 'language'] = eng
    df_.loc[df_['language'] == 'İngilice', 'language'] = eng
    df_.loc[df_['language'] == 'İngilizce - IELTS 7', 'language'] = eng
    df_.loc[df_['language'] == 'İngilizce/English', 'language'] = eng
    df_.loc[df_['language'] == 'İngilize', 'language'] = eng
    df_.loc[df_['language'] == 'İngilzce', 'language'] = eng
    df_.loc[df_['language'] == 'İngizce', 'language'] = eng
    df_.loc[df_['language'] == 'İngizice', 'language'] = eng
    df_.loc[df_['language'] == 'İngilizice', 'language'] = eng
    df_.loc[df_['language'] == 'İngilizce,', 'language'] = eng
    df_.loc[df_['language'] == 'İngilzce,', 'language'] = eng
    df_.loc[df_['language'] == 'English, Middle (1100-1500)', 'language'] = eng
    df_.loc[df_['language'] == 'İngilizce, Orta (1100-1500)', 'language'] = eng
    df_.loc[df_['language'] == 'INGILIZCE', 'language'] = eng
    df_.loc[df_['language'] == 'İNGİLİZCE', 'language'] = eng
    df_.loc[df_['language'] == 'English US', 'language'] = eng
    df_.loc[df_['language'] == 'English UK', 'language'] = eng
    df_.loc[df_['language'] == '2- English', 'language'] = eng
    df_.loc[df_['language'] == '2- English', 'language'] = eng
    df_.loc[df_['language'] == 'İngilizce / English', 'language'] = eng
    df_.loc[df_['language'] == 'English (US)', 'language'] =  eng
    df_.loc[df_['language'] == 'English, Advanced', 'language'] =  eng
    df_.loc[df_['language'] == 'English C1', 'language'] =  eng
    df_.loc[df_['language'] == 'English, Pre-Advance', 'language'] =  eng
    df_.loc[df_['language'] == 'English (Upper-Intermediate)', 'language'] =  eng
    df_.loc[df_['language'] == 'English-B2 Upper Intermediate', 'language'] =  eng
    df_.loc[df_['language'] == '■ English ■', 'language'] =  eng
    df_.loc[df_['language'] == 'English(advanced)', 'language'] =  eng
    df_.loc[df_['language'] == 'English - (YDS : 93,75)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce (English)', 'language'] =  eng
    df_.loc[df_['language'] == 'English (B2, Upper-Intermediate)', 'language'] =  eng
    df_.loc[df_['language'] == 'English (B2)', 'language'] =  eng
    df_.loc[df_['language'] == 'Advanced English', 'language'] =  eng
    df_.loc[df_['language'] == 'English - Global Village Sydney Australia', 'language'] =  eng
    df_.loc[df_['language'] == 'English (Advanced)', 'language'] =  eng
    df_.loc[df_['language'] == 'English upper intermediate', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, İyi', 'language'] =  eng
    df_.loc[df_['language'] == 'İnglilizce', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce-B1 Wimbledon language academy eğitim sürecindeyim.', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce (C1)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce (orta)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce iyi', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce (Advanced)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, İyi düzeyde', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce-intermediate', 'language'] =  eng
    df_.loc[df_['language'] == 'Mesleki İng.', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, B2', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce(C1)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce ( TOEIC - 725 )', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce ( orta düzeyde )', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce- BELS english school. as upper intermediate', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce (excellent)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, iyi', 'language'] =  eng
    df_.loc[df_['language'] == 'Mesleki İngilizce', 'language'] =  eng
    df_.loc[df_['language'] == 'İngiliz', 'language'] =  eng
    df_.loc[df_['language'] == 'English - Professional working proficiency', 'language'] = eng
    df_.loc[df_['language'] == 'Ingilizce', 'language'] = eng
    df_.loc[df_['language'] == 'English,', 'language'] = eng
    df_.loc[df_['language'] == 'İngilizce Pre-Intermediate', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce(Orta)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce(Orta Seviye)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce | B1', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce (Upper-Intermediate)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, ileri', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce(B2)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce (B1 - B2)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, Upper Intermediate', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, B1', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, İleri (2500-3000)', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce, Orta/İleri Düzey', 'language'] =  eng
    df_.loc[df_['language'] == 'İngilizce %30', 'language'] =  eng
    df_.loc[df_['language'] == 'orta düzey İngilizce', 'language'] =  eng
    df_.loc[df_['language'] == 'Engilish,', 'language'] =  eng
    df_.loc[df_['language'] == '-English', 'language'] = eng
    df_.loc[df_['language'] == 'ingilizce(english)', 'language'] = eng
    df_.loc[df_['language'] == 'Englis', 'language'] = eng
    df_.loc[df_['language'] == 'Inglizce', 'language'] = eng

    df_.loc[df_['language'] == 'Germany', 'language'] = ger
    df_.loc[df_['language'] == 'almanca', 'language'] = ger
    df_.loc[df_['language'] == 'GERMAN', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca', 'language'] = ger
    df_.loc[df_['language'] == 'ALMANCA', 'language'] = ger
    df_.loc[df_['language'] == 'german', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca(Beginner)', 'language'] = ger
    df_.loc[df_['language'] == 'German (Intermediate)', 'language'] = ger
    df_.loc[df_['language'] == 'German Language', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca B2', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca, Orta Yüksek (yaklaşık 1050-1500)', 'language'] = ger
    df_.loc[df_['language'] == 'German, B2.2 Goethe Instıtut - İZMİR', 'language'] = ger
    df_.loc[df_['language'] == 'Deutsche', 'language'] = ger
    df_.loc[df_['language'] == 'Deutsch', 'language'] =  ger
    df_.loc[df_['language'] == 'Almanca(A1)', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca (Düşük Seviye)', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca (başlangıç)', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca/German', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca / Deutsch', 'language'] = ger
    df_.loc[df_['language'] == 'Germanic languages', 'language'] = ger
    df_.loc[df_['language'] == 'German(beginner)', 'language'] = ger
    df_.loc[df_['language'] == 'German (A2)', 'language'] = ger
    df_.loc[df_['language'] == 'German (Beginner)', 'language'] = ger
    df_.loc[df_['language'] == 'Gerrman', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca (IAnfänger A2) (Elementary)', 'language'] = ger
    df_.loc[df_['language'] == 'Almanca (basic)', 'language'] = ger
    df_.loc[df_['language'] == 'German (Deutsches Sprachdiplom - 2.Stufe)', 'language'] = ger
    df_.loc[df_['language'] == 'Deutsch - B1', 'language'] = ger
    
    df_.loc[df_['language'] == 'arapça', 'language'] = ara
    df_.loc[df_['language'] == 'arapca', 'language'] = ara
    df_.loc[df_['language'] == 'arabic', 'language'] = ara
    df_.loc[df_['language'] == 'Arabic', 'language'] = ara
    df_.loc[df_['language'] == 'Arapca', 'language'] = ara
    df_.loc[df_['language'] == 'Arabic (only very basic speaking skills)', 'language'] = ara
    df_.loc[df_['language'] == 'Arabe', 'language'] = ara
    df_.loc[df_['language'] == 'Arapça-A1-A2', 'language'] = ara
    df_.loc[df_['language'] == 'Arabish', 'language'] = ara
    df_.loc[df_['language'] == 'South Sudanese Arabic', 'language'] = ara
    df_.loc[df_['language'] == 'Arapça', 'language'] = ara
    df_.loc[df_['language'] =='Arapça(Temel Düzeyde)', 'language'] = ara

    df_.loc[df_['language'] == 'ISPANYOLCA', 'language'] = spa
    df_.loc[df_['language'] == 'Español', 'language'] = spa
    df_.loc[df_['language'] == 'ispanyolca', 'language'] = spa
    df_.loc[df_['language'] == 'SPANISH', 'language'] = spa
    df_.loc[df_['language'] == 'Espanol', 'language'] = spa
    df_.loc[df_['language'] == 'İspanyolca', 'language'] = spa
    df_.loc[df_['language'] == 'Başlangıç seviyesinde İspanyolca', 'language'] = spa
    df_.loc[df_['language'] == 'İspanyolca, Başlangıç', 'language'] = spa
    df_.loc[df_['language'] == 'Spani', 'language'] = spa
    df_.loc[df_['language'] == 'Espańol', 'language'] = spa
    
    df_.loc[df_['language'] == 'İtalian', 'language'] = ita
    df_.loc[df_['language'] == 'italian', 'language'] = ita
    df_.loc[df_['language'] == 'Italyanca', 'language'] = ita
    df_.loc[df_['language'] == 'İtalyanca', 'language'] = ita
    df_.loc[df_['language'] == 'italyanca', 'language'] = ita
    df_.loc[df_['language'] == 'Italien', 'language'] = ita
    df_.loc[df_['language'] == 'Italiano', 'language'] =  ita
    df_.loc[df_['language'] == 'italiano', 'language'] = ita

    df_.loc[df_['language'] == 'Fransizca', 'language'] = fr
    df_.loc[df_['language'] == 'Fransızca', 'language'] = fr
    df_.loc[df_['language'] == 'Fransızca(Université Galatasaray)', 'language'] = fr
    df_.loc[df_['language'] == 'Französisch', 'language'] =  fr
    df_.loc[df_['language'] == 'France', 'language'] =  fr
    df_.loc[df_['language'] == 'Francais', 'language'] =  fr
    df_.loc[df_['language'] == 'Fransa', 'language'] =  fr
    df_.loc[df_['language'] == 'Fransız', 'language'] =  fr
    df_.loc[df_['language'] == '1- French', 'language'] = fr
    df_.loc[df_['language'] == 'Français', 'language'] = fr

    df_.loc[df_['language'] == 'Chinese (Simplified)', 'language'] =  chi
    df_.loc[df_['language'] == 'Elementary Chinese', 'language'] =  chi
    df_.loc[df_['language'] == 'Chinese(Simplified-Mandarin) - 中文', 'language'] =  chi
    df_.loc[df_['language'] == 'Çince (Mandarin)', 'language'] =  chi
    df_.loc[df_['language'] == 'Çince (Basitleştirilmiş)', 'language'] =  chi
    df_.loc[df_['language'] == 'Çinçe', 'language'] =  chi
    df_.loc[df_['language'] == 'Çince', 'language'] = chi
    df_.loc[df_['language'] == 'CHINESE', 'language'] = chi

    df_.loc[df_['language'] == 'Japanese(Beginner)', 'language'] = jp
    df_.loc[df_['language'] == 'Japanese (Roomaji)', 'language'] = jp
    df_.loc[df_['language'] == 'Japanesse(Roomaji)', 'language'] = jp
    df_.loc[df_['language'] == 'Japonca | A1', 'language'] = jp
    df_.loc[df_['language'] == 'Japonca', 'language'] = jp
    df_.loc[df_['language'] == 'japonca', 'language'] = jp

    df_.loc[df_['language'] == 'Rusça', 'language'] = rus
    df_.loc[df_['language'] == 'rusca', 'language'] = rus
    df_.loc[df_["language"] == "rusça", "language"] = rus
    df_.loc[df_['language'] == 'Russian Русский Язык', 'language'] = rus
    df_.loc[df_["language"] == 'russian', "language"] = rus

    df_.loc[df_['language'] == 'Makedonca', 'language'] =  mak
    df_.loc[df_['language'] == 'Makedonski', 'language'] =  mak

    df_.loc[df_['language'] == 'Türkçe İşaret Dili', 'language'] = 'Turkish Sign Language'
    df_.loc[df_['language'] == 'Kürtçe', 'language'] = 'Kurdish'
    df_.loc[df_['language'] == 'Korece', 'language'] = 'Korean'
    df_.loc[df_['language'] == 'Bulgarca', 'language'] = 'Bulgarian'
    df_.loc[df_['language'] == 'Azerice', 'language'] = 'Azerbaijani'
    df_.loc[df_['language'] == 'Azərbaycan', 'language'] = 'Azerbaijani'
    df_.loc[df_['language'] == 'Portekizce', 'language'] = 'Portuguese'
    df_.loc[df_['language'] == 'Yunanca', 'language'] = 'Greek'
    df_.loc[df_['language'] == 'Latince', 'language'] = 'Latin'
    df_.loc[df_['language'] == 'kürdi', 'language'] = 'Kurdish'
    df_.loc[df_['language'] == 'Kurdî', 'language'] = 'Kurdish'
    df_.loc[df_['language'] == 'Macarca', 'language'] = 'Hungarian'
    df_.loc[df_['language'] == 'Litvanyaca', 'language'] = 'Lithuanian'
    df_.loc[df_['language'] == 'Korece | A2', 'language'] = 'Korean'
    df_.loc[df_['language'] == 'Dutch (beginner)', 'language'] = 'Dutch'
    df_.loc[df_["language"] == "İşaret Dilleri", "language"] = "Sign Languages"
    df_.loc[df_['language'] == 'Türkçe, Osmanlıca (1500-1928)', 'language'] = 'Turkish, Ottoman (1500-1928)'
    df_.loc[df_['language'] == 'Boşnakça', 'language'] = 'Bosnian'
    df_.loc[df_['language'] == 'Türk İşaret Dili', 'language'] = 'Turkish Sign Language'
    df_.loc[df_['language'] == 'İngilizce, Eski (yaklaşık 450-1100)', 'language'] = 'English, Old (ca.450-1100)'
    df_.loc[df_["language"] == "Farsça", "language"] = "Persian"
    df_.loc[df_["language"] == "Sırpça", "language"] = "Serbian"
    df_.loc[df_["language"] == "İsveççe", "language"] = "Swedish"
    df_.loc[df_["language"] == "Kazakça", "language"] = "Kazakh"
    df_.loc[df_["language"] == "Arnavutça", "language"] = "Albanian"
    df_.loc[df_["language"] == "Çekçe", "language"] = "Czech"
    df_.loc[df_["language"] == "Özbekçe", "language"] = "Uzbek"
    df_.loc[df_["language"] == "Hırvatça", "language"] = "Croatian"
    df_.loc[df_["language"] == "Bokmål, Norveç", "language"] = "Norwegian"
    df_.loc[df_["language"] == "Norveççe", "language"] = "Norwegian"
    df_.loc[df_["language"] == "Osmanlı Türkçesi", "language"] = "Ottoman Turkish"
    df_.loc[df_["language"] == "Slovakça", "language"] = "Slovak"
    df_.loc[df_["language"] == "Endonezya dili", "language"] = "Indonesian"
    df_.loc[df_["language"] == "İşaret Dilleri", "language"] = "Sign Languages"
    df_.loc[df_["language"] == "Gürcüce", "language"] = "Georgian"
    df_.loc[df_["language"] == "Türkmence", "language"] = "Turkmen"
    df_.loc[df_["language"] == "Türkmen", "language"] = "Turkmen"
    df_.loc[df_["language"] == "İbranice", "language"] = "Hebrew"

    return df_

def fix_degree(dataframe) -> pd.DataFrame:

    df_ = dataframe.copy()

    # Degree
    high_school = 'High School'
    bachelors_deg = "Bachelor's degree"
    associates_deg = "Associate's degree"
    masters_deg = "Master's degree"
    bsc = 'Bachelor of Science'
    msc = 'Master of Science'
    mba = 'Master of Business Administration'
    bba = 'Bachelor of Business Administration'
    phd = 'Doctor of Philosophy'
    basc = 'Bachelor of Applied Science'
    be = 'Bachelor of Engineering'
    me = 'Master of Engineering'

    #for i in df_['degree'].dropna().unique():
    #    if re.sub(r'[^\w\s]', '', re.sub(r'\d+', '', i)).strip() == '':
    #        df_.loc[df_['degree'] == i, 'degree'] = np.nan

    #Bachelors degree
    df_.loc[df_['degree'] == 'Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi (2,68/4)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor's Degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor’s Degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "bachelor", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor Degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "bachelor's degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "bachelor's", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "bachelor degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == " bachelor's degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Electrical and Electronics Engineering bachelor degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "bachelor degree 3.07", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Three years of informatics completed successfully out of 5 years bachelor degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Bachelor degree 2,79/4', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Lisans Derecesi / Bachelor's degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi (2.90/4.00 - 4.00/4.00)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi ', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi  ', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi 3.35\\4.00', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == '3.16/4 Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == '2.72/4 Çift Anadal / Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == ' 3.35 Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi(Erasmus)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi Mezunu', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Lisans Derecesi,Bachelor's Degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi, İngilizce', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi, Çift Anadal', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi Fakülte Birinciliği ', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi, 3.45', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Çift Anadal Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi / License Degree', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi, 3.57/4.00', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi (Bölüm Birincisi) ', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi (Bachelor)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi- Dokuz Eylül Üniversitesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi 3.3/4', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi , 3.49', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi: 2.90', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Bachelors / Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi  GPA : 3.36/4', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi - 3,32 / 4', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi 2.83/4', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi %25 Burslu', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == '2.91/4-Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi (terk) / Undergraduate (dropped)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi: (TERK)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi / Bachelor Degree', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi (Bachelor’s Degree)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi, 3.32', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor's degree, Bachelor of Science - BS, Lisans Derecesi", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Bilgisayar Mühendisliği Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi (Erasmus)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'YG, Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi - Bachelor Degree', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi-  3.76/4 ', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == '3.30 Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi(4.)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == '3,21 - Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi (3.36)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi - Erasmus', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi, Bölüm Birincisi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == '3.52 - Lisans Derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi ( Mühendislik Fakültesi )', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi,Hazırlık(İng)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Derecesi:2.54', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans ', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans, Anadal', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Lisans / Bachelor's degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans, İkinci Anadal', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans ( Açık Öğretim)', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Mezunu', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Öğrencisi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans Açık Öğretim', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor's degree / Lisans", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == 'Lisans derecesi', 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor's degree / Lisans ", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "lisans", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor degree", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelor's", 'degree'] = bachelors_deg
    df_.loc[df_['degree'] == "Bachelors", 'degree'] = bachelors_deg

    df_.loc[df_['degree'] == "Önlisans", 'degree'] = associates_deg
    df_.loc[df_['degree'] == "Ön Lisans", 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'önlisans', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'ön lisans', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'ön lisans ', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'önlisans ', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'ön Lisans', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'ön lisana', 'degree'] = associates_deg
    df_.loc[df_['degree'] == ' ön lisans', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'ÖnLisans Derecesi', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'Ön Lisans Derecesi', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'On Lisans Derecesi', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'ÖnLisans', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'Ön Lisans ', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'Açık Öğretim / Ön Lisans', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'Ön Lisans (derece ile)', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'Ön lisans', 'degree'] = associates_deg
    df_.loc[df_['degree'] == 'Associate Degree', 'degree'] = associates_deg
    df_.loc[df_['degree'] == "Associate’s Degree", 'degree'] = associates_deg
    df_.loc[df_['degree'] == "Associate's Degree", 'degree'] = associates_deg
    df_.loc[df_['degree'] == "Associate", 'degree'] = associates_deg
    df_.loc[df_['degree'].isin(['associate degree', ' Aassociate Degree', ' associate degree']), 'degree'] = associates_deg

    df_.loc[df_['degree'] == 'Tezli Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == "Yüksek Lisans (Master)", 'degree'] = masters_deg
    df_.loc[df_['degree'] == "Yüksek Lisans", 'degree'] = masters_deg
    df_.loc[df_['degree'] == "Master’s Degree", 'degree'] = masters_deg
    df_.loc[df_['degree'] == "Master's Degree", 'degree'] = masters_deg
    df_.loc[df_['degree'] == "Master", 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans ', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master-3,25)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Fen Edebiyat Fakültesi - Yüksek Lisans - Tez Safhası ( Devam Etmekte )', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master) Özel Öğrenci', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans Fen Bilimleri Enstitüsü', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master Degree)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Y.Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksel Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yuksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Tezli Yüksek Lisans (Master)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master), Bilgisayar Mühendisliği', 'degree'] = masters_deg
    df_.loc[df_['degree'] == '(Yüksek Lisans)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'İstatistik Bölümü Yüksek Lisans Mezunu', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master) 3.90', 'degree'] = masters_deg    
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master), İngilizce', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (MSc)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Mühendislik ve Fen Bilimleri Enstitüsü, Bilgisayar Mühendisliği Yüksek Lisans (İngilizce)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master) ', 'degree'] = masters_deg
    df_.loc[df_['degree'] == '3.56 Yüksek Lisans (Master)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Tezli Yüksek Lisans-MSc Degree', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Ms)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans - Fen Bilimleri Enstitüsü', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master) Elektronik ve Haberleşme Mühendisliği', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Fen Bilimleri Enstitüsü (Yüksek Lisans)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Tezsiz Yüksek Lisans Özel Öğrenci', 'degree'] = masters_deg
    df_.loc[df_['degree'] == "Master's degree / Yüksek Lisans", 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Mühendislik Fakültesi - Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Fen Bilimleri Enstitütüsü / Tezsiz Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Lisans ile Birleştirilmiş Tezsiz Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master) Tam Burslu', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Tezsiz Yüksek Lisans (Master)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Gazi Üniversitesi Fizik Yüksek Lisans Mezunu', 'degree'] = masters_deg
    df_.loc[df_['degree'] == "Yüksek Lisans / Master's degree", 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master) (Tezli)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Pazarlama Yüksek Lisans Öğrencisi', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Fen Bilimleri Fakültesi - Yüksek Lisans (Tezli)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans Öğrenci', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master’s degree)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans Tezli (Master)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Fen Bilimleri Enstitüsü Elektronik ve Bilgisayar Eğitimi Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == "Yüksek Lisans / Master's Degree", 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Elektronik ve Haberleşme Mühendisliği Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Fen Bilimleri (Yüksek Lisans)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'İşletme Yüksek Lisans Mezun', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Enformatik Tezli Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Tezli Yüksek Lisans (Master) ', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master) - dropped', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master)-Tezli', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans - Tezsiz -', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Tezsiz) ', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master) Öğrencisi', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Bilgisayar Mühendisliği Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Tezsiz Yüksek Lisans', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans Öğrencisi', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'Yüksek Lisans (Master)(REMEDIAL)', 'degree'] = masters_deg
    df_.loc[df_['degree'] == 'yüksek lisans', 'degree'] = masters_deg

    df_.loc[df_['degree'] == "Lise", 'degree'] = high_school
    df_.loc[df_['degree'] == "High School Diploma", 'degree'] = high_school
    df_.loc[df_['degree'] == 'lise', 'degree'] = high_school
    df_.loc[df_['degree'] == 'lise ', 'degree'] = high_school
    df_.loc[df_['degree'] == 'Anadolu lisesi mezunu', 'degree'] = high_school
    df_.loc[df_['degree'] == 'süper lise', 'degree'] = high_school
    df_.loc[df_['degree'] == 'Endüstri lise', 'degree'] = high_school
    df_.loc[df_['degree'] == 'Teknik lise', 'degree'] = high_school
    df_.loc[df_['degree'] == 'Ortaöğretim(lise)', 'degree'] = high_school
    df_.loc[df_['degree'] == 'lise mezunu', 'degree'] = high_school
    df_.loc[df_['degree'] == 'imkb anadolu teknik lisesi', 'degree'] = high_school
    df_.loc[df_['degree'] == 'Meslek lisesi', 'degree'] = high_school
    df_.loc[df_['degree'] == 'lise/ High School ', 'degree'] = high_school
    df_.loc[df_['degree'] == 'Anadolu meslek lisesi mezunu', 'degree'] = high_school
    df_.loc[df_['degree'] == 'Fen lisesi mezunu', 'degree'] = high_school
    df_.loc[df_['degree'] == 'izmit lisesi', 'degree'] = high_school
    df_.loc[df_['degree'] == '700. yıl anadolu lisesi', 'degree'] = high_school
    df_.loc[df_['degree'] == 'LİSE', 'degree'] = high_school
    df_.loc[df_['degree'] == 'ANADOLU TEKNİK LİSESİ', 'degree'] = high_school
    df_.loc[df_['degree'] == 'SABANCI ANADOLU TEKNİK LİSESİ', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Kabataş Erkek Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Hınıs imam hatip lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Atatürk lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Malatya lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'İzmir Atatürk Lisesi'] = high_school
    #df_.loc[df_['school_name'] == 'Ankara Atatürk Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Bornova Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Sırrı Yırcalı Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Kadıköy Anadolu Lisesi ve Maarif Koleji', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Bornova Anadolu Lisesi (İzmir Koleji)', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Beşiktaş Atatürk Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Ankara Atatürk Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Istanbul Erkek Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Burak Bora Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Haydarpaşa Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Karşıyaka Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Kadıköy Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Bursa Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Denizli Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'İstanbul Köy Hizmetleri Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Şehremini Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Tekirdağ Fen Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'İstanbul Atatürk Fen Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Bahçelievler Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Kocaeli Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Adnan Menderes Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Hüseyin Avni Sözen Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Tekirdağ Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'] == 'Gazi Anadolu Lisesi', 'degree'] = high_school
    #df_.loc[df_['school_name'].astype(str).str.contains('Lisesi'), 'degree'] = high_school
    #df_.loc[df_['school_name'].astype(str).str.contains('lisesi'), 'degree'] = high_school

    df_.loc[df_['degree'] == 'Bachelor of Science - BS', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BS)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (B.Sc.)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (B.S.)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - BSc', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science(B.Sc.)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Double Major, Bachelor of Science - BSc', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (B.Sc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science Degree', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - BSc.', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc), 2nd Degree of Graduation', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc),', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - B.Sc', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - BSc (Hons)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (Hons.B.Sc.)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Double Major, Bachelor of Science (B.S.)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BS), Mechanical Engineering', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science ', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - (BSc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science in Economics- BSc(Econ)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - BS, Physics, 3.06/4.00', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - B.Sc.', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science(BS)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science Computer Science - BS(CS)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - B.S.', 'degree'] = bsc
    df_.loc[df_['degree'] == ' Bachelor of Science', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BS) ', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc) - Not graduate', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc: Bachelor of Science', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (M.S.)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science in Computer Engineering', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science(B.Sc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science in Engineering', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BS), Chemistry', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BS), Electrical and Electronics Engineering, GPA 4.00', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science - BS, Computer Technologies and Information Systems', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Sciences - BS', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science With Honours', 'degree'] = bsc
    df_.loc[df_['degree'] == 'bachelor of science', 'degree'] = bsc
    df_.loc[df_['degree'] == "Bachelor's of science", 'degree'] = bsc
    df_.loc[df_['degree'] == "BS", 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc.', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc Engineering ', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc., Electrical Engineering', 'degree'] = bsc
    df_.loc[df_['degree'] == "Bachelor's Degree (BSc)", 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc in Computer Engineering 3.66/4.00', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Lisans Derecesi(BSc)', 'degree'] = bsc
    df_.loc[df_['degree'] == "Bachelor's degree (BSc)", 'degree'] = bsc
    df_.loc[df_['degree'] == 'Electrical and Electronics Engineering (BSc.)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science(BSc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc. Honours', 'degree'] = bsc
    df_.loc[df_['degree'] == '3.22/4, Lisans (BSc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc) - Computer Engineering', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science  (BSc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc-Eng)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc, Jeodesy and Photogrameteri', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc) ( Transferred to Middle East Technical University after 1st year)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc(Double Major)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science - BSc', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc,', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Mechanical Engineering (BSc) 2005', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc) (3.16 honored student)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc. Department of Electronic', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc) (3.06 honored student)', 'degree'] = bsc
    df_.loc[df_['degree'] == "Bachelor's degree, BSc.", 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc. Degree', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BS / BSc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science(BSc), Computer Enginnering', 'degree'] = bsc
    df_.loc[df_['degree'] == "Bachelor's of Science (BSc)", 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc (Hons) in Software Design with Cloud Computing', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc in Computer Science, 3.48/4.00 (CGPA)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc. In Dept. of Computer Engineering', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Undergraduate (BSc) with honors', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science (BSc)', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc) - Department of Information Technologies', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc), Geology/Earth Science', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science(BSc), Computer Science', 'degree'] = bsc
    df_.loc[df_['degree'] == 'Bachelor of Science (BSc) with Honors', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc Degree', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc. Electronics&Communication Engineering', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSc, Computer Engineering', 'degree'] = bsc
    df_.loc[df_['degree'] == 'BSC','degree'] = bsc
    df_.loc[df_['degree'] == 'BSCS(HONOURS)','degree'] = bsc
    df_.loc[df_['degree'] == 'B.Sc.','degree'] = bsc
    df_.loc[df_['degree'] == 'B.S.','degree'] = bsc
    df_.loc[df_['degree'] == 'B.S','degree'] = bsc

    df_.loc[df_['degree'] == 'Master of Science (MSc)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MS', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.Sc.)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MS)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.S.)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MSc', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science ', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc), 1st Degree of Graduation', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - M.Sc.', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science Degree', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science(M.Sc.)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc) ', 'degree'] = msc
    df_.loc[df_['degree'] == 'Research Assistant - Master of Science (MSc)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in IT (M.I.T.)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MS, Physics, 3.61/4.00', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science MSc', 'degree'] = msc
    df_.loc[df_['degree'] == 'Msc Master of Science in Electrical& Electronics Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc), Computer Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc.)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - Informatik', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MSc.', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.Sc)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - M.Sc', 'degree'] = msc
    df_.loc[df_['degree'] == ' Master of Science - MS ', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Information Technology', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science(MSc), Computer Software Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - Ms', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Engineering Management - MSEM', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science- MSc', 'degree'] = msc
    df_.loc[df_['degree'] == '3.44/4.00 Master of Science (MSc)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M. Sc.)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Engineering - MScEng', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - M. Sc.', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc), Mak. Dinamiği, Titreşim&Akustiği', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.Sc.) in Electronics Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Yüksek Lisans - Master of Science (MSc)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science  (M.Sc.)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science(MSc)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Information Systems (MSIS)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science / Yüksek Lisans', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MS(without thesis)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science -MS', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MSc Student', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - M.Sc. , 3.83/4', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Informatics Institute   ', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.Sc.) - DROPPED', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - M.S', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MS) with Thesis', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc)-Solid Mechanics', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science(MS)', 'degree'] = msc
    df_.loc[df_['degree'] == 'M.Sc. - Master of Science', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MS (with thesis) ', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MS,Computer Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Bachelor and Master of Science-MS,Chemistry(German)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc) with Thesis', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.Sc.) ', 'degree'] = msc
    df_.loc[df_['degree'] == '3.88 / 4 - Master of Science (MS), Applied Informatics', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.Sc.) Mechanical Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science, CGPA: 3.86/4.00', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Computer Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Computer Science (MSCS)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc) without thesis', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc-dropped)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.Sc) - dropped due to financial reasons', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science  (M.Sc.) with Thesis', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science and Engineering', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science - MSc with Thesis', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (M.Sc.) Biology', 'degree'] = msc
    df_.loc[df_['degree'] == 'Yüksek Lisans ( M.Sc.), Graduate, Master of Science', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc), Geology/Earth Science', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc)  (UK)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science  (MSc)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science in Engineering (M.Sc.Eng)', 'degree'] = msc
    df_.loc[df_['degree'] == 'Master of Science (MSc) - Non Thesis', 'degree'] = msc
    df_.loc[df_['degree'] == 'MSC - Thesis', 'degree'] = msc
    df_.loc[df_['degree'] == 'MSC', 'degree'] = msc
    df_.loc[df_['degree'] == 'MSC.', 'degree'] = msc
    df_.loc[df_['degree'] == 'MSc', 'degree'] = msc
    df_.loc[df_['degree'] == 'Yüksek Lisans (M.Sc)', 'degree'] = msc
    df_.loc[df_['degree'] == 'MS', 'degree'] = msc
    df_.loc[df_['degree'] == 'M.Sc.', 'degree'] = msc
    df_.loc[df_['degree'] == 'M.S.', 'degree'] = msc

    df_.loc[df_['degree'] == 'Master of Business Administration - MBA', 'degree'] = mba
    df_.loc[df_['degree'] == 'Master of Business Administration (MBA)', 'degree'] = mba
    df_.loc[df_['degree'] == 'Master of Business Administration (M.B.A.)', 'degree'] = mba
    df_.loc[df_['degree'] == 'MBA', 'degree'] = mba
    df_.loc[df_['degree'] == 'Mba', 'degree'] = mba
    df_.loc[df_['degree'] == '(MBA)-Marketing', 'degree'] = mba
    df_.loc[df_['degree'] == 'Yüksek Lisans (MBA)', 'degree'] = mba
    df_.loc[df_['degree'] == 'MBA,Master of Business Administration', 'degree'] = mba
    df_.loc[df_['degree'] == '(MBA)', 'degree'] = mba
    df_.loc[df_['degree'] == "MBA 3.8", 'degree'] = mba
    df_.loc[df_['degree'] == "Master of Business Management - (MBA)", 'degree'] = mba
    df_.loc[df_['degree'] == "Master of Business Administration, MBA", 'degree'] = mba
    df_.loc[df_['degree'] == "Master of Business Administration - MBA ", 'degree'] = mba
    df_.loc[df_['degree'] == "MBA (Master Degree) ", 'degree'] = mba
    df_.loc[df_['degree'] == "Master of Business Administration (MBA) - English", 'degree'] = mba
    df_.loc[df_['degree'] == "MBA degree", 'degree'] = mba
    df_.loc[df_['degree'] == "Master of Business Administration   3.87/4.00 GPA", 'degree'] = mba
    df_.loc[df_['degree'] == "Master of Business Administration (M. B. A.)", 'degree'] = mba
    df_.loc[df_['degree'] == "The Master of Business Administration (MBA)", 'degree'] = mba
    df_.loc[df_['degree'] == "Master of Business Administration -3,75", 'degree'] = mba
    df_.loc[df_['degree'] == 'İşletme Yüksek Lisans Programı (MBA)', 'degree'] = mba
    df_.loc[df_['degree'] == 'İşletme Tezsiz Yüksek Lisans Programı (MBA)', 'degree'] = mba
    df_.loc[df_['degree'] == 'İşletme Yüksek Lisans Programı / MBA', 'degree'] = mba
    df_.loc[df_['degree'] == 'GPA: 4.00/4.00 ,Master of Business Administration (M.B.A.)', 'degree'] = mba

    df_.loc[df_['degree'] == 'Bachelor of Applied Science (B.A.Sc.)', 'degree'] = basc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science - BASc', 'degree'] = basc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science (BASc)', 'degree'] = basc
    df_.loc[df_['degree'] == 'Erasmus Student at Bachelor of Applied Science - BASc', 'degree'] = basc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science (B.A.Sc.), Physics Engineering', 'degree'] = basc
    df_.loc[df_['degree'] == 'Bachelor of Applied Sciences - BAS', 'degree'] = basc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science - (BASc) ', 'degree'] = basc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science - B.A.Sc.', 'degree'] = basc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science (B.A.Sc', 'degree'] = basc
    df_.loc[df_['degree'] == 'Bachelor of Applied Science (B.Sc.)', 'degree'] = basc

    df_.loc[df_['degree'] == 'Bachelor of Engineering - BE', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B.E.)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B.Eng.)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (BEng)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (BE)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B. E.)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering(BE)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering - BE (Honoured)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering - BE ', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B.Eng)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (Eng.)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering - BE Outstanding Scholarship (%100)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering(Minor Program)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B.E.), Computer Engineering', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B.Eng.), High Honor List', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering - BEng', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B.Eng.) Honour Degree', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B.Sc.)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering - Computer Engineering', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (BE), Computer Science', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (B.E.), Makine Mühendisliği', 'degree'] = be
    df_.loc[df_['degree'] == "Bachelor of Engineering (B.Eng.)ineer's degree", 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering (%100 B.Eng.)', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering - B.Eng.', 'degree'] = be
    df_.loc[df_['degree'] == 'Bachelor of Engineering ', 'degree'] = be
    df_.loc[df_['degree'] == 'bachelor of engineering', 'degree'] = be
    df_.loc[df_['degree'] == "Engineer's degree", 'degree'] = be
    df_.loc[df_['degree'] == "Engineer's degree, Mechanical Engineering", 'degree'] = be
    df_.loc[df_['degree'] == "Engineer's degree,", 'degree'] = be
    df_.loc[df_['degree'] == "Engineer's degree (Exchange Student)", 'degree'] = be

    df_.loc[df_['degree'] == 'Master of Engineering - MEng', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering (M.Eng.)', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering (MEng)', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering (M. E.)', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering Management', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering (M.Eng.), Thermodynamics', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering (MSc)', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering (M.Eng.) Student', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering - Mechanical', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering (M.Eng.), Makine Mühendisliği', 'degree'] = me
    df_.loc[df_['degree'] == 'Master of Engineering and Business (M.Eng.)', 'degree'] = me

    df_.loc[df_['degree'] == 'Bachelor of Business Administration - BBA', 'degree'] = bba
    df_.loc[df_['degree'] == 'Bachelor of Business Administration (B.B.A.)', 'degree'] = bba
    df_.loc[df_['degree'] == 'Bachelor of Business Administration (BBA)', 'degree'] = bba
    df_.loc[df_['degree'] == 'Bachelor of Business Administration', 'degree'] = bba
    df_.loc[df_['degree'] == 'Bachelor of Business Administration - BBA ', 'degree'] = bba
    df_.loc[df_['degree'] == 'Bachelor of Bussines Administration', 'degree'] = bba
    df_.loc[df_['degree'] == 'Bachelor of Business Administration - BA', 'degree'] = bba
    df_.loc[df_['degree'] == 'BA', 'degree'] = bba

    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D.)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (PhD)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD(C)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD Candidate', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD ', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD (Candidate)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD cand.', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (PhD) (Candidate)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D.) - Abandoned', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD(Integrated)', 'degree'] = phd
    df_.loc[df_['degree'] == '•\tUniversiad Azteca Doctor of Philosophy in Business Administration , PhD (ID : 161AS19D0316)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (PhDc)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (PhD) (Dropped Out)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D.) in Business Administration', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (PhD.) Candidate', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D., Dr.-Ing.)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D.) (Not finished)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy(PhD)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD candidate', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (PhD) Student ', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D.) (not completed)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy Canditate- PhDc', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph. D.)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD (Working)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD (leave)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D.) (c)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy - PhD  (Dropout)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D. Candidate)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D.)(leave of absence)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (Ph.D.) - Dropped out of', 'degree'] = phd
    df_.loc[df_['degree'] == "Doctor of Philosophy - PhD (Cont'd)", 'degree'] = phd
    df_.loc[df_['degree'] == 'Drop Out - Doctor of Philosophy (PhD) ', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doctor of Philosophy (MSc & PhD Integrated) ', 'degree'] = phd
    df_.loc[df_['degree'] == 'PhD', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doktora ( PHD )', 'degree'] = phd
    df_.loc[df_['degree'] == 'Doktora (Dr.)', 'degree'] = phd
    df_.loc[df_['degree'] == 'Ph.D.', 'degree'] = phd
    df_.loc[df_['degree'] == 'phd', 'degree'] = phd

    df_.loc[df_['degree'] == 'Mühendislik Fakültesi', 'degree'] = 'Engineering'
    df_.loc[df_['degree'] == 'Mühendislik Fakültesi Mezunu', 'degree'] = 'Engineering'
    df_.loc[df_['degree'] == 'Engineering Faculty', 'degree'] = 'Engineering'
    df_.loc[df_['degree'] == 'Engineering', 'degree'] = 'Engineering'
    df_.loc[df_['degree'] == 'Faculty Of Engineering', 'degree'] = 'Engineering'
    df_.loc[df_['degree'] == 'Faculty of engineering', 'degree'] = 'Engineering'
    df_.loc[df_['degree'] == 'faculty of engineering', 'degree'] = 'Engineering'
    df_.loc[df_['degree'] == 'İşletme Fakültesi Mezunu', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Fakültesi', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Mezunu', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Fatültesi', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme fakültesi', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Fakültesi / Faculty of Management', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İŞLETME FAKÜLTESİ', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İŞLETME FAKULTESİ', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Fakültesi ', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Fakültesi Terk', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Fakültesi, MYO Mezunu', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'Sosyal Bilimler Enstitüsü, İşletme Anabilim Dalı, İşletme', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Fakültesi Öğrencisi', 'degree'] = 'Management'
    df_.loc[df_['degree'] == 'İşletme Fakultesi', 'degree'] = 'Management'

    df_.loc[df_['degree'] == '3.65/4.00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '52.95', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.5', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.50', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.01', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,5', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.14', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,7', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.30', 'degree'] = np.nan
    df_.loc[df_['degree'] == '73,25', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.2', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,14', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.05', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.0', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.20', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.15', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.10', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.3', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.02', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.65', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.22', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.06', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.55', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.16', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.40', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,12', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.68', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.27', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,25', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.1', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.75', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.28', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,20', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.03', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.33', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.54', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.21', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.23', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,50', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,2', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.09', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,78', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.13', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.17', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.66', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.47', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.11', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.24', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.07', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,21', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2.83', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.80', 'degree'] = np.nan
    df_.loc[df_['degree'] == '4.63', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,05', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2.63', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.26', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,13', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,86', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.08', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3,62', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.52', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.5/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '93', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.04', 'degree'] = np.nan
    df_.loc[df_['degree'] == '83', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2.73', 'degree'] = np.nan
    df_.loc[df_['degree'] == '3.70', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,80', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,5', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,7/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,75', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,87', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,42', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,77', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,50', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,63', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,66', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,82', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,92', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,7', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,71', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,48', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,65', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,98/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,95', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,76 / 4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,5/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,84', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,97', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,67', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,85', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,60', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,34', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,74', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,63/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,58', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,61', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,62', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,51', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,59', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,47', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,8', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,99', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,79', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,29', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,89', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,57', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,72', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,91', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,53', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,52', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,6', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,73', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,0', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,47 / 4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,35', 'degree'] = np.nan
    df_.loc[df_['degree'] == '92,2 /100', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,54 / 4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,90/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,36/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,54', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,36', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,86', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,70', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,58 / 4,0', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,55', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,30', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,64', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,44', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,96', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,99 / 4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,93', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,76/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,76', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,9', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,38', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,98', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,12', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,19', 'degree'] = np.nan
    df_.loc[df_['degree'] == '92,04', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,43', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,37', 'degree'] = np.nan
    df_.loc[df_['degree'] == '92,6 / 100', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,76/4.00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,2', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,80/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,41', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,88', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,56', 'degree'] = np.nan
    df_.loc[df_['degree'] == '72,28/100', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,22', 'degree'] = np.nan
    df_.loc[df_['degree'] == '4 / 2,92', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,69/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,85/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,90', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,67/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,85 / 4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,44 / 4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,68', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,78', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,25', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,24 /4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,02/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,58/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,35/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '82,29', 'degree'] = np.nan
    df_.loc[df_['degree'] == '92,07', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,5 / 4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,28', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,75/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,70/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,3/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,86/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '92,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '72,65', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,81', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,90/4', 'degree'] = np.nan
    df_.loc[df_['degree'] == '82,5', 'degree'] = np.nan
    df_.loc[df_['degree'] == '92,25', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,3', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,94', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,39', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,47/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,52/4,00', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,69', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,15', 'degree'] = np.nan
    df_.loc[df_['degree'] == '2,83', 'degree'] = np.nan

    return df_
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

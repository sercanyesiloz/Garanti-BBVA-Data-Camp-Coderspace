import numpy as np
import pandas as pd
import matplotlib
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




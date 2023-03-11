import re
from utils import *
import warnings
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')

def my_tokenizer(text):
    return re.split("\\s+",text)

def load_skills(path: str, size: int = 50, exact_match: bool = True) -> pd.DataFrame:

    df_ = pd.read_csv(path)
    df_ = df_.drop_duplicates().dropna(subset=["skill"])

    if exact_match:
        most_freq_skills = df_["skill"].value_counts()[:size].keys().tolist()
        for skill in tqdm(most_freq_skills):
            df_[f"skill_{skill}"] = df_["skill"].apply(lambda x: 1 if skill == x else 0)

        return (
            df_.drop(columns=["skill"], axis=1)
            .groupby(by=["user_id"], as_index=False)
            .sum()
            .merge(
                df_.groupby(by="user_id", as_index=False).agg(
                    total_skills=("skill", "nunique")
                ),
                on=["user_id"],
                how="left",
            )
        )

    else:
        vectorizer = CountVectorizer(
            max_features=size,
            stop_words=stopwords.words("english"),
            ngram_range=(1, 3),
            tokenizer=my_tokenizer,
        )

        return (
            pd.DataFrame(
                vectorizer.fit_transform(df_["skill"]).toarray(),
                columns=[f"skill_{str(f)}" for f in vectorizer.get_feature_names()],
            )
            .assign(user_id=df_["user_id"].tolist())
            .groupby(by="user_id", as_index=False)
            .sum()
            .merge(
                df_.groupby(by="user_id", as_index=False).agg(
                    total_skills=("skill", "nunique")
                ),
                on=["user_id"],
                how="left",
            )
        )

def load_languages(path: str, size: int = 8):

    df_ = pd.read_csv(path)
    df_ = df_.drop_duplicates()

    vectorizer = CountVectorizer(max_features=size, ngram_range=(1, 1))

    return (
        pd.DataFrame(
            vectorizer.fit_transform(df_["language"]).toarray(),
            columns=[f"language_{str(f)}" for f in vectorizer.get_feature_names()],
        )
        .assign(user_id=lambda x: df_["user_id"].tolist())
        .groupby(by="user_id", as_index=False)
        .sum()
        .merge(
            df_.groupby(by="user_id", as_index=False).agg(
                total_languages=("language", "nunique")
            ),
            on=["user_id"],
            how="left",
        )
    )

def load_school(path: str, size: int = 20, exact_match: bool = True) -> pd.DataFrame:

    df_ = pd.read_csv(path)[['user_id', 'school_name']]

    if exact_match:
        most_freq_schools = df_["school_name"].value_counts()[:size].keys().tolist()
        for school in tqdm(most_freq_schools):
            df_[f"school_name_{school}"] = df_["school_name"].apply(lambda x: 1 if school == x else 0)

        return (
            df_.drop(columns=["school_name"], axis=1)
            .groupby(by=["user_id"], as_index=False)
            .sum().merge(
                df_.groupby(by="user_id", as_index=False).agg(
                    total_education=("school_name", "count")
                ),
                on=["user_id"],
                how="left",
            )
        )
    
    else:
        vectorizer = CountVectorizer(
            max_features=size,
            stop_words=stopwords.words("english"),
            ngram_range=(1, 3),
        )

        return (
            pd.DataFrame(
                vectorizer.fit_transform(df_["school_name"]).toarray(),
                columns=[f"school_name_{str(f)}" for f in vectorizer.get_feature_names()],
            )
            .assign(user_id=df_["user_id"].tolist())
            .groupby(by="user_id", as_index=False)
            .sum()
            .merge(
                df_.groupby(by="user_id", as_index=False).agg(
                    total_education=("school_name", "count")
                ),
                on=["user_id"],
                how="left",
            )
        )
        
def load_degree(path: str, size: int = 20, exact_match: bool = True) -> pd.DataFrame:

    df_ = pd.read_csv(path)[['user_id', 'degree']].fillna('')

    if exact_match:
        most_freq_degrees = df_.loc[df_['degree'] != '', 'degree'].value_counts()[:size].keys().tolist()
        for degree in tqdm(most_freq_degrees):
            df_[f"degree_{degree}"] = df_["degree"].apply(lambda x: 1 if degree == x else 0)

        return (
            df_.drop(columns=["degree"], axis=1)
            .groupby(by=["user_id"], as_index=False)
            .sum()
        )
    
    else:
        vectorizer = CountVectorizer(
            max_features=size,
            stop_words=stopwords.words("english"),
            ngram_range=(1, 2),
        )

        return (
            pd.DataFrame(
                vectorizer.fit_transform(df_["degree"]).toarray(),
                columns=[f"degree_{str(f)}" for f in vectorizer.get_feature_names()],
            )
            .assign(user_id=df_["user_id"].tolist())
            .groupby(by="user_id", as_index=False)
            .sum()
        )
        
def load_study(path: str, size: int = 20, exact_match: bool = True) -> pd.DataFrame:

    df_ = pd.read_csv(path)[['user_id', 'fields_of_study']].fillna('')

    if exact_match:
        most_freq_studies = df_.loc[df_['fields_of_study'] != '', 'fields_of_study'].value_counts()[:size].keys().tolist()
        for study in tqdm(most_freq_studies):
            df_[f"fields_of_study_{study}"] = df_["fields_of_study"].apply(lambda x: 1 if study == x else 0)

        return (
            df_.drop(columns=["fields_of_study"], axis=1)
            .groupby(by=["user_id"], as_index=False)
            .sum()
        )
    
    else:
        vectorizer = CountVectorizer(
            max_features=size,
            stop_words=stopwords.words("english"),
            ngram_range=(1, 3),
        )

        return (
            pd.DataFrame(
                vectorizer.fit_transform(df_["fields_of_study"]).toarray(),
                columns=[f"fields_of_study_{str(f)}" for f in vectorizer.get_feature_names()],
            )
            .assign(user_id=df_["user_id"].tolist())
            .groupby(by="user_id", as_index=False)
            .sum()
        )

def load_work_experiences(path: str) -> pd.DataFrame:

    df_ = pd.read_csv(path)
    tr_cities = load_tr_cities()
    df_["start_date"] = pd.to_datetime(
        df_["start_year_month"].apply(lambda x: "-".join([str(x)[:4], str(x)[4:]]))
    )
    df_ = df_.drop(columns=["start_year_month"], axis=1)
    df_.loc[
        df_["location"].astype(str).str.contains("Kahraman Maras"), "location"
    ] = "Kahramanmaras, Turkey"
    df_.loc[
        df_["location"].astype(str).str.contains("Şanliurfa"), "location"
    ] = "Sanliurfa, Turkey"
    df_.loc[
        df_["location"].astype(str).str.contains("İçel"), "location"
    ] = "Mersin, Turkey"
    df_.loc[
        df_["location"].astype(str).str.contains("Afyon"), "location"
    ] = "Afyonkarahisar, Turkey"
    df_["location"] = df_["location"].apply(
        lambda x: str(x).replace("Türkiye", "Turkey")
    )
    df_["location"] = df_["location"].apply(lambda x: x.upper().strip())
    df_["location"] = df_["location"].apply(lambda x: translation(str(x)))
    for city in tr_cities:
        df_["location"] = df_["location"].apply(lambda x: city if city in x else x)

    df_ = (
        df_.loc[df_["start_date"].dt.year != 2019]
        .sort_values(by=["user_id", "start_date"])
        .reset_index(drop=True)
    )
    df_ = df_.drop_duplicates(subset=["user_id", "company_id"])
    df_["quit_date"] = df_.groupby("user_id")["start_date"].shift(-1)
    df_["days_to_quit"] = (df_["quit_date"] - df_["start_date"]).apply(
        lambda x: np.nan if str(x).split()[0] == "NaT" else int(str(x).split()[0])
    )

    emp_df = (
        df_.groupby(by="user_id", as_index=False)
        .agg(
            employee_lifetime=(
                "start_date",
                lambda x: int(str(pd.to_datetime("2019-01-01") - x.min()).split()[0]),
            ),
            employee_last_experience=(
                "start_date",
                lambda x: int(str(pd.to_datetime("2019-01-01") - x.max()).split()[0]),
            ),
            employee_total_experience=(
                "start_date",
                lambda x: int(str(x.max() - x.min()).split()[0]),
            ),
            employee_last_days_to_quit=("days_to_quit", "last"),
            employee_min_days_to_quit=("days_to_quit", "min"),
            employee_max_days_to_quit=("days_to_quit", "max"),
            employee_std_days_to_quit=("days_to_quit", "std"),
            employee_med_days_to_quit=("days_to_quit", "median"),
            employee_last_experience_month=("start_date", lambda x: x.max().month),
            employee_last_experience_year=("start_date", lambda x: x.max().year),
            # employee_first_experience_month=("start_date", lambda x: x.min().month),
            employee_first_experience_year=("start_date", lambda x: x.min().year),
            employee_nunique_company=("company_id", "nunique"),
            # employee_nunique_location = ('location', 'nunique'),
            # employee_last_location=('location', 'last'),
            company_id=("company_id", "last"),
        )
        .assign(
            employee_avg_days_to_quit=lambda x: x.employee_lifetime
            / x.employee_nunique_company,
            employee_last_experience_month_sin=lambda x: np.sin(
                2 * np.pi * x.employee_last_experience_month / 12
            ),
            employee_last_experience_month_cos=lambda x: np.cos(
                2 * np.pi * x.employee_last_experience_month / 12
            ),
        )
    )

    emp_df = emp_df.merge(
        df_[["user_id"]]
        .drop_duplicates()
        .merge(
            df_.loc[df_["start_date"].dt.year == 2018]
            .groupby(by="user_id", as_index=False)
            .agg(company_count_2018=("company_id", "count")),
            on=["user_id"],
            how="left",
        )
        .fillna({"company_count_2018": 0}),
        on=["user_id"],
        how="left",
    )
    
    emp_df = emp_df.merge(
        df_[["user_id"]]
        .drop_duplicates()
        .merge(
            df_.loc[df_["start_date"].dt.year == 2017]
            .groupby(by="user_id", as_index=False)
            .agg(company_count_2017=("company_id", "count")),
            on=["user_id"],
            how="left",
        )
        .fillna({"company_count_2017": 0}),
        on=["user_id"],
        how="left",
    )

    comp_df = df_.groupby(by="company_id", as_index=False).agg(
        company_avg_days_to_quit=("days_to_quit", "mean"),
        company_std_days_to_quit=("days_to_quit", "std"),
        company_max_days_to_quit=("days_to_quit", "max"),
        # company_min_days_to_quit=("days_to_quit", "min"),
        company_med_days_to_quit=("days_to_quit", "median"),
        company_skew_days_to_quit=("days_to_quit", "skew"),
        company_nunique_employees=("user_id", "nunique"),
        # company_nunique_location = ('location', 'nunique'),
        company_lifetime=(
            "start_date",
            lambda x: int(str(pd.to_datetime("2019-01-01") - x.min()).split()[0]),
        ),
        company_last_hire=(
            "start_date",
            lambda x: int(str(pd.to_datetime("2019-01-01") - x.max()).split()[0]),
        ),
    )

    return emp_df.merge(comp_df, on=["company_id"], how="left").assign(
        avg_days_to_quit_diff=lambda x: x.company_avg_days_to_quit
        - x.employee_avg_days_to_quit,
        avg_days_to_quit_ratio=lambda x: x.company_avg_days_to_quit
        / x.employee_avg_days_to_quit,
        company_hire_ratio=lambda x: x.company_lifetime / x.company_nunique_employees,
    )


        

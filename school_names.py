import pandas as pd

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
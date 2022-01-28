import pandas as pd
data_path="Intermediate_ml\AER_credit_card_data.csv"
data=pd.read_csv(data_path,true_values=['yes'],false_values=['no'])#jika tidak dirubah kedalam bolean yes or no nya maka akan menyebabkan adanya data yg bersifat categorical
#select target
y=data.card
#data predictor
X=data.drop(['card'],axis=1)
print("Number row pada data set:",X.shape)#hasil dari shape(baris,kolom)
print(X.head())
print(y.head())
#untuk memvalidasi data testing digunakan sebuah metode berupa cv(cross validation)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
"""
# Karena tidak ada preprocessing, kita tidak
# membutuhkan pipeline (tetap digunakan sebagai praktik terbaik!)
"""
my_pipeline=make_pipeline(RandomForestClassifier(n_estimators=100))#tidak perlu preporcessing karena tidak ada missing value dan tidak ada sebuah jenis data categorical
cv_scores=cross_val_score(my_pipeline, X,y,
                          cv=5,scoring='accuracy')
print("Cross-validation accuracy : {}".format(cv_scores))
print("Cross-validation accuracy : %f" %cv_scores.mean())
#rata rata akurasi => 0.980294->98%
#apakah data ini mengalami data leagkage
"""
Card=>1 jika card di terima 0 jika card tidak diterima
reports=>jumlah laporan mayoritas/utama
age=>umur
income=>yearly income(devided by 10,000)
share=>Rasio pengeluaran kartu kredit bulanan terhadap pendapatan tahunan
expenditure=>Pengeluaran kartu kredit bulanan rata-rata
owner=>1 jika memiliki rumah , 0 jika tidak memiliki rumah
selfempl=>1 jika selftemployed 9 jika tidak
dependents=>1+jmlh_tanggungan 
months=>bulan tinggal di alamat skrng
majorcards=>jumlah dari nomber kartu kredit utama yang dimiliki 
activate=>jumlah activate kartu kredit

apakah pengeluaran berarti pengeluaran untuk kartu kredit (expenditure)  atau untuk kartu yang digunakan sebelum  membayar?

"""
#Pada titik ini, perbandingan data dasar bisa sangat membantu:
expediture_carholders=X.expenditure[y]
expediture_noncarholders=X.expenditure[~y]
print("Fraksi yang memegang kartu kredit dan tidak ada pengeluaran \n{}".format((expediture_carholders==0).mean()))
#0.020527859237536656=>ada kemungkinan bahwa card ditunjukan untuk kartu kredit
print("Fraksi yang tidak memegang kartu kredit dan tidak ada pengeluaran\n{}".format((expediture_noncarholders==0).mean()))
#1.0 =>ada kemungkinan bahwa card ditunjukan untuk kartu yang digunakan untuk pengeluaran kartu kredit sebelum membayar
    #=>target lackage->data prediksi berasal dari data predictor, jika tidak ada expenditure (prediktor)maka tidak memiliki kartu(prediksi)
    #->jika ada expenditure bisa dinyatakan memiliki card namun ada kemungkinan 2% yang tidak memiliki expenditure juga memiliki kartu
    #pada hasil 100% fraksi yang tidak pegang kartu dan tidak ada pengeluaran 

#kejanggalan
"""
->pada share pada data yang tidak memiliki pengeluaran seharusnya tidak ada rasio pengeluaran terhadap pendapatan tahunan atau berisikan sebuah pendapatan tahunan dikarenakan
tidak ada kartu kredit maka tidak ada rasio pengeluaran
->pada expediture juga bermasalah dengan kartu kredit bahwa orng yang memiliki kartu kredit seharusnya memiliki pengeluaran
->pada data major card=>orng yang tidak memiliki kartu kredit seharusnya tidak memiliki kartu utama
->pada  data activate juga mengalami kebocoran yg mana orng yang tidak memiliki sebuah card
"""
potential_leaks=['expenditure','share','majorcards','active']
X2=X.drop(potential_leaks,axis=1)
print(X2)
# Evaluate the model with leaky predictors removed
cv_scores=cross_val_score(my_pipeline, X,y,cv=5,scoring='accuracy')
print(cv_scores.mean()*100,"%")
#97.87734762069363 %
"""
sangat tinggi ->nice
Kebocoran data bisa menjadi kesalahan jutaan dolar di banyak aplikasi ilmu data. Pemisahan data pelatihan dan validasi yang cermat dapat mencegah kontaminasi uji-latihan,
dan pipeline dapat membantu menerapkan pemisahan ini. Demikian juga, kombinasi kehati-hatian, akal sehat, dan eksplorasi data dapat membantu mengidentifikasi kebocoran target.
"""
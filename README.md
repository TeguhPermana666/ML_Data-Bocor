# ML_Data-Bocor
data bocor sering kali sangat berpengaruh terhadap kualitas model dalam implementasinya
terkadang kita tertipu dengan hasil training yang sangat baik namum jika dikalkulasi evaluasi modelnya memiliki nilai yang buruk untuk di produksi
dikarenakan ada dua penyebab:
->target lackage => yang terjadi karena nilai dari y beberasal dari x yang mana kadang tidak selaras dengan kenyataan misal dengan
pada data card yang 0=>tidak memiliki kartu terhadap penghasilan apakah seseorang yang memiliki kartu ini harus memiliki pengeluaran 
pada dta sebagian besr jika punya kartu mmiliki pengeluaran tapi ada 2% yang memiliki kartu tapi tidak ada pengeluaran maka dari itu 
terjadi sebuah target lackage yang mana data pada predictor tidak ditemukan pada saat prediksi yang mana dapat diartikan bahwa apakah prediksi ini merupakan sebuah pengeluaran
kartu kredit atau pengeluaran pada saat sebelum melakukan pembelian barang dengan kartu kredit
=>cara nya adalah mengilangkan sebuah column yang mengalami keboroan dengan y
1.pengeluaran ->dikarenakan adanya kebocoran jika diartikan bahwa data ini adalah pengeluaran kartu kredit bahwa adanya  orng yang memiliki kartu kredit dengan tidak selalu melakukan pengeluaran yang mana setidaknya pasti orang tersebut melakukan pengeluaran jika menggunakan kartu kredit,
2.selisih antara pengeluaran dengan pendapatan yang mana orng yang tidak memiliki kartu kredit seharusnya tidak memiliki pencatatan pengeluaran
3.kepemilikikan kartu kredit utama yang mana orng yang tidak punya kartu seharusnya tidak memiliki kartu kredit
4.pada activate card seharusnya orng yang tidak ada kartu tidak memiliki kartu yg aktif

->train test contamination
yang mana adanya kontaminasi data validasi dengan train
train digunakan untuk model dalam mengenal data sesuai pemodelan yang dilakukan. 
validasi digunakan untuk mengetst model dalam mengenali data dengan memberikan nilai data yang berbeda tetapi bentuk data dalam pelatihan sama

kontaminasi terjadi jika sebuah data yang dikenal dilakukan sebuah pengujian validasi yang mana pada code kali ini dilakukan pengujian Cross validation dengan cv=5 kali dan
5 folds, yang mana masing masing folds akan di validasi dengan menyilang.

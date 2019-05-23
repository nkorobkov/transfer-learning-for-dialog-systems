### Надо сделать  еще 
 

### Можно сделать еще

- проверить для какого-нибдуь  еще языка такую же ботву. 
- по каким запросам больше всего ошибок
- 

### Вопросы к которым подготовиться перед защитой

- что такое адам оптимайзер
- что такое негатив лог  лайклихуд
- Как работает  лстм, gru еще разок
- можно ли шарашить для других совсем языков такие же штуки
- Критерий остановки
- как решать линейную систему методом наименьших квадратов


### Статьи
https://code.fb.com/ai-research/laser-multilingual-sentence-embeddings/

https://towardsdatascience.com/grus-and-lstm-s-741709a9b9b1

### Результаты

RNN2label fastText-->label -- 96.7% (отлично, значит мы выбили максимум на трансфере)

sw
gru
rnn - nn 94.2%   f1=  0.939
[0.97511521, 0.97435897, 0.93165468, 0.97506925, 0.83315393,0.96577243, 0.92418773]
rnn - useclass 96.4%   f1= 0.964
[0.98795181, 0.9912892 , 0.94357077, 0.98978644, 0.90146341,
 0.97844424, 0.95692026]
 {0: 'BookRestaurant',
  1: 'GetWeather',
  2: 'SearchScreeningEvent',
  3: 'RateBook',
  4: 'SearchCreativeWork',
  5: 'AddToPlaylist',
  6: 'PlayMusic'})

finnish

fasttext --> avg --> label = 84.4% (cv5)
fasttext --> weighted avg --> label = 94.3% (cv5)

fasttext --> avg --> USE --> NN = 88.6% (cv5)
fasttext --> weighted avg --> USE --> NN = 91.8% (cv5)


fasttext --RNN--> USE --> NN = 93.7% (cv5)  f1 = 0.936
fasttext --RNN--> USE --> USEclass = 96.1% (cv5)  f1 = 0.964





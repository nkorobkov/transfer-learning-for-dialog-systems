@startuml
'https://www.planttext.com/
skinparam ArrowColor darkslategrey
skinparam shadowing  false
skinparam card {
  borderColor black
  backgroundColor white
  
}
skinparam agent {
  borderColor black
  backgroundColor cornsilk
  ArrowColor black
}
skinparam node {
  borderColor black
  backgroundColor white
  ArrowColor black
}
skinparam queue {
  borderColor black
  backgroundColor cornsilk
  ArrowColor black
}
skinparam rectangle {
	BorderColor black
	    BackgroundColor White
}
skinparam interface {
	BorderColor darkslategrey
	    BackgroundColor cornsilk
}


node node1 [
Матрица fastText векторов
для слов в предложении
<b>300 x 33
]

node conv1 [
128 x 32
]
node conv2[
128 x 31
]
node conv3[
128 x 29
]

rectangle mconv1 [
128 
]
rectangle mconv2[
128 
]
rectangle mconv3[
128 
]

rectangle rnn1[
256 
]
rectangle rnn2[
256 
]

interface concatenate[
Склейка
]

rectangle predrop[
896 
]

agent fcl[
Полносвязный
слой
]

rectangle use[
Предсказанный USE вектор
<b>512 
]



queue "       Прямая RNN     "{
    card GRU1[
    GRU/LSTM
    <b>256
    ]
    
    GRU1 ..> GRU1
}
queue "       Обрантая RNN     "{
    card GRU2[
    GRU/LSTM
    <b>256
    ]
    
    GRU2 ..> GRU2
}

node1 --> GRU1 
node1 --> GRU2

GRU2 --> rnn2 
GRU1 --> rnn1  

node1 #--> conv3 : Cвертка 300x5
node1 #--> conv2 : Cвертка 300x3
node1 #--> conv1 : Cвертка 300x2

conv3 --> mconv3 : Mаксимум
conv2 --> mconv2 : Mаксимум
conv1 --> mconv1 : Mаксимум


rnn2 --> concatenate
rnn1 --> concatenate

mconv1 --> concatenate
mconv2 --> concatenate
mconv3 --> concatenate

concatenate -> predrop
predrop -> fcl: Дропаут  (0.35)
fcl --> use



@enduml
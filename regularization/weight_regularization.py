from keras import regularizers
from keras.model import Sequential
from keras.layers import Dense


## 정규화 기법 사용 예(L2)
model = Sequential()
model.add(Dense(16, kernel_regularizer = regularizers.l2(0.001),
                        activation='relu', input_shape=(10000,)))
model.add(Dense(16, kernel_regularizer = regularizers.l2(0.001),
                        activation='relu', input_shape=(10000,)))
model.add(Dense(1,activation = 'sigmoid'))                                               

## 정규화 기법
regularizers.l1(0.001)   # l1 규제

regularizers.l2(0.001)   # l2 규제 - weight decay=가중치 감쇠라고도함

regularizers.l1_l2(l1=0.001, l2=0.001)  # l1과 l2규제 병행

# 위의 함수들은 regularizers.L1L2 클래스의 객체를 반환하는 함수이다. 이 대신 
# regularizers.L1L2(l2=0.001)을 사용해도 l2 규제를 사용 할 수 있음

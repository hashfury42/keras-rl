#!/usr/bin/python
# -*- coding: utf-8 -*-
# import numpy as np
# from keras.utils import to_categorical
#
#
# data = np.array([1, 9, 3, ])
# print(data)
#
#
# def encode(data):
#     print('Shape of data (BEFORE encode): %s' % str(data.shape))
#     encoded = to_categorical(data)
#     print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
#     return encoded
#
#
# encoded_data = encode(data)
# print(encoded_data)
#
#
# def decode(datum):
#     return np.argmax(datum)
#
#
# for i in range(encoded_data.shape[0]):
#     print("\n")
#     datum = encoded_data[i]
#     print('index: %d' % i)
#     print('encoded datum: %s' % datum)
#     decoded_datum = decode(encoded_data[i])
#     print('decoded datum: %s' % decoded_datum)
#     print("\n")

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)

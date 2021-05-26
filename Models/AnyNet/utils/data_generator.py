import os

dirname = os.path.dirname(__file__)
train_output = open(os.path.join(dirname, 'train.txt'), 'wb')
validation_output = open(os.path.join(dirname, 'val.txt'), 'wb')

for i in range(6000):
    train_output.write((str(i).zfill(6)+'\n').encode())

for i in range(6000, 7481):
    validation_output.write((str(i).zfill(6)+'\n').encode())
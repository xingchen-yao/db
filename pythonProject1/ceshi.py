import mmcv

dict_data = mmcv.dict_from_file('abc.txt')
print('dict_data = ', dict_data)

list_data = mmcv.list_from_file('abc.txt')
print('list_data =', list_data)

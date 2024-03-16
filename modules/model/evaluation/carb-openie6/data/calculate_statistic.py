with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/data/gold/test.tsv', 'r', encoding='utf-8') as f_in:
    sent_dict = {}
    lines = f_in.readlines()
    for line in lines:
        sent = line.split('\t')[0]
        if sent not in sent_dict.keys():
            sent_dict[sent] = 1
        else:
            sent_dict[sent] += 1
    
# print(sent_dict)
    
    value_dict = {}
    for key, value in sent_dict.items():
        if value not in value_dict.keys():
            value_dict[value] = 1
        else:
            value_dict[value] += 1
    
    print(value_dict)

lsoie_test = {2: 2726, 3: 1218, 1: 4376, 5: 154, 6: 64, 4: 468, 7: 33, 8: 15, 11: 1, 9: 2, 57: 1, 25: 1, 10: 2, 12: 1}

a, b = 0, 0
for key, value in lsoie_test.items():
    if key < 3:
        a += value
    else:
        b += value

print(a, b)
print( a / (a+b) )
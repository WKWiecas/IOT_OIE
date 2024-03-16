# 把文件变成carb的输入形式

a = 'Barra , 51 , will be the first woman to lead a firm in the American auto industry . 	lead	the first woman	a firm in the American auto industry'
import json

with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/multiOIE_data/Re-OIE2016-Spanish-Binary.json', 'r', encoding='utf-8') as f:
    with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/multiOIE_data/MultiOIE_test_sp.tsv', 'w', encoding='utf-8') as f_out:
        data = json.load(f)
        for sent, indexs in data.items():
            print(sent)
            print(indexs)
            indexs = list(indexs)
            for index in indexs:
                arg0 = index['arg0']
                pred = index['pred']
                arg1 = index['arg1']
                arg2 = index['arg2']
                arg3 = index['arg3']

                if not arg0 or not pred:
                    continue

                output = sent + '\t' + pred + '\t' + arg0 + '\t' + arg1
                if arg2:
                    output = output + '\t' + arg2
                if arg3:
                    output = output + '\t' + arg3
                output = output + '\n'
                f_out.write(output)


with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/multiOIE_data/Re-OIE2016-Spanish-Binary.json', 'r', encoding='utf-8') as f:
    with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/multiOIE_data/MultiOIE_sentences_sp.txt', 'w', encoding='utf-8') as f_out:
        data = json.load(f)
        for sent, indexs in data.items():
            output = sent
            output = output + '\n'
            f_out.write(output)
                
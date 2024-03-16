

'''把lsoie测试集变成carb的格式'''
with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/lsoie_data/eval.oie', 'r', encoding='utf-8') as f_in:
    wiki_data = f_in.readlines()
    print(wiki_data[1])
    print(wiki_data[1].split('\t'))
    wiki_sentences = []
    for wiki_one in wiki_data:
        wiki_sent = wiki_one.split('\t')[0]
        if wiki_sent not in wiki_sentences:
            wiki_sentences.append(wiki_sent)

with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/lsoie_data/science_eval.oie', 'r', encoding='utf-8') as f_in:
    science_data = f_in.readlines()
    print(science_data[1])
    print(science_data[1].split('\t'))
    science_sentences = []
    for science_one in science_data:
        science_sent = science_one.split('\t')[0]
        if science_sent not in science_sentences:
            science_sentences.append(science_sent)


# with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/lsoie_data/lsoie_test.tst', 'w', encoding='utf-8') as f_out:
#     data = wiki_data + science_data
#     f_out.writelines(data)

with open('/mnt/workspace/wkw_work/DetIE-main/modules/model/evaluation/carb-openie6/lsoie_data/lsoie_sentences.txt', 'w', encoding='utf-8') as f_out:
    sentences = wiki_sentences + science_sentences
    for sent in sentences:
        f_out.write(sent + '\n')
import stanza
import json
print("start")
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

print("start_2")
# for gpt result
file_list = ['path to gpt result']
             

for file in file_list:
    with open(file) as f:
        print(file) 
        result = json.load(f)
        gt_total_word = 0
        gt_total_entity = 0
        gt_entity_set_list = []
        for item in result['gt']:
            doc_gt = nlp(item)
            entity_set = set()
            for sentence in doc_gt.sentences:
                for token in sentence.tokens:
                    gt_total_word += 1
                    if token.ner != 'O':
                        gt_total_entity += 1
                        entity_set.add(token.text)
            gt_entity_set_list.append(entity_set)

        pred_total_word = 0
        pred_total_entity = 0
        pred_entity_set_list = []
        for item in result['pred']:
            doc_pred = nlp(item)
            entity_set = set()
            for sentence in doc_pred.sentences:
                for token in sentence.tokens:
                    pred_total_word += 1
                    if token.ner != 'O':
                        pred_total_entity += 1
                        entity_set.add(token.text)
            pred_entity_set_list.append(entity_set)


        correct_pred_count = 0
        for i in range(len(gt_entity_set_list)):
            correct_set = gt_entity_set_list[i] & pred_entity_set_list[i]
            correct_pred_count += len(correct_set)


        gt_nerr = gt_total_entity/gt_total_word
        pred_nerr = pred_total_entity/pred_total_word
        real_pred_nerr = correct_pred_count/gt_total_word
        print("gt_total_entity", end = '')
        print(gt_total_entity)
        print("gt_total_word", end = '')
        print(gt_total_word)
        print("pred_total_entity", end = '')

        print(pred_total_entity)
        print("pred_total_word", end = '')
        print(pred_total_word)
        print("correct_pred_count", end = '')
        print(correct_pred_count)
        print("gt_nerr", end = '')
        print(gt_nerr)
        print("pred_nerr", end = '')
        print(pred_nerr)
        print("real_pred_nerr", end = '')
        print(real_pred_nerr)

        ner = correct_pred_count/gt_total_entity
        print("ner")
        print(ner)

# for persona_data result
file_list = ['path to persona_data result']
             

for file in file_list:

    entity_set = set()
    total_entity = 0   
    with open(file, "r", encoding='UTF-8') as f:
        total_word = 0
        print("dev")
        out = f.read()
        out = json.loads(out)
        for element in out:
            sentences = ""
            for sentence in element["partner_persona"]:
                sentences += sentence
            for sentence in element["your_persona"]:
                sentences += sentence
            for sentence in element["conv"]:
                sentences += sentence
            doc_sentences = nlp(sentences)
            for sentence in doc_sentences.sentences:
                # print(str(sentence))
                for token in sentence.tokens:
                    # print(token.text)
                    # print(token.ner)
                    total_word += 1
                    if token.ner != 'O':
                        total_entity += 1
                        entity_set.add(token.text)
    print("unique entity total")
    print(len(entity_set))
    print("total_entity")
    print(total_entity)


# for RNN result
file_list = ['path to RNN result']

for file in file_list:
    with open(file) as f:
        print(file)
        result = json.load(f)

        for i in result:
            sentences = tokenizer.encode(i['input'])
            for word in sentences:
                decode_item = tokenizer.decode(word)

        gt_total_word = 0
        gt_total_entity = 0
        correct_pred_count = 0
        for i in result:
            decode_entity_set = set()
            gt_set = set()
            pred_set = set()
            doc_gt = nlp(i['input'])
            for sentence in doc_gt.sentences:
                for token in sentence.tokens:
                    if token.ner != 'O':
                        # print(token.text)
                        word_tokens = tokenizer.encode(token.text)
                        for word_token in word_tokens:
                            decode_token = tokenizer.decode(word_token)
                            decode_entity_set.add(decode_token)
            for item in i['gt']:
                gt_set.add(item)
            for item in i['pred']:
                pred_set.add(item)
            gt_total_word += len(gt_set)
            gt_total_entity += len(decode_entity_set)
            correct_set = decode_entity_set & pred_set
            correct_pred_count += len(correct_set)

        gt_nerr = gt_total_entity/gt_total_word
        real_pred_nerr = correct_pred_count/gt_total_word
        print("gt_total_word", end = '')
        print(gt_total_word)
        print("gt_total_entity", end = '')
        print(gt_total_entity)
        print("correct_pred_count", end = '')
        print(correct_pred_count)
        print("gt_nerr", end = '')
        print(gt_nerr)
        print("real_pred_nerr", end = '')
        print(real_pred_nerr)

        ner = correct_pred_count / gt_total_entity
        print("ner", end = '')
        print(ner)

        


# for NN result
file_list = ['path to NN result']
for file in file_list:
    with open(file) as f:
        print(file)
        result = json.load(f)

        for i in result:
            sentences = tokenizer.encode(i['input'])
            # print(sentences)
            for word in sentences:
                decode_item = tokenizer.decode(word)
                # print(decode_item)

        gt_total_word = 0
        gt_total_entity = 0
        correct_pred_count = 0
        for i in result:
            decode_entity_set = set()
            gt_set = set()
            pred_set = set()
            doc_gt = nlp(i['input'])
            for sentence in doc_gt.sentences:
                for token in sentence.tokens:
                    if token.ner != 'O':
                        # print(token.text)
                        word_tokens = tokenizer.encode(token.text)
                        for word_token in word_tokens:
                            decode_token = tokenizer.decode(word_token)
                            decode_entity_set.add(decode_token)
            for item in i['gt']:
                gt_set.add(item)
            for item in i['pred']:
                pred_set.add(item)
            # print(decode_entity_set)
            gt_total_word += len(gt_set)
            gt_total_entity += len(decode_entity_set)
            correct_set = decode_entity_set & pred_set
            '''
            if len(correct_set) != 0:
                print(i)
                print("decode_entity_set")
                print(decode_entity_set)
                print("pred_set")
                print(pred_set)
                print("correct_set")
                print(correct_set)
            '''
            correct_pred_count += len(correct_set)

        gt_nerr = gt_total_entity/gt_total_word
        real_pred_nerr = correct_pred_count/gt_total_word
        print("gt_total_word", end = '')
        print(gt_total_word)
        print("gt_total_entity", end = '')
        print(gt_total_entity)
        print("correct_pred_count", end = '')
        print(correct_pred_count)
        print("gt_nerr", end = '')
        print(gt_nerr)
        print("real_pred_nerr", end = '')
        print(real_pred_nerr)

        ner = correct_pred_count / gt_total_entity
        print("ner", end = '')
        print(ner)

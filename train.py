from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from MarianMTPhoneticModel import MarianMTPhoneticModel
from generate_phonetic_feature_vectors import *
from tqdm import tqdm 
import re 
import pickle
import pandas as pd
import torch 
from KoG2P.g2p import runKoG2P
import hgtk
import random
from tqdm import tqdm
tqdm.pandas()


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
model = MarianMTPhoneticModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

'''
ko_vocab_dict = {}
for vocab in tqdm(tokenizer.encoder):
    if re.findall('[^가-힣▁]', vocab) or vocab == '▁':
        continue 
    ko_vocab_dict[vocab] = tokenizer.encoder[vocab]

len(ko_vocab_dict) # 31307

list(ko_vocab_dict.keys())[:10] # ['▁들', '▁이', '▁을', '▁의', '▁은', '▁에', '▁가', '▁는', '▁를', '▁그']

vocab2phonvec = {}
for vocab in tqdm(ko_vocab_dict):
    vocab2phonvec[ko_vocab_dict[vocab]] = word2phonvec(vocab)

with open('vocab2phonvec.pickle', 'wb') as fp:
    pickle.dump(vocab2phonvec, fp)
'''


with open ('feat2vec.pickle', 'rb') as fp:
    feat2vec = pickle.load(fp)
    
with open ('vocab2phonvec.pickle', 'rb') as fp:
    vocab2phonvec = pickle.load(fp)

ipadf = pd.read_csv('ipa_feats.csv')
ipadf.features = ipadf.features.apply(lambda x: tuple(x.split()))
phone_feature_map = dict(ipadf.values)
phone_feature_map['^'] = tuple(['bgn']) 
phone_feature_map['$'] = tuple(['end']) 



def word2phonvec(word):
    phones = '^ ' if word.startswith('▁') else ''
    phones += runKoG2P(word.replace('▁', ''), 'KoG2P/rulebook.txt') 

    features = Counter(feature_bigrams(phones.split(), phone_feature_map, bgn_end=False))
    word_phonvec = np.zeros(512)
    count = 0
    for feature in features:
        word_phonvec += feat2vec[feature] * features[feature]
        count += features[feature]

    word_phonvec = word_phonvec / count if count > 0 else word_phonvec
    
    return word_phonvec


def get_phonetic_embedding(input_ids, sent):
    embed_phonetic = []
    if tokenizer.unk_token_id not in input_ids:
        for token_id in input_ids:
            if int(token_id) not in vocab2phonvec:
                embed_phonetic.append(torch.zeros(512))
            else:
                embed_phonetic.append(torch.from_numpy(vocab2phonvec[int(token_id)]))
    else:
        tokenized_sent = tokenizer.tokenize(sent)
        for token_id, token in zip(input_ids, tokenized_sent + ['<pad>'] * (len(input_ids) - len(tokenized_sent))):
            if token_id == tokenizer.unk_token_id:
                embed_phonetic.append(torch.from_numpy(word2phonvec(token)))
            elif int(token_id) in vocab2phonvec:
                embed_phonetic.append(torch.from_numpy(vocab2phonvec[int(token_id)]))
            else:
                embed_phonetic.append(torch.zeros(512))
                
    return torch.stack(embed_phonetic).float()



phon_dict = {'ㄱ':['ㄲ','ㅋ'], 'ㄷ':['ㄸ','ㅌ'], 'ㅂ':['ㅃ','ㅍ'], 'ㅅ':['ㅆ'], 'ㅈ':['ㅉ','ㅊ']}

def add_noise(eumjeol):
    result = []
    for i,phoneme in enumerate(hgtk.letter.decompose(eumjeol)):
        # 종성은 변환하지 않음
        if i == 2 or phoneme not in phon_dict:
            result.append(phoneme)
        else:
            result.append(random.choice(phon_dict[phoneme]))
            
    if len(result) == 2:
        return hgtk.letter.compose(result[0], result[1])
    else:
        try:
            return hgtk.letter.compose(result[0], result[1], result[2])
        except:
            print(eumjeol, result)
            return eumjeol


# -

def build_noise_data(sent):
    random_eojeols = random.choices(sent.split(), k=2)
    for random_eojeol in random_eojeols:
        if hgtk.checker.is_hangul(random_eojeol):
            noised_eojeol = ''.join([add_noise(eumjeol) for eumjeol in random_eojeol])
            sent = sent.replace(random_eojeol, noised_eojeol)
    return sent 


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ko_sents=None, en_sents=None, tokenizer=None, label_yn=True):
        self.label_yn = label_yn
        inputs = tokenizer(ko_sents, return_tensors="pt", padding=True, truncation=True)

        self.input_ids = []
        self.embed_phonetic = []
        for input_id, sent in zip(inputs['input_ids'], ko_sents):
            input_id = input_id.clone().detach()
            emb_pho = get_phonetic_embedding(input_id, sent)
            self.input_ids.append(input_id)
            self.embed_phonetic.append(emb_pho)
            assert len(input_id) == len(emb_pho)
        self.attention_mask = [i.clone().detach() for i in inputs['attention_mask']]
        
        if self.label_yn:
            labels = tokenizer(en_sents, return_tensors="pt", padding=True, truncation=True).input_ids
            self.labels = [i.clone().detach() for i in labels]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.label_yn:
            return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], \
                    'embed_phonetic': self.embed_phonetic[idx], 'labels': self.labels[idx]}
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], \
                    'embed_phonetic': self.embed_phonetic[idx]}


    
df = pd.read_csv('mt-data.csv')
df['noised_ko'] = df.ko.progress_apply(lambda x: build_noise_data(x))

msk = np.random.rand(len(df)) < 0.8

train_df = df[msk]
eval_df = df[~msk]

train_df.to_csv('train.csv', index=False, encoding='utf-8')
eval_df.to_csv('eval.csv', index=False, encoding='utf-8')


train_data = CustomDataset(train_df.noised_ko.tolist(), train_df.en.tolist(), tokenizer)
eval_data = CustomDataset(eval_df.noised_ko.tolist(), eval_df.en.tolist(), tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# freeze original Bart Encoder, Decoder except the last layer
# for param in model.model.encoder.layers[:-2].parameters():
#     param.requires_grad = False
# for param in model.model.decoder.layers[:-1].parameters():
#    param.requires_grad = False


# +

training_args = TrainingArguments(
    'PhoneticTranslation-noise2-epoch15-ckpt',
    # load_best_model_at_end = True,
    num_train_epochs=15,  # total # of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    # weight_decay=0.01,  # strength of weight decay
    # eval_accumulation_steps=1,
    save_steps = 999999999999999 # checkpoint 중간에 저장하지 말라고
)
# -

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_data,  # training dataset
    eval_dataset=eval_data,
    # compute_metrics= compute_metrics
)

trainer.train()

trainer.evaluate()

model.save_pretrained('./PhoneticTranslationModel-noise2-epoch15')




def get_inference_batch(ko_sents):
        inputs = tokenizer(ko_sents, return_tensors="pt", padding=True, truncation=True)

        input_ids = []
        embed_phonetic = []
        for input_id, sent in zip(inputs['input_ids'], ko_sents):
            input_ids.append(input_id.clone().detach())
            embed_phonetic.append(get_phonetic_embedding(input_id, sent))
        attention_mask = [i.clone().detach() for i in inputs['attention_mask']]
        
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        embed_phonetic = torch.stack(embed_phonetic).float()
            
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'embed_phonetic':embed_phonetic}


sample_texts = ['박물관이 어디 있나요?', "빡물관이 어디 있나요?", "빡물콴이 어디 있나요?", "완쩐 뼐루에요"]
batch = get_inference_batch(sample_texts)
generated_ids = model.generate(**batch)
translated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
translated_texts

for sample, result in zip(sample_texts, translated_texts):
    print(f'{sample} --> {result}')





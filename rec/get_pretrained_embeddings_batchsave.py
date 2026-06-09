import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel

if __name__ == "__main__":
    precision = 8
    batch_size = 200
    data_set_name = 'Amazon_movies'
    pretrained_model_name = './llm_model'

    # Prepare PLM modle and tokenizers
    config = AutoConfig.from_pretrained(pretrained_model_name)
    hidden_size = config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    PLM = AutoModel.from_pretrained(pretrained_model_name).cuda()
    print("PLM initialized")

    # [ 'Amazon_movies', 'Enron', 'GDELT', 'Googlemap_CT', 'ICEWS1819', 'Stack_elec', 'Stack_english', 'Stack_ubuntu', 'Yelp']
    for data_set_name in ['Yelp']:
        print(data_set_name)
        edge_list = pd.read_csv('./DyLink_Datasets/' + data_set_name + '/edge_list.csv')
        num_node = max(edge_list['u'].max(), edge_list['i'].max())
        num_rel = edge_list['r'].max()

        # Prepare datasets
        rel_embeddings = [np.zeros(hidden_size)]
        rel_text_reader = pd.read_csv('./DyLink_Datasets/' + data_set_name + '/relation_text.csv', chunksize=batch_size)

        with tqdm(total = num_rel) as pbar:
            num_batch = 0
            num_pack = 0
            for batch in rel_text_reader:
                id_batch = batch['i'].tolist()
                text_batch = batch['text'].tolist()
                if 0 in id_batch:
                    id_batch = id_batch[1:]
                    text_batch = text_batch[1:]
                if np.nan in text_batch:
                    text_batch = ['NULL' if type(i) != str else i for i in text_batch]
                encoded_input = tokenizer(text_batch, padding = True, truncation=True, max_length=512 , return_tensors='pt')
                with torch.no_grad():
                    output = PLM(**encoded_input.to('cuda'))[1]
                    output = output.cpu()
                for i in range(len(output)):
                    # assert len(rel_embeddings) == id_batch[i]
                    rel_embeddings.append(np.around(output[i].numpy(), precision))
                    pbar.update(1)
                num_batch+=1

                if num_batch > 4000:
                    num_batch = 0
                    rel_embeddings = np.array(rel_embeddings)
                    np.save('./DyLink_Datasets/' + data_set_name + '/r_feat_'+str(num_pack)+'.npy', rel_embeddings)
                    num_pack += 1
                    rel_embeddings = []

        # rel_embeddings = np.array(rel_embeddings)
        # assert len(rel_embeddings) == num_rel + 1
        # print(rel_embeddings.shape)
        # np.save(data_set_name + '/r_feat.npy', rel_embeddings)


import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import codecs
from torch import nn
from torch import optim
import torch.nn.functional as F
import codecs

import transformers
from transformers import BertJapaneseTokenizer
from transformers.modeling_bert import  BertForMaskedLM
from transformers import BertModel, BertConfig, BertForPreTraining, BertForSequenceClassification, BertForTokenClassification


tag_list = ["00117",
            "00166",
            "00545",
            "00755",
            "01889",
            "02843",
            "02998",
            "05167",
            "05541",
            "08783",
            "09590",
            "09667",
            "10703",
            "14411",
            "15615",
            "17877",
            "20676",
            "21128",
            "22293",
            "24646",
            "26839",
            "27236",
            "31166",
            "31472",
            "31640",
            "34522",
            "34626",
            "35478",
            "35881",
            "37713",
            "40289",
            "40333",
            "40699",
            "41135",
            "41138",
            "41150",
            "41912",
            "43494",
            "44126",
            "46086",
            "47634",
            "48488",
            "49355",
            "49812",
            "50038",
            "51332",
            "51409",
            "51421",
            "52310",
            "52935"]

gogi_list = [4, 6, 9, 4, 3, 5, 4, 4, 4, 3, 3, 4, 3, 3, 3, 3, 3, 4, 4,  3, 4, 6, 3, 5, 5, 3, 6, 4, 3, 9, 5, 3, 5, 3, 4, 3, 3, 5, 3, 3, 3, 4, 5, 5, 6, 5, 5, 3, 5, 5]
# print(np.array(tag_list).shape)
# print(np.array(gogi_list).shape)


wsd_dict = dict(map(lambda x,y:[x,y], tag_list , gogi_list))

for tag , gogi in wsd_dict.items():

        def load_wsd_train_x():
            wsd_train_x = codecs.open('{}_train_sentence'.format(tag), mode='r', encoding='utf-8')
            line = wsd_train_x.readline()
            list1 = []
            while line:
                b = line[:].strip()
                list1.append(b)
                line = wsd_train_x.readline()
            return np.array(list1).reshape(50, 1)
            wsd_train_x.close()


        def load_wsd_test_x():
            wsd_test_x = codecs.open('{}_test_sentence'.format(tag), mode='r', encoding='utf-8')
            line = wsd_test_x.readline()
            list1 = []
            while line:
                b = line[:].strip()
                list1.append(b)
                line = wsd_test_x.readline()
            return np.array(list1).reshape(50, 1)
            wsd_test_x.close()


        def load_wsd_train_y():
            wsd_train_y = codecs.open('{}_train_target'.format(tag), mode='r', encoding='utf-8')
            line = wsd_train_y.readline()
            list1 = []
            while line:
                a = line.split()
                b = a[1:2]
                list1.append(b)
                line = wsd_train_y.readline()
            return (np.array(list1)).reshape(50, )
            wsd_train_y.close()


        def load_wsd_test_y():
            wsd_test_y = codecs.open('{}_test_target'.format(tag), mode='r', encoding='utf-8')
            line = wsd_test_y.readline()
            list1 = []
            while line:
                a = line.split()
                b = a[1:2]
                list1.append(b)
                line = wsd_test_y.readline()
            return (np.array(list1)).reshape(50, )
            wsd_test_y.close()


        wsd_train_x = load_wsd_train_x()
        wsd_test_x = load_wsd_test_x()

        wsd_train_y = load_wsd_train_y().astype(float)
        wsd_test_y = load_wsd_test_y().astype(float)

        max_epoch = 10
        train_size = 50
        batch_size = 10
        n_batch = train_size // batch_size

        gogi_num = gogi


        class DealDataSet(Dataset):

            def __init__(self, data, label, maxlen):

                self.data = data
                self.target = label

                self.tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')

                self.maxlen = maxlen

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):

                sentence = self.data[index]
                label = self.target[index]

                tokens = self.tokenizer.tokenize(str(sentence)[2:-2])

                tokens = ['[CLS]'] + tokens + ['[SEP]']

                if len(tokens) < self.maxlen:
                    tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding sentences
                else:
                    tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Prunning the list to be of specified max length

                tokens_ids = self.tokenizer.convert_tokens_to_ids(
                    tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
                tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

                # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
                attn_mask = (tokens_ids_tensor != 0).long()

                return tokens_ids_tensor, attn_mask, label


        dealTrainDataSet = DealDataSet(data=wsd_train_x, label=wsd_train_y, maxlen=128)
        dealTestDataSet = DealDataSet(data=wsd_test_x, label=wsd_test_y, maxlen=128)

        train_loader = DataLoader(dataset=dealTrainDataSet, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=dealTestDataSet, batch_size=batch_size, shuffle=False, num_workers=0)


        class WSDClassifier(nn.Module):

            def __init__(self, freeze_bert=True):
                super(WSDClassifier, self).__init__()

                self.bert_layer = BertModel.from_pretrained('bert-base-japanese-whole-word-masking')

                if freeze_bert:
                    for param in self.bert_layer.parameters():
                        param.requires_grad = False

                self.cls_layer = nn.Linear(768, gogi_num)

            def forward(self, seq, attn_masks):
                """
                Inputs:
                :param seq: Tensor of shape [B, T] containing token ids of sequences
                :param attn_masks: Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
                :return:
                """

                # Feed the input to BERT model to obtain contextualized representations.
                cont_reps, _ = self.bert_layer(seq, attn_masks)

                # Obtaining the representation of [CLS] head
                cls_rep = cont_reps[:, 0]

                # Feeding cls_rep to the classifier layer
                logits = self.cls_layer(cls_rep)

                return logits


        def train():
            best_acc = 0
            model = WSDClassifier(freeze_bert=True)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            cost = nn.CrossEntropyLoss()
            # train
            for epoch in range(max_epoch):
                running_loss = 0.0
                for i, (inputs, attn_masks, labels) in enumerate(train_loader):

                    optimizer.zero_grad()

                    # Obtaining the logits from the model
                    outputs = model(inputs, attn_masks)

                    loss = cost(outputs.squeeze(-1), labels.long())

                    # Backpropagating the gradients
                    loss.backward()

                    # Optimization step
                    optimizer.step()
                    running_loss += loss.item()
                    if i % n_batch == n_batch - 1:
                        # print("[%d %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / n_batch))
                        running_loss = 0.0

            # print("Finished Training")
            torch.save(model, "model.pkl")


        def reload_model():
            train_model = torch.load('model.pkl')
            return train_model


        def test():
            test_loss = 0
            correct = 0
            model = reload_model()
            for seq, attn_masks, test_target in test_loader:
                outputs = model(seq, attn_masks)
                # sum up batch loss
                seq, attn_masks, test_target = seq, attn_masks, test_target.long()

                test_loss += F.nll_loss(outputs, test_target, reduction='sum').item()
                pred = outputs.data.max(1, keepdim=True)[1]
                correct += pred.eq(test_target.data.view_as(pred)).cpu().sum()

                test_loss /= len(test_loader.dataset)
            print('{} Test Accuracy: {}/{} ({:.2f}%)\n'.format(tag,
                correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            f = open("fintune_wsd_result.tsv", "a")
            f.write('{}\t{:.2f}\n'.format(tag,
                1.0 * correct / len(test_loader.dataset)))


        if __name__ == '__main__':
            train()
            reload_model()
            test()




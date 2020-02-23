import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import Inception3

vocab = []
# open file and read the content in a list
with open('vocab.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        vocab.append(currentPlace)

n_tokens = len(vocab)

class CaptionNet(nn.Module):
    def __init__(self, n_tokens=n_tokens, emb_size=128, lstm_units=256, cnn_feature_size=2048):
        super(self.__class__, self).__init__()

        # два линейных слоя, которые будут из векторов, полученных на выходе Inseption,
        # получать начальные состояния h0 и c0 LSTM-ки, которую мы потом будем
        # разворачивать во времени и генерить ею текст
        self.cnn_to_h0 = nn.Linear(cnn_feature_size, lstm_units).to(device)
        self.cnn_to_c0 = nn.Linear(cnn_feature_size, lstm_units).to(device)

        # вот теперь recurrent part

        # create embedding for input words. Use the parameters (e.g. emb_size).
        self.emb = nn.Embedding(n_tokens, 64, padding_idx=pad_ix).to(device)

        # lstm: настакайте LSTM-ок (1 или более, но не надо сразу пихать больше двух, замечаетесь ждать).
        self.lstm = nn.LSTM(batch_first=True, input_size=64, hidden_size=lstm_units).to(device)
        #         self.lstm2 = nn.LSTM(batch_first = True, input_size = 256, hidden_size = lstm_units).to(device)

        # ну и линейный слой для получения логитов
        self.logits = nn.Linear(lstm_units, n_tokens).to(device)

    def forward(self, image_vectors, captions_ix):
        """
        Apply the network in training mode.
        :param image_vectors: torch tensor, содержаший выходы inseption. Те, из которых будем генерить текст
                shape: [batch, cnn_feature_size]
        :param captions_ix:
                таргет описания картинок в виде матрицы
        :returns: логиты для сгенерированного текста описания, shape: [batch, word_i, n_tokens]
        """
        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)

        #
        captions_emb = self.emb(captions_ix)

        # применим LSTM:
        # 1. инициализируем lstm state с помощью initial_* (сверху)
        # 2. скормим LSTM captions_emb
        # 3. посчитаем логиты из выхода LSTM

        lstm_out, (cell_next, hid_next) = self.lstm(captions_emb, (
        initial_cell[None], initial_hid[None]))  # shape: [batch, caption_length, lstm_units]
        #         lstm_out, (cell_next, hid_next) = self.lstm2(lstm_out, (cell_next, hid_next)) # shape: [batch, caption_length, lstm_units]

        logits = self.logits(lstm_out)

        return logits

class BeheadedInception3(Inception3):
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else: warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x


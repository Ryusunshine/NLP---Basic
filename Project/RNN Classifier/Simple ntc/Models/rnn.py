import torch.nn as nn


class RNNClassifier(nn.Module):

    def __init__( #보통 두개의 메소드를 override해서 구현하는데 그게 __init__과 forward임
        self,
        input_size, #input사이즈는 vocabulary개수(지정할필요없이 torchtext가 자동으로 읽어와서 만들어줄거고 사이즈만 넣어주면됨)
        word_vec_size, #우리는 embedding layer통과시킬거니깐 word embedding layer가 몇차원으로 project될거냐
        hidden_size,
        n_classes,
        n_layers=4,
        dropout_p=.3, 
    ): #입력받아와서 self 에 assign
        self.input_size = input_size  # vocabulary_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()
       #layer 선언
        self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True,
        )
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)

        return y

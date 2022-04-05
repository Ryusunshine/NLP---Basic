#데이터셋과 데이터셋를 가져오는 데이터로더 정의

from torchtext import data


class DataLoader(object):
    '''
    Data loader class to load text file using torchtext library.
    '''

    def __init__(
        self, train_fn, #train file 넣을거임
        batch_size=64,
        valid_ratio=.2, #train : valid = 8 : 2
        device=-1, #GPU or CPU 중 어느 device 에 올릴지
        max_vocab=999999, # vocabulary maximum 개수
        min_freq=1, #minimum 몇개 이상 나온 단어를 vocab에 포함시킬건지
        use_eos=False, # EOS(end of sentence) token를 쓸지말지
        shuffle=True
    ):
        '''
        DataLoader initialization.
        :param train_fn: Train-set filename
        :param batch_size: Batchify data fot certain batch size.
        :param device: Device-id to load data (-1 for CPU)
        :param max_vocab: Maximum vocabulary size
        :param min_freq: Minimum frequency for loaded word.
        :param use_eos: If it is True, put <EOS> after every end of sentence.
        :param shuffle: If it is True, random shuffle the input data.
        '''
        super().__init__()
        
        #torch.text에서 data.Field 함수 가져옴
        
        
        # Define field of the input file. 먼저 필드를 정의해줘야한다. 
        # 두개의 column(positive/negative = label, 글 = text)로 되어있기때문에 self.label, self.text를 각각 정의
        self.label = data.Field( 
            sequential=False, #class만 있으니깐 sequential인 데이터 아님
            use_vocab=True, #어느 class(negative or positive)에 있는지 세어주면 좋으니깐
            unk_token=None #모르는 class 있으면 안됨
        )
        self.text = data.Field( #sequential은 default로 true가 됨
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
            eos_token='<EOS>' if use_eos else None
        )

        
        #train, valid 데이터셋 만들기
        train, valid = data.TabularDataset(# TabularDataset에 넣어서 데이터를 직접 불러옴
            path=train_fn, #train_fn에서 파일불러오고
            format='tsv', 
            fields=[
                ('label', self.label),
                ('text', self.text),
            ],
        ).split(split_ratio=(1 - valid_ratio))#tarin(0.8)과 valid(0.2) 비율로 split
        
        
        #dataset을 얻었으니깐 dataloader에 넣어서 self,train_loader, self.valid_loader로 각각의 loader를 만들어준다
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=shuffle,
            sort_key=lambda x: len(x.text),#로딩을 할때 길이에 따라서 로딩을 해주는게 좋아서 같은 길이끼리 미니배치를 해주는 sort함수를 lambda를 통해 씀 
            sort_within_batch=True, #미니배치 내에서 sort를 해줄것인지(지금은 필요없음)
        )
       
        
        #label, test 각각에 대한 vocabulary를 만들어준다
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq) # It is making mapping table between words and indice.

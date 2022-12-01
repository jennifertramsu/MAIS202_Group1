# Practice with Huggingface

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

def generate_model(args, num_labels):
    ''' 
        Function that generates a pipeline. 
        Transformers need three components to infer or train your model
        1 ) Autoconfig : sets up model and tokenizer configurations
            Can change to BertConfig or any other architectures available in Transformers
        2 ) Autotokenizer : loaded the same as configuration
        3 ) AutoModelForSequenceClassification : loaded from config because training from scratch
    '''

    config = AutoConfig.from_pretrained(
        args.model_name_or_path, 
        num_labels = num_labels,
        finetuning_task = args.task_name
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case = args.do_lower_case
    )

    model = AutoModelForSequenceClassification.from_confid(
        config
    )

    return config, tokenizer, model

class DisasterDataset():
    def __init__(self, data_path, eval_path, tokenizer):
        d_data = pd.read_table(data_path, sep=',')
        d_eval = pd.read_table(eval_path, sep=',')

        row, col = d_data.shape
        d_train = d_data[:int(row * 0.8)]
        d_test = d_data[int(row * 0.8):]

        d_train.reset_index(drop=True, inplace=True)
        d_test.reset_index(drop=True, inplace=True)

        self.tokenizer = tokenizer
        self.dataset = {'train': (d_train, len(d_train)), 
                        'test' : (d_test, len(d_test)), 
                        'eval' : (d_eval, len(d_eval))}
        self.num_labels = len(d_train.target.unique().tolist())
        self.set_split('train')

    def get_vocab(self):
        text = " ".join(self.data.text.tolist())
        text = text.lower()
        vocab = text.split(" ")

        with open('vocab.txt', 'w') as file:
            for word in vocab:
                file.write(word)
                file.write('\n')

        file.close()
        
        return 'vocab.txt'




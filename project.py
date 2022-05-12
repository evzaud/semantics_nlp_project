import numpy as np
import amrlib
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator
from nltk.translate.bleu_score import sentence_bleu
import random
import math
import time



##### Data acquisition

pd.set_option('display.max_colwidth', 1000)
sentences = pd.read_csv('wiki-sentences.tsv', sep='\t', header=None)
qanda = pd.read_csv('dev.tsv', sep='\t', header=None)

def get_qac(question_id):
    '''
    Function for getting a question, sentence, answer set from QAMR dataset

    Parameters
    ----------
    question_id : int
        Index of the question from the dataset
    Returns
    -------
    dict
        Dictionary with a question, answer, and context
    '''
    
    sentence_id = qanda.iloc[question_id][0]
    answer_words = qanda.iloc[question_id][6]
    context = sentences.loc[sentences[0] == sentence_id][1].to_string().split()[1:]
    answer = ' '.join([context[i] for i in map(int, answer_words.split())])
    return {'question' : qanda.iloc[question_id][5], 'answer' : answer, 'context' : ' '.join(context)}


def tokenizer(string):
    '''
    Tokenizer for AMR, allowing it to be passed to seq2seq

    Parameters
    ----------
    string : str
        String representation of AMR graph

    Returns
    -------
    list
        List of tokens
    '''
    
    delims = "()"
    for delim in delims:
        string = string.replace(delim, " "+delim+" ")
    return string.split()


sos_token = "<sos>"
break_token = "<BREAK>"
eos_token = "<eos>"
def process_input(qac):
    '''
    Function for processing input to be passed to seq2seq (src of NN)

    Parameters
    ----------
    qac : dict
        dictionary with "question" and "context" fields

    Returns
    -------
    list
        List of tokens from question and context
    '''
    
    graphs = stog.parse_sents([qac['question'], qac['context']])
    return [sos_token]+tokenizer(graphs[0])+[break_token]+tokenizer(graphs[1])+[eos_token]


def process_output(qac):
    '''
    Function for processing the answer for the dataset (target of NN)

    Parameters
    ----------
    qac : dict
        dictionary with "answer" field

    Returns
    -------
    list
        List of tokens from question and context

    '''
    return qac['answer'].split()


def acquire_data():
    '''
    Function for creating dataframe with QAMR data

    Parameters
    ----------

    Returns
    -------
    df: Pandas dataframe
        Dataframe holding data from QAMR dataset

    '''
    dic = {'src':[],
            'trg':[]
           }
    
    df = pd.DataFrame(dic)

      
    for i in range(len(qanda)):
        qac = get_qac(i)
        src = process_input(qac)
        trg = process_output(qac)
        df.loc[len(df.index)] = [src,trg]
    return df

# Get data, get vocabs, and split data into batches
df = acquire_data()
train_data, valid_data, test_data = np.split(df.sample(frac=1), [int(.8 * len(df)), int(.9 * len(df))])
src_vocab = build_vocab_from_iterator(train_data.src, min_freq=2)
trg_vocab = build_vocab_from_iterator(train_data.trg, min_freq=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
train_data = np.array_split(train_data, len(train_data)/BATCH_SIZE)
valid_data = np.array_split(valid_data, len(valid_data)/BATCH_SIZE)
stog = amrlib.load_stog_model()
gtos = amrlib.load_gtos_model()




##### Method 1: AMR+templates

# Basic templates used, which are rule-based
qw = [
    ('for who', [':ARG2']),
    ('how long', [':duration']),
    ('from where', [':origin']),
    ('to where', [':destination']),
    ('who', [':ARG0']),
    ('what', [':topic', ':ARG1']),
    ('when', [':time']),
    ('where', [':location']),
    ('how many', [':quant']),
    ('how', [':manner', ':instrument', ':mode']),
    ('why', [':purpose'])
]


def likely_role(graph):
    '''
    Outputs likeliest role(s) for the answer of a question to have

    Parameters
    ----------
    graph : str
        String representation of AMR graph

    Returns
    -------
    roles : list or None
        list of likeliest roles OR None if no entry in qw is in the input graph

    '''
    for q_word,roles in qw:
        if q_word in graph.lower():
            return roles
    return None


def search_likely_role(q_graph, context_graph):
    '''
    Searches the context's AMR graph for the presence of the
    likeliest role(s) for the answer. If present, extracts the
    AMR representation of the likely answer

    Parameters
    ----------
    q_graph : str
        String representation of the question's AMR graph
    context_graph : str
        String representation of the context's/sentence's AMR graph

    Returns
    -------
    str
        String representation of the proposed answer's AMR graph

    '''
    # Find likely roles for answer
    roles = likely_role(q_graph)
    if roles is None:
        return None
    
    # Search for likely role(s) in the context graph
    candidate = None
    for role in roles:
        if role in context_graph:
            candidate = role
            break
    if candidate is None:
        return None
    
    # Parse the likely answer from the context graph
    candidate_answer = context_graph.split(role, 1)[1]
    ret_string = '('
    paren_count = 1
    index = 2
    while paren_count > 0:
        curr = candidate_answer[index]
        if curr =='(':
            paren_count +=1
        elif curr ==')':
            paren_count -=1
        ret_string += curr
        index+=1
    return ret_string
    

def answer_via_roles(question, context):
    '''
    Answer prediction function for the template method

    Parameters
    ----------
    question : str
        A question
    context : str
        The question's associated context

    Returns
    -------
    str
        The proposed answer

    '''
    graphs = stog.parse_sents([question, context])
    answer_graph = search_likely_role(graphs[0], graphs[1])
    if answer_graph is None:
        return None
    answer, _ = gtos.generate([answer_graph])
    return answer[0]




##### Method 2: AMR+seq2seq

class Encoder(nn.Module):
    '''
    Class for simple encoder
    '''
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #src = [src len, batch size]        

        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]        
        
        return hidden, cell


class Decoder(nn.Module):
    '''
    Class for simple decoder
    '''
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        #input = [batch size]
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    '''
    Class for seq2seq NN, combining Encoder and Decoder
    '''
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(torch.tensor(src))
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs


# Parameters! Play around with them :)
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512 # Same for encoder and decoder
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    '''
    Function for weight initialization of a model
    
    Parameters
    ----------
    m:
        NN model
    '''
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
        
model.apply(init_weights)

# Set up optimizer & loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, clip):
    '''
    Function to train NN model

    Parameters
    ----------
    model:
        NN model
    iterator:
        Batch iterator over which we will get data
    criterion:
        Loss function
    clip: int/float
        Value to which we clip the gradients

    Returns
    -------
    epoch_loss/len(iterator) : float
        Average loss for the epoch
    '''
    
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()

        output = model(src, trg)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    '''
    Function to evaluate results of NN model

    Parameters
    ----------
    model:
        NN model
    iterator:
        Batch iterator over which we will get data
    criterion:
        Loss function

    Returns
    -------
    epoch_loss/len(iterator) : float
        Average loss for the epoch
    '''
    
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)       
            epoch_loss += loss.item()   
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    '''
    Simple function for calulcating minute length of an epoch
    ''' 
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_data, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_data, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


def test_metrics(model, test_data):
    '''
    Function for getting data for results of both models
    Get: Exact Match percentage and BLEU
    Methods: AMR+templates and AMR+seq2seq
    
    Parameters
    ----------
    model:
        NN model
    test_data:
        Batch iterator over which we will get test data
    criterion:
        Loss function
    '''
    template_exact=0
    seq2seq_exact=0
    template_bleu=0
    seq2seq_bleu=0
    
    for i, test in enumerate(test_data):
        raw_info = test.src.split('# ::snt  ')
        raw_question = raw_info[1].split('\n')[0]
        raw_context = raw_info[2].split('\n')[0]
        
        template_answer = answer_via_roles(raw_question, raw_context)
        if template_answer == test.trg:
            template_exact+=1
        template_bleu += sentence_bleu([test.trg.split()], template_answer.split())
        
        seq2seq_answer = model(test.src)
        if seq2seq_answer == test.trg:
            seq2seq_exact+=1
        seq2seq_bleu += sentence_bleu([test.trg.split()], seq2seq_answer.split())
    
    template_exact /= len(test_data)
    seq2seq_exact /= len(test_data)
    template_bleu /= len(test_data)
    seq2seq_bleu /= len(test_data)
    f_out = open("./results.txt", "w")
    f_out.write('Template Exact Match: {:.3f}\n'.format(template_exact))
    f_out.write('Template BLEU: {:.3f}\n\n'.format(template_bleu))
    f_out.write('seq2seq Exact Match: {:.3f}\n'.format(seq2seq_exact))
    f_out.write('seq2seq BLEU: {:.3f}'.format(seq2seq_bleu))
        
test_metrics(model, test_data)
    
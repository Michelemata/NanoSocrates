"""
NanoSocrates: A Very Small Semantic Language Model
Autore: Michele Matarangolo
Professore: Prof. Vito Walter Anelli, PhD
Test code: 2025_VIII
"""

# PyTorch e gestione modello
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter   # Per monitorare l'addestramento con TensorBoard

# Moduli del progetto
from dataset_creator import DBpediaCollector    # Per creare il dataset
from tokenizer import BPETokenizer              # Per la tokenizzazione del testo

# Librerie di utilità e supporto
import numpy as np
import os
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')   # Nasconde i messaggi di avviso

# Librerie per la valutazione (Metriche)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # Metrica BLEU
from rouge_score import rouge_scorer                                    # Metrica ROUGE
from nltk.translate.meteor_score import meteor_score                    # Metrica METEOR

# Libreria per i grafici
import matplotlib.pyplot as plt


# Imposta seed random per la riproducibilità
torch.manual_seed(42)
np.random.seed(42)


class Config:
    """Raggruppa tutte le impostazioni e gli iperparametri del modello NanoSocrates"""
    # Architettura del modello
    d_model = 512           # Dimensione dei vettori di embedding e dei layer interni del modello
    n_heads = 8             # Numero di "teste" nel meccanismo di Multi-Head Attention
    n_encoder_layers = 4    # Numero di blocchi Encoder
    n_decoder_layers = 4    # Numero di blocchi Decoder
    d_ff = 1024             # Dimensione dell'hidden layer all'interno dei Feed-Forward Network
    dropout = 0.1           # Tasso di dropout
    rope_frac = 0.25        # Frazione della dimensione della testa da usare per il Decoupled RoPE
    max_seq_len = 512       # Lunghezza massima delle sequenze (in token) che il modello può processare

    # Flag per la selezione del meccanismo di attenzione da utilizzare all'interno dell'architettura
    # Opzioni: 'MHA', 'MLA', 'ALTERNATING'
    attention_mode = 'MLA'

    # Configurazione della Multihead Latent Attention configuration
    latent_dim = 128  # Dimensione dello spazio "latente" compresso usato da MLA per ridurre il costo computazionale

    # Configurazione Spanned Masking
    span_masking_prob = 0.15    # Probabilità che un token sia l'inizio di una span da mascherare.
                                # 15% è un valore standard ispirato a BERT.
    max_span_length = 5         # Lunghezza massima di una span mascherata
    mean_span_length = 3        # Lunghezza media della span (usata per una distribuzione geometrica)

    # Tokenizer
    vocab_size = 32000      # Dimensione del vocabolario

    # Gestione file e cartelle
    # Definisce i percorsi per salvare i pesi del modello e i log di TensorBoard.
    model_folder = 'weights'            # Nome della cartella dove salvare i checkpoint del modello
    model_basename = 'tmodel_'          # Prefisso per i file dei pesi (es. 'tmodel_20.pt')
    experiment_name = 'runs/tmodel'     # Path per i log di TensorBoard

    # Iperparametri che controllano il processo di addestramento
    batch_size = 16                 # Numero di esempi elaborati contemporaneamente in un singolo passo di addestramento
    learning_rate = 3e-4            # Learning Rate iniziale
    min_lr = learning_rate / 100    # Learning rate minimo che lo scheduler può raggiungere.
    num_epochs = 60                 # Numero di epoche
    start_factor = 0.01             # Fattore iniziale per lo scheduler del learning rate (usato nella fase di warmup)
    end_factor = 1.0                # Fattore finale per lo scheduler (fine del warmup)

    # Dati
    max_triples_per_movie = 20      # Numero massimo di triple RDF da raccogliere per ogni film.

    # Dizionario che mappa i token speciali a un ID numerico univoco
    special_tokens = {
        '<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
        '<SOT>': 4, '<EOT>': 5, '<SUBJ>': 6, '<PRED>': 7, '<OBJ>': 8,
        '<Text2RDF>': 9, '<RDF2Text>': 10, '<CONTINUERDF>': 11, '<MASK>': 12
    }


class SpannedMasking:
    """
    A differenza del mascheramento standard (come in BERT) che nasconde singoli token,
    lo Spanned Masking nasconde intere sequenze contigue di token (chiamate "span").
    """

    def __init__(self, mask_token_id: int, config: Config):
        self.mask_token_id = mask_token_id              # Salva l'ID del token di mascheramento.
        self.masking_prob = config.span_masking_prob    # Probabilità che un token diventi l'inizio
                                                        # di una span da mascherare
        self.max_span_length = config.max_span_length   # Lunghezza massima consentita per una span.
        self.mean_span_length = config.mean_span_length # Lunghezza media desiderata per le span

    def _sample_span_length(self) -> int:
        """Campiona la lunghezza di una singola span da una distribuzione geometrica"""
        # Distribuzione geometrica con media = mean_span_length
        p = 1.0 / self.mean_span_length
        # Campiona una lunghezza dalla distribuzione
        length = np.random.geometric(p)
        # Si assicura che la lunghezza non superi il massimo consentito
        return min(length, self.max_span_length)

    def apply_span_masking(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applica lo spanned masking ad un batch di sequenze di input"""
        # Ottiene le dimensioni del tensore di input
        batch_size, seq_len = input_ids.shape
        # Crea una copia del tensore di input per non modificare l'originale
        masked_input = input_ids.clone()
        # Crea un tensore di zeri della stessa forma, che verrà usato per tracciare le posizioni mascherate
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)

        # Itera su ogni sequenza nel batch
        for b in range(batch_size):
            i = 0
            # Scorre la sequenza token per token
            while i < seq_len:
                # Decide se iniziare una nuova span da mascherare in questa posizione
                if np.random.random() < self.masking_prob:
                    # Se sì, campiona la lunghezza della span
                    span_len = self._sample_span_length()
                    # Calcola l'indice di fine della span, assicurandosi di non andare oltre la lunghezza della sequenza
                    span_end = min(i + span_len, seq_len)

                    # Sostituisce tutti i token all'interno della span con l'ID del token <MASK>
                    masked_input[b, i:span_end] = self.mask_token_id
                    # Imposta le posizioni corrispondenti nel tensore booleano a True
                    mask_positions[b, i:span_end] = True

                    # Salta l'indice alla fine della span appena mascherata per evitare sovrapposizioni
                    i = span_end
                else:
                    # Se non maschera, passa al token successivo
                    i += 1

        return masked_input, mask_positions


class MovieDataset(Dataset):
    """
    Trasforma i dati grezzi (testi e triple RDF) in un formato che il modello
    può utilizzare per l'addestramento. Da ogni singolo dato, genera esempi
    di addestramento per quattro task diversi arricchendo il dataset ed aiutando
    il modello a generalizzare meglio.
    """

    def __init__(self, data: List[Dict], tokenizer: BPETokenizer, max_seq_len: int = 512, config: Config = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.config = config or Config()

        # Definisce i quattro task di addestramento che verranno generati dal dataset
        self.tasks = ['text2rdf', 'rdf2text', 'rdf_completion_1', 'rdf_completion_2']

        # Inizializza l'oggetto SpannedMasking per il task rdf_completion_1
        mask_token_id = tokenizer.special_tokens['<MASK>']
        self.span_masker = SpannedMasking(mask_token_id, self.config)

    def __len__(self):
        """
        Restituisce la dimensione totale del dataset.
        Poiché per ogni film vengono creati 4 esempi (uno per task), la dimensione totale
        è il numero di film moltiplicato per il numero di task.
        """
        return len(self.data) * len(self.tasks)

    def _serialize_triple(self, triple: Tuple[str, str, str]) -> str:
        """
        Funzione di utilità per convertire una tripla RDF (tupla)
        in una stringa formattata con i token speciali.
        Esempio: ('dbr:The_Matrix', 'dbo:director', 'dbr:Wachowski') ->
                 "<SOT> <SUBJ> dbr:The_Matrix <PRED> dbo:director <OBJ> dbr:Wachowski <EOT>"
        """
        subj, pred, obj = triple
        return f"<SOT> <SUBJ> {subj} <PRED> {pred} <OBJ> {obj} <EOT>"

    def _create_text2rdf_example(self, item: Dict) -> Tuple[List[int], List[int]]:
        """Crea un esempio per il task Text2RDF (da testo a triple)"""
        text = item['text']
        # Prende le prime 3 triple e le serializza in una singola stringa
        triples = [self._serialize_triple(t) for t in item['triples'][:3]]

        # L'input per il modello è il testo seguito dal token <Text2RDF>.
        input_text = f"{text} <Text2RDF>"
        # L'output (target) che il modello deve imparare a generare è la stringa di triple
        target_text = " ".join(triples)

        # Codifica testo di input e target in ID numerici
        input_ids = self.tokenizer.encode(input_text)
        # Aggiunge i token di inizio (<SOS>) e fine (<EOS>) alla sequenza target
        target_ids = [self.tokenizer.special_tokens['<SOS>']] + \
                    self.tokenizer.encode(target_text) + \
                    [self.tokenizer.special_tokens['<EOS>']]

        # Tronca le sequenze se superano la lunghezza massima
        return input_ids[:self.max_seq_len], target_ids[:self.max_seq_len]

    def _create_rdf2text_example(self, item: Dict) -> Tuple[List[int], List[int]]:
        """Crea un esempio per il task RDF2Text (da triple a testo)"""
        # Prende le prime 3 triple e le serializza in una singola stringa
        triples = [self._serialize_triple(t) for t in item['triples'][:3]]
        text = item['text']

        # L'input per il modello è la stringa di triple seguita dal token di controllo <RDF2Text>
        input_text = " ".join(triples) + " <RDF2Text>"
        # Il target è il testo descrittivo del film
        target_text = text

        # Codifica testo di input e target in ID numerici
        input_ids = self.tokenizer.encode(input_text)
        # Aggiunge i token di inizio (<SOS>) e fine (<EOS>) alla sequenza target
        target_ids = [self.tokenizer.special_tokens['<SOS>']] + \
                    self.tokenizer.encode(target_text) + \
                    [self.tokenizer.special_tokens['<EOS>']]

        # Tronca le sequenze se superano la lunghezza massima
        return input_ids[:self.max_seq_len], target_ids[:self.max_seq_len]


    def _create_rdf_completion_1_example(self, item: Dict) -> Tuple[List[int], List[int]]:
        """
        Crea un esempio per il task RDF Completion 1 (ricostruzione da mascheramento).
        Il modello deve "riempire i buchi" in una sequenza di triple.
        """
        # Se non ci sono triple, restituisce sequenze vuote
        if not item['triples']:
            return [], []

        # Serializza 3 triple (non solo la prima) per avere più contesto
        triples_text = " ".join([self._serialize_triple(t) for t in item['triples'][:3]])
        input_ids = self.tokenizer.encode(triples_text)

        # Applica Spanned Masking
        input_tensor = torch.tensor([input_ids])
        masked_input, mask_positions = self.span_masker.apply_span_masking(input_tensor)

        # Il target è la sequenza originale completa
        target_ids = [self.tokenizer.special_tokens['<SOS>']] + input_ids + \
                     [self.tokenizer.special_tokens['<EOS>']]

        # Converte il tensore mascherato in lista e lo restituisce
        return masked_input[0].tolist()[:self.max_seq_len], target_ids[:self.max_seq_len]


    def _create_rdf_completion_2_example(self, item: Dict) -> Tuple[List[int], List[int]]:
            """
            Crea un esempio per il task RDF Completion 2.
            Il modello riceve alcune triple come contesto e deve generare la tripla successiva.
            """
            # Richiede almeno 2 triple per creare un contesto e un target
            if len(item['triples']) < 2:
                return [], []

            # Le prime 2 triple formano il contesto
            context_triples = [self._serialize_triple(t) for t in item['triples'][:2]]
            # La tripla successiva è il target da generare
            target_triple = self._serialize_triple(item['triples'][2]) if len(item['triples']) > 2 else ""

            # L'input è il contesto seguito dal token <CONTINUERDF>
            input_text = " ".join(context_triples) + " <CONTINUERDF>"

            # Codifica testo di input e target in ID numerici
            input_ids = self.tokenizer.encode(input_text)
            # Aggiunge i token di inizio (<SOS>) e fine (<EOS>) alla sequenza target
            target_ids = [self.tokenizer.special_tokens['<SOS>']] + \
                        self.tokenizer.encode(target_triple) + \
                        [self.tokenizer.special_tokens['<EOS>']]

            # Tronca le sequenze se superano la lunghezza massima
            return input_ids[:self.max_seq_len], target_ids[:self.max_seq_len]

    def __getitem__(self, idx):
        """
        Restituisce un singolo esempio di addestramento (input e target).
        Questo è il metodo principale chiamato dal DataLoader.
        """
        # Calcola quale film e quale task usare in base all'indice
        data_idx = idx // len(self.tasks)
        task_idx = idx % len(self.tasks)

        item = self.data[data_idx]
        task = self.tasks[task_idx]

        # Chiama la funzione appropriata per creare l'esempio in base al task selezionato
        if task == 'text2rdf':
            input_ids, target_ids = self._create_text2rdf_example(item)
        elif task == 'rdf2text':
            input_ids, target_ids = self._create_rdf2text_example(item)
        elif task == 'rdf_completion_1':
            input_ids, target_ids = self._create_rdf_completion_1_example(item)
        else:  # rdf_completion_2
            input_ids, target_ids = self._create_rdf_completion_2_example(item)

        # Effettua il Padding alla fine di entrambe le sequenze fino a raggiungere
        # la lunghezza massima (`max_seq_len`) poichè i tensori in un batch devono
        # avere tutti la stessa dimensione
        pad_id = self.tokenizer.get_special_token_id('<PAD>')
        input_ids = input_ids + [pad_id] * (self.max_seq_len - len(input_ids))
        target_ids = target_ids + [pad_id] * (self.max_seq_len - len(target_ids))

        # Restituisce un dizionario contenente i tensori pronti per il modello
        return {
            'input_ids': torch.tensor(input_ids[:self.max_seq_len], dtype=torch.long),
            'target_ids': torch.tensor(target_ids[:self.max_seq_len], dtype=torch.long),
            'task': task
        }


# Creating Input Embeddings
class InputEmbeddings(nn.Module):
    """
    Converte gli ID dei token in vettori densi, detti "embeddings".
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model                              # Dimensione dei vettori
        self.vocab_size = vocab_size                        # Dimensione totale del vocabolario
        self.embedding = nn.Embedding(vocab_size, d_model)  # PyTorch layer che converte gli ID in embeddings

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # Normalizzazione della varianza degli embeddings


class PositionalEncoding(nn.Module):
    """
    Aggiunge informazioni sulla posizione dei token all'interno della sequenza,
    poichè i Transformer non hanno una nozione intrinseca dell'ordine delle parole.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model              # Dimensione dei vettori
        self.max_seq_len = max_seq_len      # Lunghezza massima della sequenza gestibile
        self.dropout = nn.Dropout(dropout)  # Layer di dropout per la regolarizzazione

        # Crea una matrice di positional encoding di dimensione (seq_len, d_model) riempita con zeri
        pe = torch.zeros(max_seq_len, d_model)

        # Crea un tensore che rappresenta le posizioni (da 0 a seq_len - 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Calcola il termine divisore della formula, che determina le diverse frequenze delle onde sinusoidali
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Applica la funzione seno alle colonne con indice pari della matrice 'pe'
        pe[:, 0::2] = torch.sin(position * div_term)
        # Applica la funzione coseno alle colonne con indice dispari della matrice 'pe'
        pe[:, 1::2] = torch.cos(position * div_term)

        # Aggiunge una dimensione all'inizio del tensore 'pe' per poterlo sommare a batch di dati
        pe = pe.unsqueeze(0)

        # Registra 'pe' come un "buffer", ovvero un tensore che è parte
        # dello stato del modello ma non è un parametro addestrabile
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Somma i positional encoding ai tensori di input X
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # Applica il dropout al risultato per regolarizzazione
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Implementa la Layer Normalization. Questa tecnica stabilizza
    l'addestramento normalizzando gli output di un layer per
    avere media zero e varianza uno.
    """

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        # Salva epsilon come 0.000001 per evitare divisioni per zero durante la normalizzazione
        self.eps = eps

        # Definisce 'alpha' come un parametro addestrabile (guadagno), inizializzato a 1
        self.alpha = nn.Parameter(torch.ones(1))

        # Definisce 'bias' come un parametro addestrabile (scostamento), inizializzato a 0.
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # Calcola la media dell'input lungo l'ultima dimensione
        std = x.std(dim=-1, keepdim=True)   # Calcola la deviazione standard dell'input lungo l'ultima dimensione

        # Applica la formula della normalizzazione: (x - media) / (std + epsilon),
        # e poi scala il risultato con 'alpha' e lo trasla con 'bias'
        return self.alpha * (x-mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Implementa il blocco Feed-Forward Network (FFN)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        # Definisce il primo layer
        self.linear_1 = nn.Linear(d_model, d_ff)    # W1 & b1
        self.dropout = nn.Dropout(dropout)          # Inizializza un layer di dropout per la regolarizzazione

        # Definisce il second layer
        self.linear_2 = nn.Linear(d_ff, d_model)    # W2 & b2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (batch, seq_len, d_ff) -->(batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# Creating the Multi-Head Attention block
class MultiHeadAttentionBlock(nn.Module):
    """
    Implementa il meccanismo di Multi-Head Attention (MHA).
    Esegue l'attenzione più volte in parallelo (teste) per pesare
    l'importanza delle parole e catturare aspetti diversi del contesto.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None: # h = number of heads
        super().__init__()
        self.d_model = d_model  # Salva la dimensione del modello
        self.n_heads = n_heads  # Salva il numero di attention heads

        # Assicura che la dimensione del modello sia divisibile per il numero di teste
        assert d_model % n_heads == 0, 'd_model is not divisible by h'

        # d_k è la dimensione di ogni singola attention head
        self.d_k = d_model // n_heads

        # Definisce la matrice dei pesi the weight matrices
        self.w_q = nn.Linear(d_model, d_model)  # W_q
        self.w_k = nn.Linear(d_model, d_model)  # W_k
        self.w_v = nn.Linear(d_model, d_model)  # W_v
        self.w_o = nn.Linear(d_model, d_model)  # W_o

        # Inizializza un layer di dropout per la regolarizzazione sui punteggi di attenzione
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Definisce la funzione per il calcolo dell'attenzione
        mask => Quando vogliamo che certe parole NON interagiscano con altre, le "nascondiamo"
        """

        # Ottiene la dimensione della testa dall'ultima dimensione del tensore query.
        d_k = query.shape[-1]

        # Calcola i punteggi di attenzione come Attention(Q,K,V) = (Q * K^T) / sqrt(d_k).
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Prima di applicare la softmax, applichiamo la maschera per nascondere alcune interazioni tra le parole
        if mask is not None:
            # Applica la maschera sostituendo con un valore molto negativo (-1e9) dove mask è 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Applica la funzione softmax per convertire i punteggi in probabilità
        attention_scores = attention_scores.softmax(dim=-1)

        # Controlla se è stato fornito un layer di dropout
        if dropout is not None:
            # Applica il dropout ai pesi di attenzione per la regolarizzazione
            attention_scores = dropout(attention_scores)

        # Moltiplica i pesi di attenzione per i Value per ottenere l'output contestualizzato
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):

        query = self.w_q(q) # matrice Q'
        key = self.w_k(k)   # matrice K'
        value = self.w_v(v) # matrice V'

        # Suddivisione dei risultati in matrici più piccole per le diverse teste
        # Suddivisione degli embedding (terza dimensione) in h parti
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        # Calcola gli output e gli attention scores
        x, _ = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Calcola la matrice H
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        # Moltiplica la matrice H per la matrice dei pesi W_o, ottenendo la matrice MH-A
        return self.w_o(x)


# MLA utilizza il Decoupled RoPE
class DecoupledRotaryPositionalEmbedding(nn.Module):
    """
    Decoupled Rotary Position Embedding (RoPE)
    Applica la rotazione sinusoidale solo su una parte del vettore
    (posizionale), lasciando inalterata la parte semantica.
    Questa classe verrà poi utilizzata insieme alla Multi-Head Latent Attention.
    """

    def __init__(self, dim_head: int, rope_frac: float = 0.5, base: int = 10000):
        super().__init__()

        # Assicura che la frazione per RoPE sia un valore valido tra 0 e 1.
        assert 0.0 <= rope_frac <= 1.0, "rope_frac deve essere in [0,1]"
        self.dim_head = dim_head    # Salva la dimensione della singola attention head
        self.rope_frac = rope_frac  # Salva la frazione della dimensione da usare per RoPE

        # Calcola la dimensione effettiva della parte del vettore che verrà ruotata.
        self.rope_dim = int(dim_head * rope_frac)

        # Controlla se la dimensione è dispari.
        if self.rope_dim % 2 != 0:
            self.rope_dim -= 1  # La rende pari, poiché RoPE opera su coppie di dimensioni (per sin/cos)
        self.base = base        # Salva la base per il calcolo delle frequenze (valore standard è 10000)

    def _build_freqs(self, seq_len: int, device: torch.device, offset: int = 0):
        """Metodo helper per calcolare le frequenze"""

        dim = self.rope_dim # Usa la dimensione calcolata per RoPE

        # Calcola le frequenze angolari inverse, come nella formula originale di RoPE
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))

        # Crea un tensore di posizioni, tenendo conto di un offset per la generazione di testo
        t = torch.arange(offset, offset + seq_len, device=device).float()

        # Calcola le frequenze per ogni posizione tramite un prodotto esterno
        freqs = torch.einsum("i,j->ij", t, inv_freq)    # (seq_len, dim/2)

        # Duplica le frequenze per applicarle a coppie di dimensioni
        emb = torch.cat((freqs, freqs), dim=-1)

        # Restituisce il tensore delle frequenze
        return emb


    def _rotate(self, x: torch.Tensor, freqs: torch.Tensor):
        """Applica la rotazione RoPE solo sulla parte posizionale."""

        if self.rope_dim == 0:  # Se la dimensione di RoPE è zero, non fa nulla.
            return x            # Restituisce il tensore originale

        # Estrae la parte del vettore da ruotare (posizionale)
        x_pos = x[..., :self.rope_dim]

        # Estrae la parte del vettore da non modificare (semantica)
        x_sem = x[..., self.rope_dim:]

        # Calcola il seno e il coseno delle frequenze per la rotazione
        freqs_sin = freqs.sin()[None, None, :, :self.rope_dim // 2]
        freqs_cos = freqs.cos()[None, None, :, :self.rope_dim // 2]

        # Riformatta la parte posizionale in coppie di dimensioni
        x_pos_pairs = x_pos.view(*x_pos.shape[:-1], -1, 2)

        # Separa le due componenti di ogni coppia
        a, b = x_pos_pairs[..., 0], x_pos_pairs[..., 1]

        # Applica la formula della rotazione 2D
        rot_a = a * freqs_cos - b * freqs_sin
        rot_b = a * freqs_sin + b * freqs_cos

        # Riunisce le componenti ruotate
        x_rot = torch.stack((rot_a, rot_b), dim=-1).flatten(-2)

        # Concatena la parte posizionale ruotata con la parte semantica originale
        return torch.cat((x_rot, x_sem), dim=-1)


    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        seq_len = q.shape[2]    # Ottiene la lunghezza della sequenza dal tensore query
        device = q.device       # Ottiene il device (CPU/GPU) su cui si trovano i tensori

        # Calcola le frequenze necessarie per la lunghezza della sequenza corrente
        freqs = self._build_freqs(seq_len, device, offset=offset)

        # Applica la rotazione al tensore delle query (q)
        q = self._rotate(q, freqs)

        # Applica la rotazione al tensore delle chiavi (k)
        k = self._rotate(k, freqs)

        # Restituisce i tensori q e k ruotati
        return q, k


class MultiheadLatentAttention(nn.Module):
    """
    Implementa la Multihead Latent Attention (MLA).
    A differenza della MHA standard, che calcola l'attenzione su tutta la sequenza
    di Key e Value, la MLA prima comprime le Key e i Value in uno spazio latente
    di dimensione inferiore. Questo riduce drasticamente il costo computazionale,
    specialmente con sequenze lunghe, mantenendo prestazioni elevate.
    Reference: DeepSeek-V2 architecture
    """

    def __init__(self, d_model: int, n_heads: int, latent_dim: int, dropout: float, rope_frac: float):
        super().__init__()
        self.d_model = d_model              # Salva la dimensione del modello
        self.n_heads = n_heads              # Salva il numero di teste di attenzione.
        self.latent_dim = latent_dim        # Salva la dimensione dello spazio latente.
        self.head_dim = d_model // n_heads  # Calcola la dimensione di ogni singola testa.

        # Assicura che la dimensione del modello sia divisibile per il numero di teste.
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"

        # Layer lineare per la proiezione standard di Query
        self.q_proj = nn.Linear(d_model, d_model)

        # Layer lineare per comprimere Key e Value nello spazio latente
        self.kv_compress = nn.Linear(d_model, latent_dim)

        # Layer lineari per espandere dallo spazio latente di nuovo alla dimensione del modello
        self.k_expand = nn.Linear(latent_dim, d_model)
        self.v_expand = nn.Linear(latent_dim, d_model)

        # Projection Layer per l'output finale
        self.out_proj = nn.Linear(d_model, d_model)

        # Inizializza un layer di dropout per la regolarizzazione
        self.dropout = nn.Dropout(dropout)

        # Inizializza il Decoupled RoPE per aggiungere le informazioni posizionali
        self.rope = DecoupledRotaryPositionalEmbedding(self.head_dim, rope_frac=rope_frac)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None,
                past_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # Ottiene le dimensioni del tensore di query
        batch_size, seq_len, _ = query.shape

        # Comprime key nello spazio latente.
        kv_latent = self.kv_compress(key)   # (batch, seq_len, latent_dim)

        # Espande le key e i value dallo spazio latente
        k = self.k_expand(kv_latent)        # (batch, seq_len, d_model)
        v = self.v_expand(kv_latent)        # (batch, seq_len, d_model)

        # Proietta Query
        q = self.q_proj(query)              # (batch, seq_len, d_model)

        # Riformatta i tensori per la multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # (batch, heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Logica per la KV Cache
        # Inizializza l'offset posizionale a 0
        offset = 0

        # Controlla se è stata fornita una cache dai passi precedenti.
        if past_kv_cache is not None:
            past_k, past_v = past_kv_cache              # Estrae key e value dalla cache
            offset = past_k.shape[2]                    # Calcola l'offset in base alla lunghezza della cache
            k = torch.cat([past_k, k], dim=2)   # Concatena le key della cache con quelle attuali
            v = torch.cat([past_v, v], dim=2)   # Concatena i value della cache con quelli attuali

        # Applica il Decoupled RoPE usando l'offset per un corretto positional encoding
        q, k = self.rope(q, k, offset=offset)

        # Calcola i punteggi di attenzione (scaled dot-product)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Applica la maschera di attenzione
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # Applica la maschera di padding per ignorare i token <PAD>
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        # Applica la funzione softmax per ottenere i pesi di attenzione
        attn_weights = F.softmax(scores, dim=-1)

        # Applica il dropout ai pesi
        attn_weights = self.dropout(attn_weights)

        # Moltiplica i pesi per i value per ottenere l'output contestualizzato
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)

        # Riunisce le attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Applica il Projection Layer finale
        output = self.out_proj(attn_output)

        # Applica il dropout all'output
        output = self.dropout(output)

        # Restituisce l'output e la cache aggiornata (k, v) per il prossimo passo
        return output, (k, v)


class ResidualConnection(nn.Module):
    """Implementa la Residual Connection seguita da normalizzazione e dropout"""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Inizializza un layer di dropout per la regolarizzazione
        self.norm = LayerNormalization()    # Inizializza un layer per la normalizzazione

    def forward(self, x, sublayer):
        # Normalizziamo l'input e lo aggiungiamo all'input originale x. Questo crea il processo di Residual Connection
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """Rappresenta un singolo layer dell'Encoder del Transformer"""

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block        # Salva il blocco di self-attention
        self.feed_forward_block = feed_forward_block            # Salva il blocco feed-forward.
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)])    # Crea una lista contenente due istanze di
                                                                # ResidualConnection, una per ogni sublayer

    def forward(self, x, src_mask):
        # Definisce una funzione che gestisce entrambi i tipi di blocco di attenzione
        def self_attention_sublayer(input_tensor):
            # Controlla se il blocco di attenzione è di tipo MultiheadLatentAttention
            if isinstance(self.self_attention_block, MultiheadLatentAttention):
                # Se è MLA, il suo forward restituisce una tupla (output, cache). Prendiamo solo l'output.
                attention_output, _ = self.self_attention_block(input_tensor, input_tensor, input_tensor,
                                                                attn_mask=src_mask)
                return attention_output
            else:
                # Altrimenti, è un MHA standard che restituisce già un singolo tensore
                return self.self_attention_block(input_tensor, input_tensor, input_tensor, src_mask)

        # Applica la prima Residual Connection, passando l'input e la funzione di self-attention.
        x = self.residual_connections[0](x, self_attention_sublayer)

        # Applica la seconda Residual Connection, passando l'output precedente e il blocco feed-forward.
        x = self.residual_connections[1](x, self.feed_forward_block)

        # Restituisce l'output finale del blocco
        return x


class TransformerEncoder(nn.Module):
    """
    Stack di Encoder del modello Transformer.
    L'Encoder è composto da una pila di n_encoder_layers
    'EncoderBlock' identici, disposti in sequenza.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers             # Salva la lista di EncoderBlock che compongono l'encoder
        self.norm = LayerNormalization() # Aggiunge un layer di normalizzazione finale da applicare dopo l'ultimo blocco


    def forward(self, x, src_mask):
        # Itera sequenzialmente su ogni blocco (layer) dell'encoder
        for layer in self.layers:
            # L'output di un blocco diventa l'input del blocco successivo
            x = layer(x, src_mask)

        # Applica la normalizzazione finale all'output dell'ultimo blocco.
        return self.norm(x)


class DecoderBlock(nn.Module):
    """Rappresenta un singolo layer dell'Encoder del Transformer"""

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block    # Salva il blocco di self-attention (può essere MHA o MLA)
        self.cross_attention_block = cross_attention_block  # Salva il blocco di cross-attention (generalmente MHA)
        self.feed_forward_block = feed_forward_block        # Salva il blocco feed-forward

        # Crea una lista di tre ResidualConnection, una per ogni sublayer
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )  # self-attn, cross-attn, feed-forward

    def forward(self, x, encoder_output, src_mask, tgt_mask, past_kv_cache=None):
        # Inizializza la nuova cache a None
        new_kv_cache = None

        # Controlla se il blocco di self-attention è di tipo MLA (che supporta la cache)
        if isinstance(self.self_attention_block, MultiheadLatentAttention):
            # Se è MLA, chiama il suo forward passando la cache precedente
            attn_output, new_kv_cache = self.self_attention_block(
                x, x, x, tgt_mask, past_kv_cache=past_kv_cache
            )
            # Applica la Residual Connection usando l'output dell'attention
            x = self.residual_connections[0](x, lambda _: attn_output)
        else:
            # Altrimenti, è un MHA standard ed usa la logica originale senza cache.
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # Applica la seconda Residual Connection con il blocco di cross-attention
        # La query (q) viene da 'x' (output del decoder), mentre key (k) e value (v) vengono da 'encoder_output'
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))

        # Applica la terza Residual Connection con il blocco Feed-forward
        x = self.residual_connections[2](x, self.feed_forward_block)

        # Restituisce l'output finale del blocco e la nuova cache (che sarà None se si usa MHA standard)
        return x, new_kv_cache


class TransformerDecoder(nn.Module):
    """
    Stack di Decoder del modello Transformer.
    Il Decoder è composto da una pila di n_decoder_layers
    'DecoderBlock' identici, disposti in sequenza.
    """

    # The Decoder takes in instances of 'DecoderBlock'
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers                # Salva la lista di DecoderBlock che compongono il decoder
        self.norm = LayerNormalization()    # Aggiunge un layer di normalizzazione finale

    def forward(self, x, encoder_output, src_mask, tgt_mask, past_kv_caches=None):
        # Se non viene fornita una cache (al primo passo di decodifica), la inizializza come una lista di None
        if past_kv_caches is None:
            # Crea una lista di None, uno per ogni layer del decoder
            past_kv_caches = [None] * len(self.layers)

        # Inizializza una lista vuota per raccogliere le nuove cache da ogni layer
        new_kv_caches = []
        # Itera su ogni blocco (layer) del decoder.
        for i, layer in enumerate(self.layers):
            # Passa l'input e la cache corrispondente a questo layer
            x, new_cache = layer(x, encoder_output, src_mask, tgt_mask, past_kv_cache=past_kv_caches[i])

            # Aggiunge la nuova cache (restituita dal layer) alla lista delle nuove cache
            new_kv_caches.append(new_cache)

        # Restituisce l'output finale normalizzato e la lista delle cache aggiornate
        return self.norm(x), new_kv_caches


class ProjectionLayer(nn.Module):
    """Implementa il Projection Layer finale."""

    def __init__(self, d_model: int, vocab_size: int) -> None: # Model dimension and the size of the output vocabulary
        super().__init__()
        # Crea un layer lineare che mappa la dimensione del modello alla dimensione del vocabolario
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Restituisce i logits. La funzione CrossEntropyLoss di PyTorch applica internamente
        # la funzione log_softmax, quindi non è necessario farlo qui per motivi di efficienza
        # e stabilità numerica
        return self.proj(x)


class NanoSocrates(nn.Module):
    """NanoSocrates: Encoder-Decoder Transformer per la traduzione text-RDF"""

    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: Optional[PositionalEncoding],
                 tgt_pos: Optional[PositionalEncoding],
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder                      # Salva l'istanza dell'Encoder
        self.decoder = decoder                      # Salva l'istanza del Decoder
        self.src_embed = src_embed                  # Salva il layer di embedding per l'input sorgente
        self.tgt_embed = tgt_embed                  # Salva il layer di embedding per l'input target
        self.src_pos = src_pos                      # Salva il positional encoding per il sorgente
        self.tgt_pos = tgt_pos                      # Salva il positional encoding per il target
        self.projection_layer = projection_layer    # Salva il layer di proiezione finale

    def encode(self, src, src_mask):
        """Definisce il metodo per la fase di encoding"""

        # Applica il layer di embedding alla sequenza di input
        src = self.src_embed(src)

        # Controlla se è stato fornito un positional encoding
        if self.src_pos is not None:
            src = self.src_pos(src)     # Aggiunge le informazioni posizionali

        # Passa il risultato all'encoder e restituisce l'output.
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask, past_kv_caches=None):
        """Definisce il metodo per la fase di decoding."""

        # Applica il layer di embedding alla sequenza target
        tgt = self.tgt_embed(tgt)

        # Controlla se è stato fornito un positional encoding
        if self.tgt_pos is not None:
            tgt = self.tgt_pos(tgt)     # Aggiunge le informazioni posizionali.

        # Passa il risultato al decoder insieme all'output dell'encoder e alle maschere
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask, past_kv_caches=past_kv_caches)


    def project(self, x):
        # Prende l'output del decoder e lo passa al Projection Layer per ottenere i logits
        return self.projection_layer(x)


def build_transformer_standard(config: Config) -> NanoSocrates:
    """
    Questa funzione assembla un modello che usa esclusivamente componenti classici:
    Multi-Head Attention (MHA) per tutti i meccanismi di attenzione e il Positional
    Encoding additivo per fornire informazioni sull'ordine dei token.
    """

    # Crea layer di Embeddings per la sequenza sorgente e per la sequenza target
    src_embed = InputEmbeddings(config.d_model, config.vocab_size)
    tgt_embed = InputEmbeddings(config.d_model, config.vocab_size)

    # Crea il positional encoding per la sequenza sorgente e per la sequenza target
    src_pos = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
    tgt_pos = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

    # Inizializza una lista vuota per contenere i blocchi dell'encoder
    encoder_blocks = []

    # Itera per il numero di layer
    for _ in range(config.n_encoder_layers):
        # Crea un blocco di self-attention di tipo Multi-Head Attention standard
        encoder_self_attention_block = MultiHeadAttentionBlock(config.d_model, config.n_heads, config.dropout)

        # Crea il blocco feed-forward
        feed_forward_block = FeedForwardBlock(config.d_model, config.d_ff, config.dropout)

        # Assembla i componenti in un EncoderBlock
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, config.dropout)

        # Aggiunge il blocco completo alla lista
        encoder_blocks.append(encoder_block)

    # Inizializza una lista vuota per i blocchi del decoder
    decoder_blocks = []

    # Itera per il numero di layer specificato nella configurazione
    for _ in range(config.n_decoder_layers):
        # Crea un blocco di self-attention di tipo Multi-Head Attention standard per il decoder
        decoder_self_attention_block = MultiHeadAttentionBlock(config.d_model, config.n_heads, config.dropout)

        # Crea il blocco di cross-attention di tipo Multi-Head Attention standard
        decoder_cross_attention_block = MultiHeadAttentionBlock(config.d_model, config.n_heads, config.dropout)

        # Crea il blocco feed-forward
        feed_forward_block = FeedForwardBlock(config.d_model, config.d_ff, config.dropout)

        # Assembla i componenti in un DecoderBlock
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, config.dropout)

        # Aggiunge il blocco completo alla lista
        decoder_blocks.append(decoder_block)

    # Crea l'Encoder completo passando la lista di blocchi
    encoder = TransformerEncoder(nn.ModuleList(encoder_blocks))

    # Crea il Decoder completo passando la lista di blocchi
    decoder = TransformerDecoder(nn.ModuleList(decoder_blocks))

    # Crea il Projection Layer per l'output
    projection_layer = ProjectionLayer(config.d_model, config.vocab_size)

    # Assembla il transformer
    transformer = NanoSocrates(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Inizializzazione parametri
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Restituisce il modello completo e inizializzato
    return transformer


def build_transformer_hybrid(config: Config, is_alternating: bool) -> NanoSocrates:
    """
    Costruisce un modello NanoSocrates utilizzando Multihead Latent Attention
    o Alternating Attention (Interleaved Attention). Questa funzione può
    creare due tipi di modelli avanzati:
    1.  MLA (is_alternating=False): Usa la Multihead Latent Attention (MLA)
        per tutti i blocchi di self-attention (sia nell'encoder che nel decoder) e la
        MHA standard per la cross-attention nel decoder.
    2.  ALTERNATING (`is_alternating=True`): Alterna layer di MHA standard e MLA
        all'interno dei blocchi di self-attention, per un approccio bilanciato.
    """

    # Crea il layer di embedding per la sequenza sorgente e per la sequenza target
    src_embed = InputEmbeddings(config.d_model, config.vocab_size)
    tgt_embed = InputEmbeddings(config.d_model, config.vocab_size)

    # Imposta i positional encoding a None, perché MLA e DRoPE gestiscono le informazioni posizionali
    src_pos = None
    tgt_pos = None

    def get_self_attn(layer_idx):
        """Funzione di supporto per decidere il tipo di Self-Attention"""
        # Caso MLA: non alternato, usa sempre MLA per la self-attention
        if not is_alternating:
            return MultiheadLatentAttention(
                config.d_model, config.n_heads, config.latent_dim, config.dropout, config.rope_frac
            )
        # Caso ALTERNATING: alterna MHA e MLA in base all'indice del layer
        # MHA (0, 2) | MLA (1, 3)
        else:
            if layer_idx % 2 == 0:
                return MultiHeadAttentionBlock(config.d_model, config.n_heads, config.dropout)
            else:
                return MultiheadLatentAttention(
                    config.d_model, config.n_heads, config.latent_dim, config.dropout, config.rope_frac
                )

    # Inizializza una lista vuota per i blocchi dell'encoder
    encoder_blocks = []

    # Itera per il numero di layer specificato
    for i in range(config.n_encoder_layers):
        # Ottiene il blocco di self-attention appropriato (MHA o MLA) usando la funzione di supporto
        encoder_self_attention_block = get_self_attn(i)

        # Crea il blocco feed-forward
        feed_forward_block = FeedForwardBlock(config.d_model, config.d_ff, config.dropout)

        # Assembla i componenti in un EncoderBlock
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, config.dropout)

        # Aggiunge il blocco completo alla lista
        encoder_blocks.append(encoder_block)

    # Inizializza una lista vuota per i blocchi del decoder
    decoder_blocks = []

    # Itera per il numero di layer specificato
    for i in range(config.n_decoder_layers):
        # Ottiene il blocco di self-attention appropriato (MHA o MLA) per il decoder
        decoder_self_attention_block = get_self_attn(i)

        # La cross-attention rimane sempre MHA standard in queste architetture
        decoder_cross_attention_block = MultiHeadAttentionBlock(config.d_model, config.n_heads, config.dropout)

        # Crea il blocco feed-forward.
        feed_forward_block = FeedForwardBlock(config.d_model, config.d_ff, config.dropout)

        # Assembla i componenti in un DecoderBlock
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, config.dropout)

        # Aggiunge il blocco completo alla lista
        decoder_blocks.append(decoder_block)

    # Assemblaggio finale dell'Encoder e del Decoder
    encoder = TransformerEncoder(nn.ModuleList(encoder_blocks))
    decoder = TransformerDecoder(nn.ModuleList(decoder_blocks))

    # Crea il Projection Layer per l'output
    projection_layer = ProjectionLayer(config.d_model, config.vocab_size)

    # Assembla il transformer
    transformer = NanoSocrates(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Inizializzazione dei parametri
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Restituisce il modello completo e inizializzato
    return transformer


def casual_mask(size):
    """
    Crea una maschera causale per il self-attention del decoder, in
    modo da impedire ad ogni posizione nella sequenza target di
    "sbirciare" le posizioni future durante l'addestramento.
    """
    # Crea una matrice triangolare superiore di 1s, con la diagonale principale e la parte inferiore impostate a zero
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)

    # Inverte la maschera: dove c'era 0 diventa True, dove c'era 1 diventa False
    # Il risultato è una maschera booleana in cui le posizioni future sono False (mascherate)
    return mask == 0


def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    """
    Implementa l'algoritmo di decodifica "greedy" per generare una sequenza di output.
    Questo metodo sceglie il token con la probabilità più alta come token successivo
    nella sequenza e continua a generare token finché non raggiunge la lunghezza massima
    (`max_len`) o non produce il token di fine sequenza (`<EOS>`).
    """

    # Recupera gli ID numerici dei token speciali di inizio e fine sequenza
    sos_idx = tokenizer.get_special_token_id('<SOS>')
    eos_idx = tokenizer.get_special_token_id('<EOS>')

    # Esegue l'encoding della sequenza di input una sola volta all'inizio
    encoder_output = model.encode(source, source_mask)

    # Inizializza l'input per il decoder con un singolo token <SOS>
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Inizializza la cache per le key e i value a None. Verrà popolata al primo passo.
    past_kv_caches = None

    # Se la sequenza generata ha raggiunto la lunghezza massima il ciclo si interrompe
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Building a mask for the decoder input
        # decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoder_mask = None

        # Esegue un passo di decodifica. Passa solo l'ultimo token generato per efficienza,
        # insieme alla cache dei passi precedenti.
        decoder_output, new_kv_caches = model.decode(
            encoder_output, source_mask, decoder_input[:, -1:], decoder_mask, past_kv_caches=past_kv_caches
        )

        # Aggiorna la cache per la prossima iterazione
        past_kv_caches = new_kv_caches

        # Proietta l'output del decoder per ottenere i logits (punteggi) per ogni token del vocabolario
        prob = model.project(decoder_output[:, -1])

        # Seleziona l'ID del token con il logit più alto
        _, next_word = torch.max(prob, dim=1)
        # Concatena il nuovo token alla sequenza di input del decoder per il prossimo ciclo
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        # Condizione di arresto: se il token generato è il token di fine sequenza
        if next_word == eos_idx:
            break

    # Rimuove la dimensione del batch e restituisce la sequenza di ID generata
    return decoder_input.squeeze(0)


def _clean_decoded_text(txt, specials=None):
    """Rimuove token speciali e spazi inutili dalla stringa decodificata."""
    # Controlla se è stata fornita una lista di token speciali da rimuovere
    if specials is None:
        # Se non è stata fornita, usa una lista di default con i token più comuni
        specials = ['<SOS>', '<EOS>', '<PAD>']

    # Itera su ogni token speciale nella lista
    for s in specials:
        # Sostituisce ogni occorrenza del token speciale con una stringa vuota, di fatto eliminandolo.
        txt = txt.replace(s, '')

    # Rimuove spazi multipli e spazi all'inizio/fine della stringa
    txt = ' '.join(txt.split()).strip()

    # Restituisce la stringa di testo pulita
    return txt


def parse_triples_from_string(text: str) -> List[Tuple[str, str, str]]:
    """
    Estrae triple dal testo nella forma:
    <SOT> <SUBJ> subj_val <PRED> pred_val <OBJ> obj_val <EOT>
    Restituisce una lista di tuple (subj, pred, obj).
    """
    # Inizializza una lista vuota per memorizzare le triple estratte
    triples = []
    # Se la stringa di testo è vuota, restituisce la lista vuota senza fare nulla.
    if not text:
        return triples

    # Divide la stringa in una lista di token (parole) basandosi sugli spazi
    tokens = text.split()

    i = 0
    # Controlla se il token corrente è <SOT>
    while i < len(tokens):
        if tokens[i] == '<SOT>':
            subj = pred = obj = None
            i += 1
            # scorri fino a <EOT>
            while i < len(tokens) and tokens[i] != '<EOT>':
                # Prende il token corrente
                tok = tokens[i]
                # Controlla se il token è il soggetto
                if tok == '<SUBJ>' and i+1 < len(tokens):
                    subj = tokens[i+1]
                    i += 2
                    continue
                # Controlla se il token è il predicato
                elif tok == '<PRED>' and i+1 < len(tokens):
                    pred = tokens[i+1]
                    i += 2
                    continue
                # Controlla se il token è l'oggetto
                elif tok == '<OBJ>' and i+1 < len(tokens):
                    obj = tokens[i+1]
                    i += 2
                    continue
                else:
                    # Se non è un marcatore noto, avanza semplicemente l'indice
                    i += 1
            # Dopo il ciclo interno, controlla se sono state trovate tutte e tre le parti della tripla
            if subj is not None and pred is not None and obj is not None:
                triples.append((subj, pred, obj))
        else:
            i += 1

    # Restituisce la lista di tutte le triple estratte
    return triples

def _triples_metrics(pred_text, true_text):
    """
    Converte le stringhe in set di triple e calcola precision/recall/f1.
    Se non ci sono triple nel ground truth, definisce precision=recall=f1=0 (evitare divisione per 0).
    """

    # Converte la stringa di testo predetta in una lista di tuple di triple
    pred_triples = parse_triples_from_string(pred_text)

    # Converte la stringa di testo di riferimento in una lista di tuple di triple
    true_triples = parse_triples_from_string(true_text)

    # Converte la lista di triple predette in un insieme per operazioni di confronto efficienti
    set_pred = set(pred_triples)

    # Converte la lista di triple di riferimento in un insieme
    set_true = set(true_triples)

    # Calcola i TP/FP/FN
    tp = len(set_pred & set_true)
    fp = len(set_pred - set_true)
    fn = len(set_true - set_pred)

    # Calcola la Precision/Recall/F1-Score (con un controllo per evitare la divisione per zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Restituisce le metriche calcolate e alcuni conteggi utili per aggregazioni successive
    return precision, recall, f1, len(set_true), len(set_pred), tp

# Defining function to evaluate the model on the validation dataset
def run_validation(model, val_loader, tokenizer, max_len, device, print_msg, global_state,
                   writer=None, epoch=None):
    """
    Esegue una ciclo di validazione completo sul modello.
    Questa funzione calcola la perdita media sul dataset di validazione,
    genera alcuni esempi di output per un'ispezione qualitativa, e raccoglie
    i dati grezzi necessari per calcolare le metriche di performance dettagliate
    (come BLEU, ROUGE, F1-score per le triple, etc.) per ogni task.
    Opera in modalità di valutazione, disattivando il calcolo dei gradienti.
    """

    model.eval()                            # Imposta il modello in evaluation mode
    tasks_printed_count = defaultdict(int)  # Inizializza un contatore per gli esempi stampati per ogni task
    console_width = 80                      # Imposta una larghezza fissa per i messaggi stampati a console

    # Inizializzazione per il calcolo della loss
    pad_id = tokenizer.get_special_token_id('<PAD>')                # Ottiene l'ID del token di padding
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id,
                                  label_smoothing=0.15).to(device)  # Inizializza la funzione di loss
    total_val_loss = 0.0                                            # Inizializza la validation loss totale a zero
    val_task_loss = defaultdict(list)                               # Dizionario per accumulare le loss per ogni task
    num_batches = 0                                                 # Contatore per il numero di batch processati

    # Dizionario per salvare i risultati grezzi (predizioni e riferimenti) per ogni task
    results = {
        'rdf2text': [],
        'text2rdf': [],
        'rdf_completion_1': [],
        'rdf_completion_2': []
    }

    # Inizia il ciclo di validazione senza calcolare i gradienti per risparmiare memoria e tempo
    with torch.no_grad():

        # Itera su ogni batch del data loader di validazione
        for batch in val_loader:
            num_batches += 1

            # Sposta i tensori del batch sul device corretto (CPU/GPU)
            encoder_input = batch['input_ids'].to(device)
            decoder_target = batch['target_ids'].to(device)

            # Estrae i nomi dei task per il batch corrente
            tasks = batch['task']

            # Preparazione delle maschere e degli input per il decoder
            src_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(2)
            decoder_input = decoder_target[:, :-1]
            labels = decoder_target[:, 1:]
            tgt_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(2)

            # Forward pass per il calcolo della loss
            encoder_output = model.encode(encoder_input, src_mask)
            decoder_output, _ = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            proj_output = model.project(decoder_output)  # [B, T_dec, V]

            # Calcola la loss per l'intero batch
            batch_loss = loss_fn(proj_output.view(-1, tokenizer.vocab_size),
                                 labels.view(-1))
            total_val_loss += batch_loss.item()

            # Assicura che il batch size per la validazione sia 1 per semplificare la gestione degli esempi
            assert encoder_input.size(0) == 1, 'Batch size deve essere uguale ad 1 per la validation.'

            # Calcola e accumula la loss per ogni singolo task nel batch
            for i, task in enumerate(tasks):
                example_loss = loss_fn(proj_output[i].view(-1, tokenizer.vocab_size), labels[i].view(-1))
                val_task_loss[task].append(example_loss.item())

            # Raccolta dei dati per le metriche
            # Ottiene gli ID dei token predetti prendendo l'argmax dei logits.
            preds_ids = proj_output.argmax(dim=-1).cpu().numpy()

            # Ottiene gli ID dei token di riferimento (le etichette).
            labels_ids = labels.cpu().numpy()

            # Itera su ogni esempio nel batch
            B = encoder_input.size(0)
            for i in range(B):
                # Ottiene il task corrente
                task = tasks[i] if isinstance(tasks, (list, tuple)) else tasks

                # Decodifica gli ID in stringhe per l'ispezione e il calcolo delle metriche
                src_text = tokenizer.decode(encoder_input[i].detach().cpu().numpy())
                true_text = tokenizer.decode(decoder_target[i].detach().cpu().numpy())

                # Pulisce le stringhe testuali (rimuovendo <SOS>, <EOS>, <PAD>)
                clean_src = _clean_decoded_text(src_text)
                clean_true = _clean_decoded_text(true_text)

                # Inizializza la stringa predetta per la stampa.
                pred_str_for_print = None

                # Esegue la generazione con greedy_decode solo ogni 20 epoche per risparmiare tempo
                run_greedy = (epoch is not None) and (epoch % 20 == 0)

                # Esegue solo se non è già stato stampato un esempio per questo task
                if run_greedy and tasks_printed_count[task] < 1:
                    # Prepara l'input per la generazione (un singolo esempio)
                    single_src = encoder_input[i].unsqueeze(0)
                    single_src_mask = src_mask[i].unsqueeze(0)

                    # Genera la sequenza di output usando la decodifica greedy
                    gen_ids = greedy_decode(model, single_src, single_src_mask, tokenizer, max_len, device)

                    # Decodifica e pulisce la sequenza generata.
                    pred_str_full = tokenizer.decode(gen_ids.detach().cpu().numpy())
                    pred_str_for_print = _clean_decoded_text(pred_str_full)

                    # Stampa l'esempio a console per un controllo qualitativo
                    print_msg('-' * console_width)
                    print_msg(f'SOURCE: {clean_src}')
                    print_msg(f'TARGET: {clean_true}')
                    print_msg(f'PREDICTED: {pred_str_for_print}')

                    # Aggiornare il contatore del task
                    tasks_printed_count[task] += 1

                    # Esci dal loop se hai trovato 1 esempio per tutti i task
                    if len(tasks_printed_count) >= 4 and all(v > 0 for v in tasks_printed_count.values()):
                        # Se è l'ultima validazione, non vogliamo interrompere qui,
                        # ma vogliamo solo uscire dal loop sui batch successivi.
                        pass  # Continua a processare i batch per le metriche.

                # Per i task di generazione (rdf2text, text2rdf, etc.), usa la predizione greedy se disponibile
                if task == 'rdf2text' or task == 'text2rdf' or task == 'rdf_completion_2':
                    # Se la decodifica greedy non è stata eseguita usa la sequenza
                    # ottenuta tramite argmax dai logits calcolati per la loss
                    if pred_str_for_print is None:
                        pred_text = _clean_decoded_text(tokenizer.decode(preds_ids[i]))
                    else:
                        pred_text = pred_str_for_print

                    # Salva la coppia (predizione, riferimento) per il task corrispondente.
                    results[task].append((pred_text, clean_true))

                # Per il task di completamento
                elif task == 'rdf_completion_1':
                    # Ottiene l'ID del token <MASK>
                    mask_id = tokenizer.get_special_token_id('<MASK>')
                    # Trova le posizioni dei token mascherati nell'input
                    mask_positions = (encoder_input[i].cpu().numpy() == mask_id).nonzero()[0].tolist()

                    # Prende le predizioni e i token corretti
                    pred_ids = preds_ids[i]
                    true_ids = labels_ids[i]

                    # Inizializza i contatori per l'accuratezza
                    correct = 0
                    total_masked = 0

                    # Itera SOLO sulle posizioni che erano mascherate
                    for p in mask_positions:
                        if p < len(true_ids):
                            total_masked += 1
                            # Controlla se la predizione in quella posizione è corretta
                            if pred_ids[p] == true_ids[p]:
                                correct += 1

                    # Salva i risultati per calcolare l'accuratezza aggregata
                    results['rdf_completion_1'].append((pred_ids.tolist(), true_ids.tolist(), mask_positions))

    # Fine del ciclo di validazione: calcola le medie e restituisce i risultati
    if num_batches == 0:  # Controllo per evitare divisione per zero se il loader è vuoto.
        print_msg("Il Validation Set è vuoto.")
        return

    # Calcola la loss media totale e per task
    avg_val_loss = total_val_loss / num_batches
    avg_val_task_losses = {task: sum(vals) / len(vals) for task, vals in val_task_loss.items()}

    # Scrive i risultati su TensorBoard se disponibile
    if writer is not None and epoch is not None:
        writer.flush()

    # Restituisce un dizionario con le loss medie e i risultati grezzi per le metriche
    return {
        'avg_val_loss': avg_val_loss,
        'avg_val_task_losses': avg_val_task_losses,
        'metrics': {
            # Qui ritorniamo i risultati grezzi (liste) per permettere a train_model di fare il calcolo finale.
            'rdf2text_raw': results['rdf2text'],
            'text2rdf_raw': results['text2rdf'],
            'rdf_completion_1_raw': results['rdf_completion_1'],
            'rdf_completion_2_raw': results['rdf_completion_2'],
        }
    }


def get_model(config):
    """Sceglie l'architettura Transformer in base al flag config.attention_mode."""

    if config.attention_mode == 'MHA':
        print("Utilizzo del MultiHead Attention standard (Modello Base).")
        model = build_transformer_standard(config)

    elif config.attention_mode == 'ALTERNATING':
        print("Utilizzo dell'Interleaved Attention (Alternating Attention).")
        # L'alternanza MHA <-> MLA è abilitata passando is_alternating=True
        model = build_transformer_hybrid(config, is_alternating=True)

    elif config.attention_mode == 'MLA':
        print("Utilizzo di Multihead Latent Attention.")
        # MLA (self-attn) + MHA (cross-attn) è abilitato passando is_alternating=False
        model = build_transformer_hybrid(config, is_alternating=False)

    else:
        raise ValueError(f"attention_mode sconosciuta: {config.attention_mode}")

    # Restituisce l'istanza del modello appena costruito
    return model

# Function to construct the path for saving and retrieving model weights
def get_weights_file_path(config, epoch: str):
    """Costruisce in modo standardizzato il percorso completo di un file per salvare i pesi del modello."""
    model_folder = config.model_folder              # Estrae il nome della cartella dei pesi dalla configurazione
    model_basename = config.model_basename          # Estrae il prefisso per i file dei pesi dalla configurazione
    model_filename = f"{model_basename}{epoch}.pt"  # Costruisce il nome completo del file

    # Combina il percorso della directory corrente, la cartella dei modelli e il nome del file
    # per creare un percorso completo e valido.
    return str(Path('.') / model_folder / model_filename)

def train_model(tokenizer, train_loader, val_loader, config):
    """Gestisce l'intero ciclo di addestramento e validazione del modello NanoSocrates."""

    # Imposta il device su GPU ('cuda') se disponibile, altrimenti usa la CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzo il device: {device}")

    # Crea la cartella per salvare i pesi del modello, se non esiste già
    Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    # Inizializza il writer di TensorBoard per registrare i log dell'addestramento
    writer = SummaryWriter(config.experiment_name)

    # Inizializza il modello sul device corretto usando la funzione factory 'get_model'
    print("4. Inizializzazione del modello...")
    model = get_model(config).to(device)

    # Inizializza l'ottimizzatore
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    # Calcola il numero totale di passi di ottimizzazione
    total_opt_steps = config.num_epochs * len(train_loader)

    # Definisce il numero di passi per la fase di "warmup" (aumento graduale del learning rate)
    warmup_steps = max(100, int(0.05 * total_opt_steps))

    # Crea uno scheduler lineare per il warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=config.start_factor, end_factor=config.end_factor,
                                total_iters=warmup_steps)

    # Calcola la durata della fase di decadimento del learning rate (dopo il warmup).
    cos_T_max = max(1, total_opt_steps - warmup_steps)

    # Crea uno scheduler a decadimento cosenoidale per la fase successiva al warmup.
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cos_T_max, eta_min=config.min_lr)

    # Combina i due scheduler in sequenza: prima il warmup, poi il decadimento.
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # Ottiene l'ID del token di padding, che verrà ignorato nel calcolo della loss
    pad_id = tokenizer.get_special_token_id('<PAD>')

    # Inizializza la Loss Function, con label smoothing per regolarizzazione.
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.15).to(device)

    # Inizializza le variabili per tracciare la loss durante l'addestramento
    train_task_losses = defaultdict(list)
    total_train_loss = 0.0
    num_train_batches = 0

    # Inizializza le liste per salvare la cronologia delle loss per i grafici finali
    train_loss_history = []
    val_loss_history = []
    val_task_loss_history = defaultdict(list)

    # Inizializza gli oggetti per calcolare ROUGE e la funzione di smoothing per BLEU
    rouge_s = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth_fn = SmoothingFunction().method1

    def _agg_triple_metrics(list_pairs):
        """
        Definisce una funzione interna per aggregare le
        metriche a livello di triple (P, R, F1).
        Restituisce i valori MACRO-AVERAGED.
        """

        # Inizializza delle liste per memorizzare le metriche di ogni singolo esempio
        precisions, recalls, f1s = [], [], []

        # Inizializza i contatori globali per i TP, FP e FN
        # Questi contatori sono usati per il calcolo MICRO
        total_tp = total_fp = total_fn = 0

        # Itera su ogni coppia (stringa predetta, stringa di riferimento) nel dataset
        for pred_str, true_str in list_pairs:
            # Chiama la funzione di supporto per calcolare le metriche per la singola coppia
            p, r, f1, n_true, n_pred, tp = _triples_metrics(pred_str, true_str)

            # 1. Aggiunge le metriche del singolo esempio alle liste (per il calcolo MACRO)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        # Calcola la media aritmetica delle metriche di tutti gli esempi.
        # Ogni esempio ha lo stesso peso, indipendentemente da quante triple contiene.
        macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
        macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        # Restituisce le tre metriche globali MACRO calcolate
        return macro_precision, macro_recall, macro_f1

    initial_epoch = 0   # Inizializza l'epoca di partenza
    global_step = 0     # Inizializza il contatore globale dei passi

    for epoch in range(initial_epoch, config.num_epochs + 1):
        batch_iterator = tqdm(train_loader, desc=f'Elaborazione epoca numero {epoch:02d}')

        # Itera su ogni batch nel training loader
        for batch in batch_iterator:
            model.train()   # Imposta il modello in modalità di addestramento

            # Sposta i dati del batch sul device corretto
            encoder_input = batch['input_ids'].to(device)
            decoder_input = batch['target_ids'][:, :-1].to(device)
            labels = batch['target_ids'][:, 1:].to(device)
            tasks = batch['task']

            # Crea le maschere per l'encoder e il decoder
            src_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(2)

            target_padding_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(2)
            causal = casual_mask(decoder_input.size(1)).to(device)
            tgt_mask = target_padding_mask & causal

            # Forward pass
            encoder_output = model.encode(encoder_input, src_mask)
            decoder_output, _ = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            proj_output = model.project(decoder_output)

            # Calcola la loss per il batch
            loss = loss_fn(proj_output.view(-1, tokenizer.vocab_size), labels.view(-1))

            # Accumula loss totale
            total_train_loss += loss.item()
            num_train_batches += 1

            # Accumula la loss anche per ogni singolo task
            for i, task in enumerate(tasks):
                example_loss = loss_fn(proj_output[i].view(-1, tokenizer.vocab_size), labels[i].view(-1))
                train_task_losses[task].append(example_loss.item())

            # Aggiorna la descrizione della barra di avanzamento con la loss corrente.
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Registra la loss del batch su TensorBoard.
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backward pass e ottimizzazione
            loss.backward()         # Calcola i gradienti (backpropagation)
            optimizer.step()        # Aggiorna i pesi del modello
            scheduler.step()        # Aggiorna il learning rate
            optimizer.zero_grad()   # Azzera i gradienti per il batch successivo

            global_step += 1        # Incrementa il contatore globale dei passi

        # Calcola la loss media di addestramento per l'epoca appena conclusa
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_task_losses = {
            task: sum(losses) / len(losses)
            for task, losses in train_task_losses.items()
        }
        train_loss_history.append(avg_train_loss)

        # Stampa risultati training
        print(f"\n{'=' * 80}")
        print(f"EPOCA {epoch:02d} - RISULTATI TRAINING")
        print(f"{'=' * 80}")
        print(f"Training Loss Totale: {avg_train_loss:.4f}")
        print(f"\nTraining Losses per Task:")

        ordered_tasks = ['text2rdf', 'rdf2text', 'rdf_completion_1', 'rdf_completion_2']
        # Stampa e log nell'ordine fisso desiderato
        for task in ordered_tasks:
            if task in avg_train_task_losses:
                loss = avg_train_task_losses[task]
                print(f"  • {task:18s}: {loss:.4f}")
                writer.add_scalar(f'train_loss/{task}', loss, epoch)

        # Esegue la validazione alla fine di ogni epoca
        val_metrics = run_validation(model, val_loader, tokenizer, config.max_seq_len, device,
                       lambda msg: batch_iterator.write(msg), global_step, writer, epoch=epoch)

        # Salva la cronologia delle loss di validazione per i grafici
        val_loss_history.append(val_metrics['avg_val_loss'])
        for task, loss in val_metrics['avg_val_task_losses'].items():
            val_task_loss_history[task].append(loss)

        # Stampa le metriche dettagliate e salva il modello ogni 20 epoche
        if epoch % 20 == 0:
            avg_val_loss = val_metrics['avg_val_loss']                  # Estrae la loss media di validazione.
            avg_val_task_losses = val_metrics['avg_val_task_losses']
            results_raw = val_metrics['metrics']                        # Estrae i risultati grezzi per le metriche

            # Stampa risultati Validation
            print(f"\n{'=' * 80}")
            print(f"EPOCA {epoch:02d} - RISULTATI VALIDATION")
            print(f"{'=' * 80}")
            print(f"Validation Loss Totale: {avg_val_loss:.4f}")
            print(f"\nValidation Losses per Task:")
            for task, loss in sorted(avg_val_task_losses.items()):
                print(f"  • {task:18s}: {loss:.4f}")
                if writer is not None and epoch is not None:
                    writer.add_scalar(f'val_loss/{task}', loss, epoch)

            # Calcolo delle metriche per ogni task
            print(f"\nMETRICHE per task:")

            # Metriche per RDF2Text (BLEU, ROUGE, METEOR)
            # Estrae le coppie di testo predetto e di riferimento per il task rdf2text
            rdf2text_pairs = results_raw['rdf2text_raw']
            # Procede solo se ci sono risultati da valutare
            if len(rdf2text_pairs) > 0:
                # Inizializza le liste per raccogliere i punteggi
                bleu_scores, rougeL_scores, meteor_scores = [], [], []

                # Itera su ogni coppia
                for pred_text, true_text in rdf2text_pairs:
                    # Divide le stringhe in liste di token, come richiesto da BLEU
                    ref_tokens = true_text.split()
                    hyp_tokens = pred_text.split()
                    try:
                        # Calcola il punteggio BLEU a livello di frase, usando una funzione di smoothing
                        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth_fn)
                    except Exception:
                        # In caso di errore, assegna un punteggio di 0
                        bleu = 0.0
                    # Aggiunge il punteggio alla lista
                    bleu_scores.append(bleu)

                    # Calcola il punteggio ROUGE-L (F-measure)
                    r = rouge_s.score(true_text, pred_text)
                    rougeL_scores.append(r['rougeL'].fmeasure)

                    try:
                        # Calcola il punteggio METEOR
                        meteor = meteor_score([true_text], pred_text)
                    except Exception:
                        # In caso di errore, assegna un punteggio di 0.
                        meteor = 0.0
                    # Aggiunge il punteggio alla lista.
                    meteor_scores.append(meteor)

                # Calcola la media di ogni metrica su tutto il Validation Set
                avg_bleu = float(sum(bleu_scores) / len(bleu_scores))
                avg_rougeL = float(sum(rougeL_scores) / len(rougeL_scores))
                avg_meteor = float(sum(meteor_scores) / len(meteor_scores))

                # Stampa le medie delle metriche
                print(f"\nRDF2Text:")
                print(f"  • BLEU     : {avg_bleu:.4f}")
                print(f"  • ROUGE-L  : {avg_rougeL:.4f}")
                print(f"  • METEOR   : {avg_meteor:.4f}")

                # Registra le metriche medie su TensorBoard
                if writer is not None and epoch is not None:
                    writer.add_scalar('val_metric/rdf2text_bleu', avg_bleu, epoch)
                    writer.add_scalar('val_metric/rdf2text_rougeL', avg_rougeL, epoch)
                    writer.add_scalar('val_metric/rdf2text_meteor', avg_meteor, epoch)
            else:
                print("\nRDF2Text: nessun esempio da valutare.")

            # Metriche per Text2RDF (Precision, Recall, F1 a livello di triple)
            text2rdf_pairs = results_raw['text2rdf_raw']    # Estrae i dati grezzi

            # Procede solo se ci sono dati
            if len(text2rdf_pairs) > 0:
                # Chiama la funzione helper per calcolare le metriche aggregate (micro-averaged)
                p, r, f1 = _agg_triple_metrics(text2rdf_pairs)
                # Stampa i risultati
                print(f"\nText2RDF:")
                print(f"  • Precision: {p:.4f}")
                print(f"  • Recall   : {r:.4f}")
                print(f"  • F1       : {f1:.4f}")

                # Registra i risultati su TensorBoard
                if writer is not None and epoch is not None:
                    writer.add_scalar('val_metric/text2rdf_precision', p, epoch)
                    writer.add_scalar('val_metric/text2rdf_recall', r, epoch)
                    writer.add_scalar('val_metric/text2rdf_f1', f1, epoch)
            else:
                print("\nText2RDF: nessun esempio da valutare.")

            # Accuracy per RDF Completion 1
            rdfc1_data = results_raw['rdf_completion_1_raw']    # Estrae i dati

            # Procede solo se ci sono dati
            if len(rdfc1_data) > 0:
                total_masked = 0                                # Contatore per il totale dei token mascherati
                total_correct = 0                               # Contatore per le predizioni corrette

                # Itera sui dati (predizioni, riferimenti, posizioni delle maschere)
                for pred_ids, true_ids, mask_positions in rdfc1_data:
                    for p in mask_positions:                    # Itera solo sulle posizioni che erano mascherate
                        if p < len(true_ids):                   # Controllo di sicurezza
                            total_masked += 1                   # Incrementa il totale dei token da valutare
                            if pred_ids[p] == true_ids[p]:      # Controlla se la predizione è corretta
                                total_correct += 1              # Incrementa il contatore dei corretti

                # Calcola l'accuratezza finale
                accuracy = total_correct / total_masked if total_masked > 0 else 0.0

                # Stampa il risultato
                print(f"\nRDF Completion 1:")
                print(f"  • Accuracy: {accuracy:.4f} ({total_correct}/{total_masked})")

                # Registra su TensorBoard
                if writer is not None and epoch is not None:
                    writer.add_scalar('val_metric/rdf_completion_1_acc', accuracy, epoch)
            else:
                print("\nRDF Completion 1: nessun esempio da valutare.")

            # Metriche per RDF Completion 2 (Precision, Recall, F1 a livello di triple)
            rdfc2_pairs = results_raw['rdf_completion_2_raw']   # Estrae i dati.

            # Procede solo se ci sono dati
            if len(rdfc2_pairs) > 0:
                # Chiama la stessa funzione helper usata per text2rdf
                p2, r2, f12 = _agg_triple_metrics(rdfc2_pairs)

                # Stampa i risultati
                print(f"\nRDF Completion 2:")
                print(f"  • Precision: {p2:.4f}")
                print(f"  • Recall   : {r2:.4f}")
                print(f"  • F1       : {f12:.4f}")

                # Registra i risultati su TensorBoard
                if writer is not None and epoch is not None:
                    writer.add_scalar('val_metric/rdf_completion_2_precision', p2, epoch)
                    writer.add_scalar('val_metric/rdf_completion_2_recall', r2, epoch)
                    writer.add_scalar('val_metric/rdf_completion_2_f1', f12, epoch)
            else:
                print("\nRDF Completion 2: nessun esempio da valutare.")

            # Salva lo stato del modello, dell'ottimizzatore e dell'epoca
            model_filename = get_weights_file_path(config, f'{epoch:02d}')

            torch.save({
                'epoch': epoch,  # Current epoch
                'model_state_dict': model.state_dict(),  # Current model state
                'optimizer_state_dict': optimizer.state_dict(),  # Current optimizer state
                'global_step': global_step  # Current global step
            }, model_filename)

        # Assicura che tutti i dati in sospeso vengano scritti su disco
        if writer is not None:
            writer.flush()

    # Restituisce le cronologie delle loss e il modello finale addestrato
    return train_loss_history, val_loss_history, val_task_loss_history, model


def plot_loss_curves(train_losses, val_losses, val_task_losses):
    """Genera e salva i grafici di perdita utilizzando Matplotlib."""

    # Calcola il range di epoche
    epochs = range(1, len(train_losses) + 1)

    # Grafico Loss Totale (Train vs Val)
    plt.figure(figsize=(10, 5))

    # Disegna la curva della Training Loss
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)

    # Disegna la curva della loss di validazione, usando uno stile tratteggiato e dei marcatori.
    plt.plot(epochs[:len(val_losses)], val_losses, label='Validation Loss (Total)', marker='o', linestyle='--',
             linewidth=2)

    plt.title('Loss Totale per Epoca (Training vs Validation)')
    plt.xlabel('Epoca')
    plt.ylabel('Loss (Cross Entropy)')

    # Aggiunge una griglia per una migliore leggibilità.
    plt.grid(True, linestyle=':', alpha=0.7)

    # Mostra la legenda per identificare le curve
    plt.legend()

    # Ottimizza il layout per evitare che le etichette si sovrappongano
    plt.tight_layout()

    # Salva il grafico come file PNG
    plt.savefig('loss_total_curve.png')

    # Chiude la figura per liberare memoria
    plt.close()

    # Stampa un messaggio di conferma
    print("Grafico salvato: loss_total_curve.png")

    # Grafico Loss Per-Task (Solo Validation)
    plt.figure(figsize=(12, 6))

    # Per i task, usiamo le epoche di validazione effettive
    val_epochs = epochs[:len(val_losses)]

    # Itera sul dizionario delle loss per task
    for task, losses in val_task_losses.items():
        # Controlla se ci sono dati di loss per il task corrente
        if losses:
            # Disegna una curva separata per ogni task, con un marcatore per ogni punto.
            plt.plot(val_epochs[:len(losses)], losses, label=f'Validation Loss: {task}', linewidth=2, marker='.')

    # Imposta il titolo e le etichette per gli assi
    plt.title('Validation Loss Per-Task')
    plt.xlabel('Epoca')
    plt.ylabel('Loss (Cross Entropy)')

    # Aggiunge una griglia
    plt.grid(True, linestyle=':', alpha=0.7)

    # Mostra la legenda, posizionandola nell'angolo in alto a destra.
    plt.legend(loc='upper right')

    # Ottimizza il layout
    plt.tight_layout()

    # Salva il grafico come file PNG
    plt.savefig('loss_per_task_curve.png')

    # Chiude la figura
    plt.close()

    # Stampa un messaggio di conferma
    print("Grafico salvato: loss_per_task_curve.png")


def main():
    """
    Funzione principale che orchestra l'intera pipeline di esecuzione del progetto.
    Gestisce i seguenti passaggi in ordine:
    1.  Inizializzazione della configurazione e del collettore di dati.
    2.  Controllo della presenza di dataset pre-esistenti e richiesta all'utente se utilizzarli.
    3.  Se necessario, avvio della raccolta di un nuovo dataset da DBpedia e Wikipedia.
    4.  Addestramento di un nuovo tokenizer sui testi raccolti.
    5.  Suddivisione dei dati in training set e validation set e creazione dei DataLoader.
    6.  Avvio della funzione di addestramento del modello (`train_model`).
    7.  Al termine, avvio della funzione per generare i grafici delle curve di loss.
    """

    # Inizializza l'oggetto di configurazione con tutti gli iperparametri
    config = Config()

    # Inizializza l'oggetto per la raccolta dei dati
    collector = DBpediaCollector()

    # Definisce il nome della directory dove sono salvati i dataset
    datasets_dir = 'datasets'

    # Inizializza la variabile del dataset a None
    dataset = None

    # Controlla se la cartella dei dataset esiste
    if os.path.exists(datasets_dir):
        # Ottiene una lista dei file CSV presenti
        existing_datasets = [f.replace('.csv', '') for f in os.listdir(datasets_dir)
                             if f.endswith('.csv') and 'stats' not in f]

        # Se sono stati trovati dataset li elenca
        if existing_datasets:
            print(f"\nTrovati dataset esistenti: {existing_datasets}")
            try:
                use_existing = input("Vuoi utilizzare un dataset esistente? (y/N): ").strip().lower()
            except EOFError:
                # Gestisce il caso in cui lo script non sia eseguito in un terminale interattivo
                use_existing = 'n'

            # Se l'utente risponde affermativamente mostra i dataset
            if use_existing in ['y', 'yes']:
                print("Dataset disponibili:")
                for i, dataset_name in enumerate(existing_datasets):
                    print(f"  {i + 1}. {dataset_name}")

                try:
                    # Chiede all'utente di scegliere un dataset
                    choice = int(input("Seleziona il dataset che vuoi utilizzare: ")) - 1
                    if 0 <= choice < len(existing_datasets):
                        dataset_name_full = existing_datasets[choice] + '.csv'
                        print(f"Caricamento dataset: {dataset_name_full}")

                        # Carica il dataset dal file CSV
                        dataset = collector.load_dataset_csv(dataset_name_full)
                        print(f"Caricati {len(dataset)} film dal dataset salvato.")
                    else:
                        print("Scelta non valida. Si procede con la raccolta di nuovi dati...")
                except (ValueError, IndexError):
                    print("Input non valido. Si procede con la raccolta di nuovi dati...")
                except Exception as e:
                    print(f"Errore nel caricamento del dataset: {e}. Si procede con la raccolta di nuovi dati...")
                    dataset = None

    # Creazione Nuovo Dataset (se non è stato caricato nulla)
    if dataset is None or len(dataset) == 0:
        print("1. Raccolta dati da DBpedia e Wikipedia...")
        # Recupera una lista di URI di film da DBpedia
        movie_uris = collector.get_movie_entities()

        # Inizializza la lista per il nuovo dataset.
        dataset = []
        successful_collections = 0

        # Itera su una porzione degli URI raccolti per creare il dataset
        for movie_uri in tqdm(movie_uris[:650], desc="Collecting movie data"):
            # Recupera le triple RDF per il film corrente
            triples_data = collector.get_movie_triples(movie_uri, config.max_triples_per_movie)

            if not triples_data:    # Salta se non ci sono triple
                continue

            # Converte e pulisce le triple
            triples = []
            for triple_dict in triples_data:
                # Pulisce l'URI del soggetto e predicato usando il metodo helper del collettore
                subject = collector.clean_uri(triple_dict['subject'])
                predicate = collector.clean_uri(triple_dict['predicate'])

                # Controlla se l'oggetto della tripla è un valore letterale (es. una stringa) o un altro URI
                if triple_dict['object_type'] == 'literal':
                    # Se è un letterale, lo converte in stringa e lo tronca ai primi 100 caratteri
                    obj = str(triple_dict['object'])[:100]
                    # Controlla se il letterale, dopo aver rimosso gli spazi bianchi, è vuoto
                    if len(obj.strip()) == 0:
                        # Se è vuoto, salta questa tripla e passa alla successiva nel ciclo
                        continue
                else:
                    # Se l'oggetto non è un letterale (quindi è un URI), lo pulisce come gli altri.
                    obj = collector.clean_uri(triple_dict['object'])

                # Aggiunge la tripla pulita (ora come tupla) alla lista delle triple
                triples.append((subject, predicate, obj))

            # Dopo aver processato tutte le triple, controlla se la lista delle triple pulite è vuota
            if not triples:
                # Se è vuota salta il resto del codice per questo film e passa al prossimo URI
                continue

            # Recupera l'abstract di Wikipedia per il film
            text = collector.get_wikipedia_abstract(movie_uri)

            # Aggiunge il film al dataset solo se ha sia testo che triple di qualità sufficiente
            if (text and len(text.strip()) > 100 and triples and len(triples) >= 4):
                dataset.append({
                    'movie_uri': movie_uri,
                    'text': text[:800],                                 # Limita la lunghezza del testo
                    'triples': triples[:config.max_triples_per_movie],  # Limita il numero di triple
                })
                successful_collections += 1

        print(f"Raccolti dati per {len(dataset)} film.")

        # Se sono stati raccolti dati, li salva in un nuovo file CSV
        if dataset:
            new_dataset_filename = f"movies_dataset_{len(dataset)}.csv"
            saved_filepath = DBpediaCollector.save_dataset_csv(dataset, filename=new_dataset_filename)
            print(f"Dataset salvato in: {saved_filepath}")

    # Controllo finale: se non ci sono dati, termina l'esecuzione
    if not dataset:
        print("\nATTENZIONE: Nessun dato disponibile. Impossibile procedere con l'addestramento.")
        return None, None  # Esci da main

    # Prepara una lista di tutti i testi (abstract e triple serializzate) per addestrare il tokenizer
    all_texts = []
    for item in dataset:
        all_texts.append(item['text'])
        for triple in item['triples']:
            serialized_triple = f"<SOT> <SUBJ> {triple[0]} <PRED> {triple[1]} <OBJ> {triple[2]} <EOT>"
            all_texts.append(serialized_triple)

    print("2. Addestramento del tokenizer...")
    # Inizializza e addestra il tokenizer
    tokenizer = BPETokenizer(config.vocab_size, config.special_tokens)
    tokenizer.train(all_texts)

    # Salva il tokenizer addestrato su file
    tokenizer.tokenizer.save('tokenizer.json')

    # Preparazione dei DataLoader
    print("3. Creazione dei DataLoader...")

    # Suddivide il dataset in training (80%) e validazione (20%)
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    val_data = dataset[train_size:] if train_size < len(dataset) else dataset[-2:]  # Garantisce il Validation Set

    # Crea le istanze di MovieDataset per training e validazione
    train_dataset = MovieDataset(train_data, tokenizer, config.max_seq_len, config)
    val_dataset = MovieDataset(val_data, tokenizer, config.max_seq_len, config)

    # Crea i DataLoader, che gestiranno il caricamento dei dati in batch
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Addestramento e Plotting
    print("5. Training...")
    # Chiama la funzione principale per avviare l'addestramento del modello
    train_history, val_history, val_task_history, final_model = train_model(tokenizer, train_loader, val_loader, config)

    print("\n6. Creazione dei grafici delle curve di loss...")
    # Al termine dell'addestramento, chiama la funzione per creare i grafici.
    plot_loss_curves(train_history, val_history, val_task_history)

    # Restituisce il modello finale addestrato e il tokenizer.
    return final_model, tokenizer


if __name__ == "__main__":
    model, tokenizer = main()
    print("Addestramento di NanoSocrates completato!")

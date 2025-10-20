# Importa le classi necessarie dalla libreria 'tokenizers' di HuggingFace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from typing import List

class BPETokenizer:
    """
    Questa classe implementa un tokenizer basato sull'algoritmo Byte-Pair Encoding (BPE)
    utilizzando la libreria 'tokenizers' di HuggingFace.
    Le sue responsabilità sono:
    1. Addestrare un nuovo vocabolario partendo da un corpus di testi.
    2. Tokenizzare il testo in una sequenza di ID numerici.
    3. Decodificare una sequenza di ID in una stringa di testo leggibile.
    """
    def __init__(self, vocab_size, special_tokens):
        self.vocab_size = vocab_size            # Salva la dimensione del vocabolario
        self.special_tokens = special_tokens    # Salva i token speciali (es. <SUBJ>)

        # Inizializza il tokenizer di HuggingFace con un modello BPE
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        # Imposta il pre-tokenizer: prima di applicare BPE, il testo verrà diviso in parole
        # usando gli spazi come delimitatori.
        self.tokenizer.pre_tokenizer = Whitespace()

        # Inizializza due dizionari che serviranno per mappare ogni token al suo ID univoco e viceversa
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, texts: List[str]):
        """
        Addestra il tokenizer BPE su una lista di testi.
        Durante l'addestramento, l'algoritmo BPE impara le fusioni di sub-parole
        più frequenti per costruire il vocabolario finale.
        """
        # Crea un 'trainer' che definisce i parametri per l'addestramento BPE
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            show_progress=True,
            min_frequency=2
        )

        # Avvia l'addestramento del tokenizer
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        # Una volta terminato l'addestramento, costruisce i dizionari di mappatura
        vocab = self.tokenizer.get_vocab()
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()} # Inverte il dizionario per la decodifica

        # Aggiunge un post-processor che serve a definire un template per l'output
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A",
            pair="$A $B",
            # Specifica quali sono i token speciali e i loro ID corrispondenti
            special_tokens=[(token, self.token_to_id[token]) for token in self.special_tokens if
                            token in self.token_to_id]
        )

        print(f"Dimensione del vocabolario: {len(self.token_to_id)}")

    def encode(self, text: str) -> List[int]:
        """
        Converte una stringa di testo in una lista di ID di token
        """
        # Controlla se il tokenizer è stato addestrato prima di provare a usarlo
        if not self.token_to_id:
            raise ValueError("Il tokenizer non è stato addestrato")

        # Esegue la tokenizzazione
        encoding = self.tokenizer.encode(text)
        # Restituisce la lista degli ID
        return encoding.ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Converte una lista di ID di token di nuovo in una stringa di testo
        """
        # Controlla se il tokenizer è stato addestrato
        if not self.id_to_token:
            raise ValueError("Il tokenizer non è stato addestrato")

        # Filtra via i token di padding (<PAD>) prima della decodifica, perché non fanno parte del contenuto reale
        token_ids = [tid for tid in token_ids if tid != self.token_to_id.get('<PAD>', 0)]

        # Usa il metodo 'decode' del tokenizer di HuggingFace per la decodifica
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        return text

    def get_vocab_size(self) -> int:
        """Restituisce la dimensione attuale del vocabolario"""
        return len(self.token_to_id)

    @property
    def vocab(self):
        """
        Proprietà per accedere al dizionario del vocabolario (token -> id).
        Utile per mantenere la compatibilità con altre librerie che si aspettano un campo '.vocab'.
        """
        return self.token_to_id

    def get_special_token_id(self, token: str) -> int:
        """
        Restituisce l'ID di un token speciale (es. '<PAD>').
        Se il token non viene trovato, restituisce l'ID del token sconosciuto (<UNK>).
        """
        return self.token_to_id.get(token, self.token_to_id.get('<UNK>', 1))
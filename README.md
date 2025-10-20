# NanoSocrates: A Very Small Semantic Language Model

**NanoSocrates** is an Encoder-Decoder Transformer-based semantic language model, built from scratch with PyTorch. This project, developed for the Deep Learning course at the Politecnico di Bari, aims to explore the fundamental principles of Foundational Models by creating a unified model capable of translating between unstructured natural language text and structured data in the form of RDF triples.

The model is trained in the movie domain, using data from DBpedia and Wikipedia, and is designed to perform four distinct but related tasks, demonstrating its ability to understand and generate both text and structured knowledge.

## ✨ Key Features

The NanoSocrates model is designed to be versatile and handle multiple semantic tasks with a single architecture:

1.  **Text-to-RDF (Text2RDF) - RDF Generation on the Decoder side**: Converts a natural language text (the film's abstract) into a set of RDF triples that capture its semantic meaning.
2.  **RDF-to-Text (RDF2Text) - Text Generation on the Decoder side**: Generates a coherent and human-readable sentence from an RDF triples.
3.  **RDF Completion 1 - Masked Language Modeling with Spanned Masking**: Predicts missing components (subject, predicate, or object) within RDF triples that have been masked with the `<MASK>` token, a task analogous to link prediction in knowledge graphs.
4.  **RDF Completion 2 - RDF Generation on the Decoder side**: Generates new RDF triples that logically follow a given set of triples, similar to knowledge graph completion.

Furthermore, the implementation includes:

  - **Flexible Architecture**: Allows for easy switching between different attention mechanisms, including standard **Multi-Head Attention (MHA)**, **Multihead Latent Attention (MLA)** (inspired by DeepSeek-V2), and a hybrid **Alternating Attention** mode.
  - **Modern Positional Embeddings**: The model flexibly handles positional information. It uses classic **Sinusoidal Positional Encoding** for the standard MHA architecture and the more advanced Decoupled Rotary Position Embeddings (RoPE) when using MLA or Alternating Attention.
  - **Custom Tokenizer**: Trains a **Byte-Pair Encoding (BPE)** tokenizer from scratch on the specific movie corpus, optimized to handle both natural language and the syntax of RDF triples.

## 🏗️ Model Architecture

Model Architecture is illustrated below:

<center>
    <img src = "https://miro.medium.com/v2/resize:fit:1100/format:webp/1*s5XcjuosS8ohfsW5xFT3sQ.png" width = 400>
<p style = "font-size: 16px;
            font-family: 'Georgia', serif;
            text-align: center;
            margin-top: 10px;">Source: <a href = "https://arxiv.org/pdf/1706.03762.pdf">Attention Is All You Need</a>
</center>

NanoSocrates is based on a standard Encoder-Decoder Transformer architecture, with several improvements and configuration options.

  - **Encoder**: Its function is to create a rich contextual representation of the input sequence, whether it is text or serialized RDF triples.
  - **Decoder**: Generates the target sequence autoregressively, conditioned on the encoder's output.

### Attention Mechanisms

The model's self-attention layers are highly configurable, allowing you to switch between different implementations by modifying the `attention_mode` variable within the `Config` class in `nanosocrates.py`.

```python
# In nanosocrates.py
class Config:
    # ...
    # Flag to select the attention mechanism for the architecture.
    # Options: 'MHA', 'MLA', 'ALTERNATING'
    attention_mode = 'MLA'
    # ...
```

The available options are:

  - **`MHA` (Multi-Head Attention)**: The classic and standard implementation.
  - **`MLA` (Multihead Latent Attention)**: A more efficient variant that compresses the Key and Value matrices into a lower-dimensional latent space before computing attention. This significantly reduces computational cost, especially with long sequences.
  - **`ALTERNATING` (Alternating Attention)**: A hybrid architecture that alternates between MHA and MLA layers to balance performance and efficiency.

The cross-attention in the decoder always uses standard MHA.

## 📊 Dataset

The dataset is dynamically built by combining information from two public knowledge sources: Wikipedia and DBpedia.

1.  **DBpedia**: Its SPARQL endpoint is used to obtain a list of movie entities (`dbo:Film`) and extract all their associated RDF triples. A filter is applied to keep only the most informative predicates (director, actor, genre, etc.).
2.  **Wikipedia**: For each movie, the abstract from its corresponding English Wikipedia page is fetched via the Wikipedia API.

The `dataset_creator.py` script automates the entire process, from data collection to cleaning and saving it to a CSV file.

## 📂 Project Structure

```
.
├── datasets/               # Directory (auto-created) to save datasets in CSV format.
├── runs/                   # Directory (auto-created) for TensorBoard logs.
├── weights/                # Directory (auto-created) to save model checkpoints.
├── nanosocrates.py         # Main script: model implementation, training, and validation loop.
├── dataset_creator.py      # Script to collect and preprocess data from DBpedia and Wikipedia.
├── tokenizer.py            # Custom BPE tokenizer implementation using the `tokenizers` library.
├── requirements.txt        # Python dependencies needed to run the project.
├── tokenizer.json          # The saved, trained BPE tokenizer for reuse.
└── README.md               # This file, providing project documentation.
```

## 🚀 Installation and Usage

Follow these steps to set up and run the project.

### 1\. Clone the repository

```bash
git clone https://github.com/Michelemata/NanoSocrates.git
cd NanoSocrates
```

### 2\. Install dependencies

It is recommended to create a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3\. Run the main script

To start the entire process (data collection, tokenizer training, and model training), run:

```bash
python nanosocrates.py
```

  - **Data Collection**: The first time you run it, the script will ask if you want to use an existing dataset (if it finds one in the `datasets/` folder). If not, it will proceed to collect new data, a process that may take several minutes.
  - **Training**: The script will start the training loop, showing progress by epochs.
  - **Results**: Upon completion, the following will be saved:
      - Model weights in the `weights/` folder.
      - Logs for TensorBoard in the `runs/` folder.
      - Loss curve plots (`loss_total_curve.png` and `loss_per_task_curve.png`) in the root directory.

## ⚙️ Configuration

All hyperparameters for the model, training, and file paths can be modified in the `Config` class within `nanosocrates.py`.

## 📈 Evaluation

The model's performance is evaluated separately for each of the four tasks at the end of each validation epoch. The metrics used are:

  - **RDF2Text**: BLEU, ROUGE, and METEOR.
  - **Text2RDF**: Precision, Recall, and F1-score at the triple level.
  - **RDF Completion 1**: Accuracy in predicting the masked tokens.
  - **RDF Completion 2**: Precision, Recall, and F1-score at the triple level.

The results are printed to the console and logged to TensorBoard. At the end of the training, the following loss plots are generated:

## Acknowledgements

This project was carried out as part of the **Deep Learning** course taught by Professor **Vito Walter Anelli, Ph.D.** at the **Politecnico di Bari**.

## 📜 License

This project is released under the MIT License. See the `LICENSE` file for more details.

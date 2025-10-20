import pandas as pd
import requests
from typing import Dict, List, Optional
import time
from urllib.parse import quote, unquote
import os


class DBpediaCollector:
    """
    Questa classe si occupa di raccogliere dati sui film da DBpedia e Wikipedia.
    Le sue responsabilità includono:
    1. Eseguire query SPARQL su DBpedia per ottenere URI di film e le loro triple RDF.
    2. Interrogare l'API di Wikipedia per ottenere gli abstract dei film.
    3. Pulire e formattare i dati raccolti.
    4. Salvare e caricare il dataset in formato CSV.
    """

    def __init__(self):
        # URL dell'endpoint di DBpedia per le query SPARQL
        self.sparql_endpoint = "https://dbpedia.org/sparql"
        # Crea una sessione di richieste per riutilizzare le connessioni HTTP
        self.session = requests.Session()
        # Imposta uno User-Agent per identificare lo script e l'header Accept per richiedere JSON
        self.session.headers.update({
            'User-Agent': 'DBpedia Movie Dataset Creator/1.0',
            'Accept': 'application/sparql-results+json'
        })

        # Definisce un insieme di predicati RDF ritenuti più informativi per descrivere un film,
        # in modo da filtrare le triple RDF, mantenendo solo quelle più rilevanti.
        self.important_predicates = {
            # Proprietà principali del film
            'http://dbpedia.org/ontology/director',             # regista
            'http://dbpedia.org/ontology/starring',             # attori principali
            'http://dbpedia.org/ontology/producer',             # produttore
            'http://dbpedia.org/ontology/writer',               # sceneggiatore
            'http://dbpedia.org/ontology/distributor'           # distributore
            'http://dbpedia.org/ontology/productionCompany',    # casa di produzione

            # Dettagli del film
            'http://dbpedia.org/ontology/releaseDate',          # data di uscita
            'http://dbpedia.org/ontology/runtime',              # durata
            'http://dbpedia.org/ontology/genre',                # genere
            'http://dbpedia.org/ontology/award',                # premi vinti

            # Relazioni con altre opere
            'http://dbpedia.org/ontology/basedOn',              # Basato su (es. un libro)
            'http://dbpedia.org/ontology/sequel',               # Sequel
            'http://dbpedia.org/ontology/prequel',              # Prequel
            'http://dbpedia.org/ontology/series',               # Parte di una serie
        }

    def get_movie_entities(self, limit: int = 10000, offset: int = 0) -> List[str]:
        """
        Esegue una query SPARQL su DBpedia per recuperare un elenco di entità (URI)
        che sono classificate come film (dbo:Film) e hanno un abstract in inglese.
        """
        # Query SPARQL per selezionare URI di film
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?movie WHERE {{
            ?movie rdf:type dbo:Film .
            ?movie dbo:abstract ?abstract .
            FILTER(LANG(?abstract) = "en")
        }}
        LIMIT {limit}
        OFFSET {offset}
        """

        try:
            # esegue le GET all'endpoint SPARQL
            response = self.session.get(
                self.sparql_endpoint,
                params={'query': query, 'format': 'json'},  # Passa la query e richiede il formato JSON
                timeout=30                                  # Imposta un timeout di 30 secondi
            )
            response.raise_for_status()                     # Lancia un'eccezione se la risposta è un errore HTTP

            # Estrae i risultati dal JSON di risposta
            results = response.json()
            movies = [result['movie']['value'] for result in results['results']['bindings']]
            print(f"Trovati {len(movies)} film (offset: {offset})")
            return movies

        except Exception as e:
            # In caso di errore durante la query, restituisce una lista vuota
            print(f"Errore durante la query per i film: {e}")
            return []

    def filter_important_triples(self, triples: List[Dict]) -> List[Dict]:
        """
        Filtra una lista di triple per mantenere solo quelle il cui predicato
        è presente nella lista 'self.important_predicates'.
        """
        filtered_triples = []

        for triple in triples:
            predicate = triple['predicate']

            # Mantiene la tripla se il predicato è nell'insieme di quelli importanti
            if predicate in self.important_predicates:
                filtered_triples.append(triple)
            # Mantiene anche triple che usano vocabolari esterni comuni (per completezza)
            elif (predicate.startswith('http://schema.org/') or
                  predicate.startswith('http://purl.org/dc/terms/') or
                  predicate.startswith('http://www.wikidata.org/prop/')):
                filtered_triples.append(triple)

        return filtered_triples

    def get_movie_triples(self, movie_uri: str, max_triples: int = 20) -> List[Dict]:
        """
        Recupera le triple RDF per un dato URI di un film, filtrando per i predicati importanti.
        Esegue due tipi di query:
        1. Triple dirette: dove il film è il soggetto (es. <film> <regista> <nome_regista>).
        2. Triple inverse: dove il film è l'oggetto (es. <sequel> <è_sequel_di> <film>).
        """
        # Converte l'insieme dei predicati in una lista per poterla dividere in blocchi a causa
        # dei limiti di lunghezza delle query SPARQL
        predicate_chunks = list(self.important_predicates)

        triples = []

        try:
            # Esegue le query in blocchi ("chunk") per evitare che la stringa della query diventi troppo lunga,
            # cosa che potrebbe causare un errore "URI too long" da parte del server.
            for i in range(0, len(predicate_chunks), 15):
                chunk = predicate_chunks[i:i + 15]
                # Crea una clausola FILTER per la query SPARQL che accetta qualsiasi predicato nel blocco
                predicate_filter = ' || '.join([f'?p = <{pred}>' for pred in chunk])

                # Query per le triple dirette
                direct_query = f"""
                SELECT ?p ?o WHERE {{
                    <{movie_uri}> ?p ?o .
                    FILTER({predicate_filter})
                }}
                """

                # Esegue le GET all'endpoint SPARQL
                response = self.session.get(
                    self.sparql_endpoint,
                    params={'query': direct_query, 'format': 'json'},   # Passa la query e richiede il formato JSON
                    timeout=30                                          # Imposta un timeout di 30 secondi
                )
                response.raise_for_status() # Lancia un'eccezione se la risposta è un errore HTTP

                results = response.json()
                # Aggiunge le triple trovate alla lista
                for result in results['results']['bindings']:
                    triple = {
                        'subject': movie_uri,
                        'predicate': result['p']['value'],
                        'object': result['o']['value'],
                        'object_type': result['o'].get('type', 'uri') # Indica se l'oggetto è un letterale o un altro URI
                    }
                    triples.append(triple)

                time.sleep(0.1)  # Piccola pausa per non sovraccaricare il server di DBpedia

            # Query per le triple inverse (dove il film è l'oggetto)
            # Si cercano solo relazioni molto specifiche come sequel/prequel o persone legate al film
            reverse_query = f"""
            SELECT ?s ?p WHERE {{
                ?s ?p <{movie_uri}> .
                FILTER(?p IN (
                    <http://dbpedia.org/ontology/director>,
                    <http://dbpedia.org/ontology/starring>,
                    <http://dbpedia.org/ontology/sequel>,
                    <http://dbpedia.org/ontology/prequel>,
                    <http://dbpedia.org/property/director>,
                    <http://dbpedia.org/property/starring>
                ))
            }}
            LIMIT 50
            """

            # esegue le GET all'endpoint SPARQL
            response = self.session.get(
                self.sparql_endpoint,
                params={'query': reverse_query, 'format': 'json'},  # Passa la query e richiede il formato JSON
                timeout=30                                          # Imposta un timeout di 30 secondi
            )
            response.raise_for_status() # Lancia un'eccezione se la risposta è un errore HTTP

            results = response.json()
            # Aggiunge le triple trovate alla lista
            for result in results['results']['bindings']:
                triple = {
                    'subject': result['s']['value'],
                    'predicate': result['p']['value'],
                    'object': movie_uri,
                    'object_type': 'uri'    # L'oggetto è sempre il nostro film (un URI)
                }
                triples.append(triple)

        except Exception as e:
            print(f"Errore nel recuperare le triple per {movie_uri}: {e}")

        # Filtra ulteriormente le triple raccolte per assicurarsi che siano rilevanti
        filtered_triples = self.filter_important_triples(triples)

        return filtered_triples


    def extract_movie_title(self, movie_uri: str) -> str:
        """
        Estrae un titolo di film leggibile dall'URI di DBpedia.
        Esempio: 'http://dbpedia.org/resource/The_Matrix' -> 'The Matrix'
        """
        # Rimuove il prefisso dell'URI
        title = movie_uri.replace('http://dbpedia.org/resource/', '')
        # Decodifica eventuali caratteri speciali (es. %20) e sostituisce i trattini bassi con spazi
        title = unquote(title).replace('_', ' ')
        return title

    def get_wikipedia_title(self, movie_uri: str) -> Optional[str]:
        """
        Recupera il titolo della pagina Wikipedia inglese corrispondente ad un URI di DBpedia.
        Usa la proprietà foaf:isPrimaryTopicOf per trovare il link.
        """
        # Query per trovare il link alla pagina Wikipedia
        title_query = f"""
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX dbo: <http://dbpedia.org/ontology/>

        SELECT ?wikipediaPage WHERE {{
            <{movie_uri}> foaf:isPrimaryTopicOf ?wikipediaPage .
            FILTER(STRSTARTS(STR(?wikipediaPage), "https://en.wikipedia.org/"))
        }}
        LIMIT 1
        """

        try:
            # esegue le GET all'endpoint SPARQL
            response = self.session.get(
                self.sparql_endpoint,
                params={'query': title_query, 'format': 'json'},    # Passa la query e richiede il formato JSON
                timeout=30                                          # Imposta un timeout di 30 secondi
            )
            response.raise_for_status() # Lancia un'eccezione se la risposta è un errore HTTP

            results = response.json()
            # Se ci sono risultati, estrae il titolo dall'URL
            if results['results']['bindings']:
                wiki_url = results['results']['bindings'][0]['wikipediaPage']['value']
                # Estrae il titolo dall'URL: https://en.wikipedia.org/wiki/Movie_Title
                title = wiki_url.split('/')[-1]
                # Decodifica il titolo
                return unquote(title)

        except Exception as e:
            print(f"Errore nel recuperare il titolo Wikipedia per {movie_uri}: {e}")

        # Se la query fallisce o non dà risultati, estrae il titolo direttamente dall'URI di DBpedia
        return self.extract_movie_title(movie_uri)

    def get_wikipedia_abstract(self, movie_uri: str) -> Optional[str]:
        """
        Recupera l'abstract di un film dalla REST API di Wikipedia.
        """
        # Prima ottiene il titolo corretto della pagina Wikipedia
        wiki_title = self.get_wikipedia_title(movie_uri)
        if not wiki_title:
            return None

        # Endpoint dell'API di Wikipedia
        wikipedia_api = "https://en.wikipedia.org/api/rest_v1/page/summary/"

        try:
            # Esegue la richiesta GET, facendo l'escape del titolo per l'URL
            response = self.session.get(
                f"{wikipedia_api}{quote(wiki_title)}",
                timeout=30,
                headers={'User-Agent': 'DBpedia Movies Dataset Creator/1.0'}
            )
            # Se la richiesta ha successo (status 200)
            if response.status_code == 200:
                data = response.json()
                # Estrae il campo 'extract', che contiene l'abstract
                extract = data.get('extract', '')
                if extract:
                    return extract

            elif response.status_code == 404:
                print(f"Pagina Wikipedia non trovata per: {wiki_title}")
            else:
                print(f"L'API di Wikipedia ha restituito {response.status_code} per: {wiki_title}")

        except Exception as e:
            print(f"Errore nel recuperare l'abstract da Wikipedia per {wiki_title}: {e}")

        return None

    # def get_wikipedia_text(self, movie_uri: str) -> str:
    #     """
    #     Ottiene il riassunto di Wikipedia per un film.
    #     Questa funzione è un semplice "wrapper" mantenuto per compatibilità
    #     con versioni precedenti del codice. Chiama la funzione più completa
    #     `get_wikipedia_abstract` e si limita a troncare il risultato.
    #     """
    #     abstract = self.get_wikipedia_abstract(movie_uri)
    #     return abstract[:500] if abstract else ""  # Limit length
    #
    # def triples_to_text(self, triples: List[Dict]) -> str:
    #     """
    #     Convert triples to a readable text format.
    #     """
    #     formatted = []
    #     for triple in triples:
    #         # Clean up URIs for readability
    #         subject = self.clean_uri(triple['subject'])
    #         predicate = self.clean_uri(triple['predicate'])
    #
    #         if triple['object_type'] == 'literal':
    #             obj = f'"{triple["object"]}"'
    #         else:
    #             obj = self.clean_uri(triple['object'])
    #
    #         formatted.append(f"{subject} {predicate} {obj}")
    #
    #     return "\n".join(formatted)

    def clean_uri(self, uri: str) -> str:
        """
        Semplifica un URI completo sostituendo i prefissi comuni (namespace)
        con delle abbreviazioni (es. 'dbr:', 'dbo:').
        Questo rende le triple molto più compatte e leggibili.

        Esempio:
        'http://dbpedia.org/resource/The_Matrix' -> 'dbr:The_Matrix'
        'http://dbpedia.org/ontology/director' -> 'dbo:director'
        """
        if uri.startswith('http://dbpedia.org/resource/'):
            return uri.replace('http://dbpedia.org/resource/', 'dbr:')
        elif uri.startswith('http://dbpedia.org/ontology/'):
            return uri.replace('http://dbpedia.org/ontology/', 'dbo:')
        elif uri.startswith('http://dbpedia.org/property/'):
            return uri.replace('http://dbpedia.org/property/', 'dbp:')
        elif uri.startswith('http://www.w3.org/1999/02/22-rdf-syntax-ns#'):
            return uri.replace('http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'rdf:')
        elif uri.startswith('http://www.w3.org/2000/01/rdf-schema#'):
            return uri.replace('http://www.w3.org/2000/01/rdf-schema#', 'rdfs:')
        else:
            # Se l'URI non ha un prefisso noto, lo restituisce così com'è
            return uri

    def save_dataset_csv(dataset: List[Dict], filename: str = None) -> str:
        """
        Salva il dataset raccolto in un file CSV.
        Le triple vengono convertite in una singola stringa per essere salvate in una cella.
        """
        # Se non viene fornito un nome per il file, ne usa uno di default
        if filename is None:
            filename = f"movies_dataset.csv"

        # Assicura l'estensione .csv
        if not filename.endswith('.csv'):
            filename += '.csv'

        # Crea la cartella 'datasets' se non esiste
        os.makedirs('datasets', exist_ok=True)
        filepath = os.path.join('datasets', filename)

        # Prepara i dati per il salvataggio
        rows = []
        for i, item in enumerate(dataset):
            # Converte la lista di triple in una stringa formattata, separata da '|'
            triples_formatted = []
            for triple in item.get('triples', []):
                if len(triple) >= 3:
                    triples_formatted.append(f"({triple[0]}, {triple[1]}, {triple[2]})")

            triples_str = " | ".join(triples_formatted)

            # Aggiunge una riga al dataset
            rows.append({
                'id': i + 1,
                'movie_uri': item.get('movie_uri', ''),
                'text': item.get('text', ''),
                'text_length': len(item.get('text', '')),
                'num_triples': len(item.get('triples', [])),
                'triples': triples_str,
            })

        # Usa pandas per creare un DataFrame e salvarlo come CSV
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8')

        print(f"Dataset salvato:")
        print(f"   • CSV file: {filepath}")
        print(f"   • Film totali: {len(dataset)}")

        return filepath

    @staticmethod
    def load_dataset_csv(filename: str) -> List[Dict]:
        """
        Carica un dataset da un file CSV precedentemente salvato.
        Riconverte la stringa delle triple nella struttura dati originale (lista di tuple).
        """
        # Costruisce il percorso completo del file nella cartella 'datasets'
        filepath = os.path.join('datasets', filename)

        # Se il file non esiste, solleva un errore per fermare l'esecuzione
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File del dataset non trovato: {filepath}")

        # Legge il file CSV con pandas
        df = pd.read_csv(filepath, encoding='utf-8')
        dataset = []

        # Itera su ogni riga del DataFrame
        for _, row in df.iterrows():
            # Inizializza la lista che conterrà le triple per il film corrente
            triples = []
            # Estrae la stringa contenente tutte le triple (es. "(s,p,o) | (s2,p2,o2)")
            triples_str = str(row.get('triples', ''))
            # Procede solo se la cella delle triple non è vuota o 'nan'
            if triples_str and triples_str != 'nan':
                # Divide la stringa principale basandosi sul separatore ' | '
                triple_parts = triples_str.split(' | ')
                for triple_str in triple_parts:
                    # Controlla che la tripla sia formattata correttamente con le parentesi
                    if triple_str.startswith('(') and triple_str.endswith(')'):
                        # Per ogni tripla, rimuove le parentesi e divide per ', '
                        triple_content = triple_str[1:-1]
                        parts = [part.strip() for part in triple_content.split(', ')]
                        if len(parts) >= 3:
                            # Aggiunge la tupla (soggetto, predicato, oggetto) alla lista
                            triples.append((parts[0], parts[1], parts[2]))

            # Ricostruisce il dizionario originale per l'elemento del dataset
            dataset.append({
                'movie_uri': str(row.get('movie_uri', '')),
                'text': str(row.get('text', '')),
                'triples': triples,
            })

        return dataset

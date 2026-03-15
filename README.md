# Database of UNITS


1. Per creare lo schema de database (le tabelle sono: Insegnamento, Personale)
`python 01_create_schema.py`

2. Caricare i dati presenti nei file json dentro il database
`02_populate_db.py`

3. fare l'embedding delle colonne indicate, in modo da poter ricrere query esatte anche se la domanda dell'utente non è precisa
`03_create_rag_index.py`

4. programma per fare domande (richiede vllm locale)
`04_query.py`

Per effettuare una serie di domande e vedere anche i chunk recuperati
`test_sql.py`
# UniTS University Database — NL-to-SQL Query Engine

A natural language query system for the University of Trieste (UniTS) database.
It allows users to ask questions in plain Italian or English and get accurate answers
by combining vector search (RAG) with LLM-generated SQL queries.

<img src="asserts/clip_chainlit.gif" alt="chainlit" style="width:90%; height:auto;">

## Architecture

- **SQLite** database storing university data (courses, staff, lessons, classrooms, etc.)
- **ChromaDB** vector store for semantic column retrieval (column-level RAG)
- **LlamaIndex** `SQLTableRetrieverQueryEngine` for text-to-SQL generation
- **Local LLM** served via llama.cpp / LiteLLM (OpenAI-compatible API)
- **HuggingFace Embeddings** (`BAAI/bge-m3`) for multilingual semantic search

## Embedding Model

The system uses [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) (570M parameters) for all vector similarity searches.

This model was chosen over lighter alternatives (e.g. `intfloat/multilingual-e5-small`, 118M params)
because smaller models tend to match words based on surface-level lexical similarity rather than
true semantic meaning. A concrete example of this failure: when a user asks about *"geometria"*
(mathematics), a small model would also retrieve *"GEOMORFOLOGIA"* (geology) as a strong match,
simply because both words share the `GEO` prefix and have similar character sequences.
`bge-m3` handles these cases correctly by understanding the semantic difference between the two domains.

`bge-m3` is currently state-of-the-art for multilingual embeddings and handles mixed
Italian/English content robustly, which is important given that UniTS course names and
degree programs appear in both languages.
## Setup

### 1. Create the database schema
Creates all tables: `personale`, `insegnamento`, `lezione`, `evento_aula`, `info_aula`, `corso_di_laurea`.
```bash
python 01_create_schema.py
```

### 2. Populate the database
Loads data from JSON files into the database, with text normalization (apostrophe handling, whitespace collapsing).
```bash
python 02_populate_db.py
```
You can interact with the database SQLlite withe the following command: `sqlite3 2025-2026_data/university.db`

### 3. Build the RAG index
Embeds the values of key columns into ChromaDB collections.
This allows the system to match user queries to exact DB values even when phrasing differs
(e.g. "Martino Trevisan" → `TREVISAN MARTINO`).
```bash
python 03_create_rag_index.py
```

### 4. Run the query engine
Starts the interactive query interface. Requires a local LLM running on an OpenAI-compatible endpoint.
```bash
python 04_query.py
```

## Testing

Run a predefined set of test questions and inspect retrieved chunks, generated SQL, and final answers.
Results are saved to a markdown report with per-query timing breakdown.
```bash
python test_llm.py
# With custom paths:
python test_llm.py --db path/to/university.db --chroma-dir path/to/chroma_store --output results.md
```

## Requirements

- Python 3.11+ (I raccomand python 3.11 for problem with Chainlit)
- A local LLM server exposing an OpenAI-compatible `/v1` endpoint (e.g. llama.cpp, LiteLLM)
- See `requirements.txt` for Python dependencies

---

##  Test Results 

I tested some questions: the LLM generates the sql query, the machine executes it and the result is the new context for the same LLM (question + context + prompt). The second result is the response for the user in markdown format.

### 1. Who is Martino Trevisan?

```sql
SELECT * FROM personale WHERE UPPER(nome_and_surname) LIKE '%TREVISAN MARTINO%' LIMIT 40;
```

**Answer:** Prof. Martino Trevisan is an **Associate Professor** (Professore Associato) in the **Department of Engineering and Architecture** at the University of Trieste.  

- **Email:** martino.trevisan@dia.units.it  
- **Phone:** 040 558 3861  

(Record last updated on 06/03/2026.)


---

### 2. Who is Trevisan Martino?

```sql
SELECT id, nome_and_surname, role, department, email, phone FROM personale WHERE UPPER(nome_and_surname) LIKE '%TREVISAN MARTINO%' LIMIT 40;
```

**Answer:** Prof. Martino Trevisan is an **Associate Professor** in the **Department of Engineering and Architecture** at the University of Trieste.  

- **Email:** martino.trevisan@dia.units.it  
- **Phone:** +39 040 558 3861  

If you need further information about his teaching activities, research interests, or publications, please let me know.

---

### 3. Who is Trevisan?

```sql
SELECT id, nome_and_surname, role, department, email, phone FROM personale WHERE UPPER(nome_and_surname) LIKE '%TREVISAN%' LIMIT 40;
```

**Answer:** There are several members of the university community with the surname **Trevisan**. Here are the matches found in our personnel database:

| Full name | Role | Department | Email | Phone |
|-----------|------|------------|-------|-------|
| **Alberto Trevisan** | Dottorando di ricerca | Dipartimento di Ingegneria e Architettura | s268118@stud.units.it | N/A |
| **Martino Trevisan** | Professore Associato | Dipartimento di Ingegneria e Architettura | martino.trevisan@dia.units.it | 040 558 3861 |
| **Piergiorgio Trevisan** | Professore Associato | Dipartimento di Studi Umanistici | ptrevisan@units.it | 040 558 4413 |
| **Matilde Trevisani** (note the different spelling) | Professore Associato | Dipartimento di Scienze Economiche, Aziendali, Matematiche e Statistiche | MATILDE.TREVISANI@deams.units.it | 040 558 2519 |

Could you please specify which **Trevisan** you are interested in (e.g., full name or department) so I can provide more detailed information?

---

### 4. Who is professor Trevisan of Computer Science?

```sql
SELECT DISTINCT professors FROM insegnamento WHERE UPPER(professors) LIKE '%TREVISAN%' AND UPPER(degree_program_name) LIKE '%COMPUTER ENGINEERING%' LIMIT 40;
```

**Answer:** Assuming you are referring to **Prof. Martino Trevisan** (Computer Engineering / Computer Science), here is the information we have: he is listed as a professor in the Computer Engineering area at the University of Trieste.


---

### 5. What course does professor Paolo Vercesi teach?

```sql
SELECT subject_name, professors FROM insegnamento WHERE UPPER(professors) LIKE '%VERCESI PAOLO%' LIMIT 40;
```

**Answer:** Assuming you are referring to Prof. Paolo Vercesi, he teaches the course **“Software Development Methods.”**


---

### 6. What courses are taught by Vercesi Paolo

```sql
SELECT subject_name, degree_program_name, professors FROM insegnamento WHERE UPPER(professors) LIKE '%VERCESI PAOLO%' LIMIT 40;
```

**Answer:** Assuming you are referring to **Prof. Paolo Vercesi**, here are the courses he teaches:

- **Software Development Methods** – offered in the following degree programs:  
  • Computer Engineering  
  • Mathematics  
  • Data Science and Artificial Intelligence  

*Note: The same subject appears in multiple programmes; it is listed once with all relevant programmes indicated.*

---

### 7. What courses are taught by De Lorenzo?

```sql
SELECT subject_name, degree_program_name, teams_code, professors
FROM insegnamento
WHERE UPPER(professors) LIKE '%DE LORENZO ANDREA%'
LIMIT 40;
```

**Answer:** Assuming you are referring to **Prof. Andrea De Lorenzo** (department information not available in the current data), the courses he teaches are:

- **BASI DI DATI** – offered in the degree programs:
  - Ingegneria Elettronica e Informatica (team code 2pnfmr1)  
  - Intelligenza Artificiale e Data Analytics (team code N/A)  

- **WEB APPLICATION PROGRAMMING** – Computer Engineering (team code 2ajhwxg)

- **DATABASE** – Geophysics and Geodata (team code N/A)

These are the subjects associated with Prof. De Lorenzo in the university database. If you need further details (e.g., specific lecture times, rooms, or additional professors), please let me know.

---

### 8. What courses are taught by De Lorenzo of the Department of Engineering?

```sql
SELECT subject_name, degree_program_name, professors, teams_code FROM insegnamento WHERE UPPER(professors) LIKE '%DE LORENZO ANDREA%' LIMIT 40;
```

**Answer:** Assuming you are referring to **Prof. Andrea De Lorenzo** (Department of Engineering) — here are the courses he teaches, grouped by subject and indicating the degree programs in which each subject is offered:

- **BASI DI DATI** – offered in:
  * Ingegneria Elettronica e Informatica  
  * Intelligenza Artificiale e Data Analytics  

- **WEB APPLICATION PROGRAMMING** – offered in:
  * Computer Engineering  

- **DATABASE** – offered in:
  * Geophysics and Geodata  

If you need further details (e.g., specific class codes, schedules, or additional subjects), please let me know!

---

### 9. What courses are taught by De Lorenzo?

```sql
SELECT subject_name, degree_program_name, professors
FROM insegnamento
WHERE UPPER(professors) LIKE '%DE LORENZO ANDREA%'
LIMIT 40;
```

**Answer:** Assuming you are referring to **Prof. Andrea De Lorenzo**, here are the courses he teaches (grouped by subject and showing all degree programs in which each subject appears):

- **BASI DI DATI** – offered in:
  - Ingegneria Elettronica e Informatica  
  - Intelligenza Artificiale e Data Analytics  

- **WEB APPLICATION PROGRAMMING** – offered in:
  - Computer Engineering  

- **DATABASE** – offered in:
  - Geophysics and Geodata  

These are the distinct subjects associated with Prof. De Lorenzo in the current database. If you need further details (e.g., specific lecture times, rooms, or additional professors), please let me know.

---

### 10. What courses are taught by professors with the surname De Lorenzo?

```sql
SELECT DISTINCT subject_name, degree_program_name, professors FROM insegnamento WHERE UPPER(professors) LIKE '%DE LORENZO%' LIMIT 40;
```

**Answer:** I found two different professors with the surname **De Lorenzo** in our records:

- **Prof. Andrea De Lorenzo** – appears as the instructor for several courses (e.g., *Basi di Dati*, *Web Application Programming*, *Database*).  
- **Prof. Giuditta De Lorenzo** – appears together with another professor on the course *Processi Biochimici in Microbiologia*.

Could you let me know which professor you are interested in, or if you would like to see the courses for both of them?

---

### 11. What lessons are there on Friday, March 13, for students of Computer Engineering curriculum Informatics?

```sql
SELECT date, start_time, end_time, subject_name, room_name, site_name
FROM lezione
WHERE date = '2026-03-13'
  AND UPPER(degree_program_name) LIKE '%COMPUTER ENGINEERING%'
  AND (UPPER(curriculum) LIKE '%INFORMATICS%' OR UPPER(curriculum) LIKE '%INFORMATICA%')
LIMIT 40;
```

**Answer:** Here are the lessons scheduled for **Friday, 13 March 2026** for students enrolled in the Computer Engineering curriculum (Informatics):

| Time          | Subject                     | Room      | Building |
|---------------|----------------------------|-----------|----------|
| 11:00 – 13:00 | Cybersecurity               | Aula A    | Edificio C7 |
| 14:00 – 17:00 | Complexity and Cryptography | Sala Atti | Edificio C1 |

These are the only entries returned for the specified date and curriculum. If you need information for a different date or additional details, please let me know.

---

### 12. What lessons were there yesterday, for students of Computer Engineering curriculum Informatics?

```sql
SELECT date, start_time, end_time, subject_name, room_name, site_name FROM lezione WHERE date = '2026-03-28' AND UPPER(degree_program_name) LIKE '%COMPUTER ENGINEERING%' AND UPPER(curriculum) LIKE '%INFORMATICS%' LIMIT 40;
```

**Answer:** I’m sorry, but there are no lessons recorded for yesterday (2026‑03‑28) that match the Computer Engineering curriculum Informatics criteria in our database.

If you think the date or the curriculum name might be different, please try specifying another date or providing the exact wording of the curriculum you’re interested in. I’ll be happy to look it up for you.

---

### 13. Are there any geometry classes on Thursday?

```sql
SELECT date, start_time, end_time, subject_name, room_name, site_name FROM lezione WHERE date = '2026-03-19' AND UPPER(subject_name) LIKE '%GEOMETRIA%' LIMIT 40;
```

**Answer:** Yes. In the reference week for the second semester (16 – 20 March 2026) there is a geometry class on Thursday, 19 March 2026:

- **Subject:** Geometria 2  
- **Time:** 11:00 – 13:00  
- **Room:** Aula 3B, Edificio H2bis  

If you need information for a different week or additional details, just let me know.

---

### 14. Tell me all the degree programs (name and type) of the Department of Engineering and Architecture

```sql
SELECT name, type FROM corso_di_laurea WHERE UPPER(department) LIKE '%DIPARTIMENTO DI INGEGNERIA E ARCHITETTURA%' LIMIT 40;
```

**Answer:** Here are the degree programs offered by the **Department of Engineering and Architecture**, together with their type:

- **Computer Engineering** – Corso di Laurea Magistrale  
- **Engineering for the Energy Transition** – Corso di Laurea Magistrale  
- **Ingegneria Civile** – Corso di Laurea Magistrale  
- **Ingegneria Clinica** – Corso di Laurea Magistrale  
- **Ingegneria dell'Energia Elettrica e dei Sistemi** – Corso di Laurea Magistrale  
- **Ingegneria Meccanica** – Corso di Laurea Magistrale  
- **Ingegneria Navale** – Corso di Laurea Magistrale  
- **Materials and Chemical Engineering for Nano, Bio, and Sustainable Technologies** – Corso di Laurea Magistrale  
- **Architettura** – Laurea Magistrale Ciclo Unico 5 anni  
- **Ingegneria Civile e Ambientale** – Corso di Laurea Triennale  
- **Ingegneria Elettronica e Informatica** – Corso di Laurea Triennale  
- **Ingegneria Industriale** – Corso di Laurea Triennale  
- **Ingegneria Navale** – Corso di Laurea Triennale  

These are the programs currently listed for the department. If you need more details about any specific program, feel free to ask!

---

### 15. Tell me all the exams in the calendar for the Advanced Internet Technologies subject that were held in February 2026

```sql
SELECT date, start_time, end_time, name_event, room_name, site_name FROM evento_aula WHERE UPPER(name_event) LIKE '%ADVANCED INTERNET TECHNOLOGIES%' AND UPPER(event_type) LIKE '%ESAME%' AND date BETWEEN '2026-02-01' AND '2026-02-29' LIMIT 40;
```

**Answer:** Here are the exams for **Advanced Internet Technologies** that were scheduled in February 2026:

| Date | Time | Event (as listed) | Room | Building |
|------|------|-------------------|------|----------|
| 2026‑02‑03 | 10:00 – 12:00 | ADVANCED INTERNET TECHNOLOGIES [502MI] – Prof. Trevisan | Aula A | Edificio C2 |
| 2026‑02‑05 | 10:00 – 12:00 | ADVANCED INTERNET TECHNOLOGIES [502MI] Orale – Prof. Trevisan | Aula B | Edificio C8 |
| 2026‑02‑17 | 10:00 – 12:00 | ADVANCED INTERNET TECHNOLOGIES [502MI] – Prof. Trevisan | Aula A | Edificio C2 |
| 2026‑02‑19 | 10:00 – 12:00 | ADVANCED INTERNET TECHNOLOGIES [502MI] Orale – Prof. Trevisan | Aula B | Edificio C8 |

These are all the exam events matching the subject name that took place in February 2026. If you need further details or information about other subjects, feel free to ask.
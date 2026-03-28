"""
02_insert_data.py
-----------------

Usage:
    python 02_insert_data.py
"""

import json
import sqlite3
from pathlib import Path
from utils import normalize_text

DEFAULT_PERSONALE           =    "2025-2026_data/address_book.json"
DEFAULT_INSEGNAMENTO        =    "2025-2026_data/courses_with_teams_code.json"
DEFAULT_CORSO_DI_LAUREA     =    "2025-2026_data/full_degree_programs.json"
DEFAULT_LEZIONI_DIR         =    "2025-2026_data/lessons_calendar/"
DEFAULT_CALENDARIO_AULE_DIR =    "2025-2026_data/rooms_calendar/"
DEFAULT_INFO_AULE           =    "2025-2026_data/info_rooms.json"

DEFAULT_DB                  =    "2025-2026_data/university.db"

# ---------------------------------------------------------------------------
# Loaders

def load_personale(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", data) if isinstance(data, dict) else data

    rows = []
    for entry in entries:
        meta = entry.get("metadata", {})
        rows.append({
            "nome_and_surname":     normalize_text(meta.get("nome")),
            "role":                 normalize_text(meta.get("role")),
            "department":           normalize_text(meta.get("department")),
            "department_url":         meta.get("department_url"),   # URL — do not normalize
            "phone":                  meta.get("phone"),            # phone number — do not normalize
            "email":                  meta.get("email"),            # email — do not normalize
            "last_update":            meta.get("last_updated"),
        })
    return rows


def load_insegnamento(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", data) if isinstance(data, dict) else data

    rows = []
    for entry in entries:
        meta = entry.get("metadata", {})
        rows.append({
            "subject_code":              meta.get("AF_ID"),                          # 535588

            "degree_program_name":       normalize_text(meta.get("degree_program")),     # "COMPUTER ENGINEERING"
            "degree_program_name_eng":   normalize_text(meta.get("degree_program_eng")), # "COMPUTER ENGINEERING"
            "degree_program_code":         meta.get("degree_program_code"), # IN23 — code, do not normalize

            "subject_name":              normalize_text(meta.get("course_name")),        # COMPLEXITY AND CRYPTOGRAPHY

            "study_code":                  meta.get("course_code"),         # 504MI — code, do not normalize
            "academic_year":               meta.get("academic_year"),

            "teams_code":                  meta.get("teams_code"),          # Teams code — do not normalize
            "professors":                normalize_text(meta.get("teacher_name")),
            "main_professor_id":           meta.get("teacher_id"),
            "period":                      meta.get("period"),              # e.g. "S1" — code, do not normalize
            "last_update":                 meta.get("last_update"),
        })
    return rows


def load_corso_di_laurea(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", data) if isinstance(data, dict) else data

    rows = []
    for meta in entries:
        # Normalize equipment string values but keep the JSON structure intact
        equipment = meta.get("equipment")
        if isinstance(equipment, dict):
            equipment = {k: normalize_text(v) if isinstance(v, str) else v for k, v in equipment.items()}

        rows.append({
            "name":            normalize_text(meta.get("name")),
            "category":        normalize_text(meta.get("category")),
            "department":      normalize_text(meta.get("department")),
            "type":            normalize_text(meta.get("type")),
            "duration":        normalize_text(meta.get("duration")),
            "location":        normalize_text(meta.get("location")),
            "language":        normalize_text(meta.get("language")),
            "url":             meta.get("link"),               # URL — do not normalize
        })
    return rows


def load_lezioni(lezioni_dir: Path) -> list[dict]:
    """
    Parse all JSON files inside lezioni_dir.
    Each file contains a list of lesson objects.
    """

    if not lezioni_dir.exists() or not lezioni_dir.is_dir():
        raise FileNotFoundError(f"Events directory not found: {lezioni_dir}")

    json_files = sorted(lezioni_dir.glob("*.json"))
    if not json_files:
        print(f"[lezione]       no JSON files found in {lezioni_dir}")
        return []

    rows = []

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        entries = data if isinstance(data, list) else data.get("entries", [])

        for meta in entries:

            rows.append({
                "subject_code":          meta.get("subject_code"),          # EC535583 — code

                "degree_program_name":   normalize_text(meta.get("degree_program_name")),
                "degree_program_code":     meta.get("degree_program_code"), # IN23 — code

                "subject_name":          normalize_text(meta.get("subject_name")),

                "study_year_code":         meta.get("study_year_code"),     # IN23+1+|1 — code
                "curriculum":            normalize_text(meta.get("curriculum")),

                "date":                    meta.get("date"),                # ISO date — do not normalize
                "start_time":              meta.get("start_time"),          # HH:MM — do not normalize
                "end_time":                meta.get("end_time"),
                "department":            normalize_text(meta.get("department")),

                "room_code":               meta.get("room_code"),           # 035_2 — code
                "room_name":             normalize_text(meta.get("room_name")),
                "site_code":               meta.get("site_code"),           # AA01 — code
                "site_name":             normalize_text(meta.get("site_name")),
                "address":               normalize_text(meta.get("address")),

                "professors":            normalize_text(meta.get("professors")),
                "cancelled":               meta.get("cancelled", "no"),
                "url":                     meta.get("url"),                 # URL — do not normalize
            })

    print(f"[lezione]       loaded {len(rows):>6} rows from {len(json_files)} file(s)")
    return rows

def load_calendario_aule(calendario_aule_dir: Path) -> list[dict]:
    """
    Parse all JSON files inside calendario_aule_dir.
    Each file is a list of event objects with a 'metadata' key.
    """
    if not calendario_aule_dir.exists() or not calendario_aule_dir.is_dir():
        raise FileNotFoundError(f"Events directory not found: {calendario_aule_dir}")

    json_files = sorted(calendario_aule_dir.glob("*.json"))
    if not json_files:
        print(f"[lezione]       no JSON files found in {calendario_aule_dir}")
        return []

    rows = []
    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        entries = data if isinstance(data, list) else data.get("entries", [])

        for entry in entries:
            rows.append({
                "site_code":     entry.get("site_code"),            # code
                "room_code":     entry.get("room_code"),            # code
                "site_name":     normalize_text(entry.get("site_name")),
                "room_name":     normalize_text(entry.get("room_name")),
                "date":          entry.get("date"),                 # ISO date
                "last_update":   entry.get("last_update"),
                "start_time":    entry.get("start_time"),           # HH:MM
                "end_time":      entry.get("end_time"),
                "name_event":    normalize_text(entry.get("name_event")),
                "professors":    normalize_text(entry.get("professors")),
                "cancelled":    normalize_text(entry.get("cancelled")),
                "event_type":    normalize_text(entry.get("event_type")),
            })

    print(f"[calendario aule]       loaded {len(rows):>6} rows from {len(json_files)} file(s)")
    return rows


def load_info_aule(path: Path) -> list[dict]:

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", data) if isinstance(data, dict) else data

    rows = []
    for meta in entries:
        # Normalize equipment string values but keep the JSON structure intact
        equipment = meta.get("equipment")
        if isinstance(equipment, dict):
            equipment = {k: normalize_text(v) if isinstance(v, str) else v for k, v in equipment.items()}

        rows.append({
            "room_code":        meta.get("room_code"),         # 001_3 — code
            "room_name":        normalize_text(meta.get("room_name")),
            "site_name":        normalize_text(meta.get("site_name")),
            "site_code":        meta.get("site_code"),         # AH03 — code
            "address":          normalize_text(meta.get("address")),
            "floor":            normalize_text(meta.get("floor")),
            "room_type":        normalize_text(meta.get("room_type")),
            "capacity":         meta.get("capacity"),          # integer — do not normalize
            "accessible":       meta.get("accessible"),        # bool — do not normalize
            "maps_url":         meta.get("maps_url"),          # URL — do not normalize
            "equipment":        json.dumps(equipment, ensure_ascii=False),
            "url":              meta.get("url"),               # URL — do not normalize
        })
    return rows


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------

def insert_data(
    db_path: Path,
    personale_path: Path,
    insegnamento_path: Path,
    info_corsi_di_laurea: Path,
    lezioni_dir: Path,
    calendario_aule_dir: Path,
    info_aule: Path,
) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}. Run 01_create_db.py first."
        )

    con = sqlite3.connect(db_path)

    with con:
        # -- personale --
        personale_rows = load_personale(personale_path)
        con.executemany(
            """INSERT INTO personale
               (nome_and_surname, role, department, department_url, phone, email, last_update)
               VALUES (:nome_and_surname, :role, :department, :department_url,
                       :phone, :email, :last_update)""",
            personale_rows,
        )
        print(f"[personale]    inserted {len(personale_rows):>6} rows")

        # -- insegnamento --
        insegnamento_rows = load_insegnamento(insegnamento_path)
        con.executemany(
            """INSERT INTO insegnamento
               (subject_code, degree_program_name, degree_program_name_eng, degree_program_code,
                subject_name, study_code, academic_year, teams_code, professors,
                main_professor_id, period, last_update)
               VALUES (:subject_code, :degree_program_name, :degree_program_name_eng, :degree_program_code,
                       :subject_name, :study_code, :academic_year, :teams_code, :professors,
                       :main_professor_id, :period, :last_update)""",
            insegnamento_rows,
        )
        print(f"[insegnamento] inserted {len(insegnamento_rows):>6} rows")

      # -- corsi di laurea --
        corso_di_laurea_rows = load_corso_di_laurea(info_corsi_di_laurea)
        con.executemany(
            """INSERT INTO corso_di_laurea
               (name, url, department, type, duration, location, language)
               VALUES (:name, :url, :department,
                       :type, :duration, :location, :language)""",
            corso_di_laurea_rows,
        )
        print(f"[insegnamento] inserted {len(corso_di_laurea_rows):>6} rows")
        
        # -- lezione --
        lezione_rows = load_lezioni(lezioni_dir)
        if lezione_rows:
            con.executemany(
                """INSERT INTO lezione
                (subject_code, degree_program_name, degree_program_code, subject_name,
                    study_year_code, curriculum, date, start_time, end_time, department,
                    room_code, room_name, site_code, site_name, address,
                    professors, cancelled, url)
                VALUES (:subject_code, :degree_program_name, :degree_program_code, :subject_name,
                        :study_year_code, :curriculum, :date, :start_time, :end_time, :department,
                        :room_code, :room_name, :site_code, :site_name, :address,
                        :professors, :cancelled, :url)""",
                lezione_rows,
            )
            print(f"[lezione]       inserted {len(lezione_rows):>6} rows")


        # -- evento_aula --
        calendario_aule_rows = load_calendario_aule(calendario_aule_dir)
        if calendario_aule_rows:
            con.executemany(
                """INSERT INTO evento_aula
                   (site_code, room_code, site_name, room_name, date, last_update, start_time, end_time, name_event,
                    professors, cancelled, event_type)
                   VALUES (:site_code, :room_code, :site_name, :room_name, :date, :last_update, :start_time, :end_time, :name_event,
                           :professors, :cancelled, :event_type)""",
                calendario_aule_rows,
            )
            print(f"[calendario aule]       inserted {len(calendario_aule_rows):>6} rows")

        # -- info_aula --
        info_aula_rows = load_info_aule(info_aule)
        if info_aula_rows:
            con.executemany(
                """INSERT INTO info_aula
                   (room_code, room_name, site_name, site_code, address, floor, room_type, capacity, accessible,
                    maps_url, equipment)
                   VALUES (:room_code, :room_name, :site_name, :site_code, :address, :floor, :room_type, :capacity, :accessible,
                           :maps_url, :equipment)""",
                info_aula_rows,
            )
            print(f"[info aula]       inserted {len(info_aula_rows):>6} rows")

    con.close()
    print(f"\nData inserted into: {db_path.resolve()}")


if __name__ == "__main__":
    insert_data(
        db_path=Path(DEFAULT_DB),
        personale_path=Path(DEFAULT_PERSONALE),
        insegnamento_path=Path(DEFAULT_INSEGNAMENTO),
        info_corsi_di_laurea=Path(DEFAULT_CORSO_DI_LAUREA),
        lezioni_dir=Path(DEFAULT_LEZIONI_DIR),
        calendario_aule_dir=Path(DEFAULT_CALENDARIO_AULE_DIR),
        info_aule=Path(DEFAULT_INFO_AULE),
    )
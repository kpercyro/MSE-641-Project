import ast
import csv
from collections import Counter
from pathlib import Path


def build_filtered_dataset(input_path=None, output_path=None, min_count=500):
    if input_path is None:
        input_path = Path(__file__).resolve().parent / "data.csv"
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "data_500.csv"

    input_path = Path(input_path)
    output_path = Path(output_path)

    genre_counter = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        genre_string = row.get("genre", "")
        if not genre_string:
            continue
        try:
            genres = ast.literal_eval(genre_string)
        except (ValueError, SyntaxError):
            continue
        if isinstance(genres, list):
            genre_counter.update(genres)

    valid_genres = {genre for genre, count in genre_counter.items() if count >= min_count}

    filtered_rows = []
    for row in rows:
        genre_string = row.get("genre", "")
        if not genre_string:
            continue
        try:
            genres = ast.literal_eval(genre_string)
        except (ValueError, SyntaxError):
            continue
        if isinstance(genres, list):
            if genres and all(genre in valid_genres for genre in genres):
                filtered_rows.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Wrote {len(filtered_rows)} rows to {output_path}")
    print("Genres kept:", sorted(valid_genres))


if __name__ == "__main__":
    build_filtered_dataset()

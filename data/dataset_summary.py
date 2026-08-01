import ast
import csv
from collections import Counter
from pathlib import Path


def summarize_genre_counts(data_path=None):
    if data_path is None:
        project_root = Path(__file__).resolve().parent.parent
        data_path = project_root / "data" / "data.csv"

    genre_counter = Counter()

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre_string = row.get("genre", "")
            if not genre_string:
                continue
            try:
                genres = ast.literal_eval(genre_string)
            except (ValueError, SyntaxError):
                continue
            if isinstance(genres, list):
                genre_counter.update(genres)

    print("Genre counts:")
    for genre, count in sorted(genre_counter.items()):
        print(f"{genre}: {count}")

    print(f"\nTotal genre labels: {sum(genre_counter.values())}")
    print(f"Total unique genres: {len(genre_counter)}")


if __name__ == "__main__":
    summarize_genre_counts()

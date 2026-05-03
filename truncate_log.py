import argparse
import csv
import os


def truncate_log(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        print("Empty log file, nothing to do.")
        return

    # Walk through rows and restart whenever step goes backwards,
    # keeping only the latest contiguous run for each step range.
    clean = []
    for row in rows:
        try:
            step = int(row[0])
        except (ValueError, IndexError):
            continue
        if clean and step <= int(clean[-1][0]):
            # Step went backwards — drop everything from the last overlap onwards
            while clean and int(clean[-1][0]) >= step:
                clean.pop()
        clean.append(row)

    backup = path + ".bak"
    os.rename(path, backup)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(clean)

    print(f"Kept {len(clean)} rows (was {len(rows)}). Backup at {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", nargs="+", help="CSV log file(s) to truncate")
    args = parser.parse_args()

    for path in args.log_files:
        print(f"Processing {path}")
        truncate_log(path)

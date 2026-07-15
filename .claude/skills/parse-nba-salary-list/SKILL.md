---
name: parse-nba-salary-list
description: Parse a raw NBA player-salary list (copy/pasted text with repeating rank/name/"TEAM, POS"/"$salary" lines, e.g. from a salary-rankings site) into a single CSV with one row per player. Use when the user provides a .txt dump of ranked player salaries and asks to convert/parse/clean it into a CSV.
---

# Parse NBA salary list

Converts a plain-text salary ranking dump into a tidy CSV (`RANK, PLAYER_NAME, TEAM, POSITION, SALARY`).

## Expected input format

The source file repeats fixed groups of 4 lines per player:

```
<rank, or blank if tied with the previous rank>
<player name>
<TEAM, POSITION>
<$salary>
```

Example:

```
1
Stephen Curry
 GSW, PG
$59,606,817
2
Joel Embiid
 PHI, C
$55,224,526

Nikola Jokic
 DEN, C
$55,224,526
```

(The blank line before "Nikola Jokic" means he is tied for rank 2 with Embiid.)

If the source doesn't match this exact 4-line-per-player shape (e.g. different delimiters, extra header/footer lines, no tie-blank-line convention), inspect the file first and adjust the parsing logic — don't force-fit the bundled script onto a different format.

## Usage

Run the bundled script, pointing it at the raw input and desired output path. Salary CSVs live in `data/salaries/`:

```
python .claude/skills/parse-nba-salary-list/scripts/parse_salaries.py <input.txt> data/salaries/salaries_<season>.csv
```

It strips `$` and commas from the salary, splits `TEAM, POSITION` (positions may be multi-value like `SF/PF`), carries forward the rank across tied/blank-rank rows, and writes one row per player.

After running, spot-check the output (row count, a few known players, `df.duplicated(subset=['PLAYER_NAME'])`) before treating it as final.

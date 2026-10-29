# Participant Summary Table (reproduces the instructor's Excel exercise)

Source: `data/posture_cleaned.csv`. Rule: `Overall = "Bad" if (readings > 20°) / (total readings) > 50% else "Good"` (*Posture Machine Learning.pdf*, Step 7).

| Participant | Total readings | Readings > 20° | Readings > 45° | % > 20° | % > 45° | Overall | Matches doc? |
|---|---|---|---|---|---|---|---|
| 1 | 251 | 128 | 99 | 51.0% | 39.4% | Bad | yes |
| 3 | 314 | 189 | 102 | 60.2% | 32.5% | Bad | yes |
| 4 | 492 | 93 | 0 | 18.9% | 0.0% | Good | yes |
| 5 | 388 | 38 | 1 | 9.8% | 0.3% | Good | yes |
| 6 | 568 | 187 | 9 | 32.9% | 1.6% | Good | yes |
| 7 | 956 | 121 | 0 | 12.7% | 0.0% | Good | yes |

Every row matches the instructor's reference table in *Posture Machine Learning.pdf* (Step 8's "What you should get") exactly, integer-for-integer -- confirming `data/posture_cleaned.csv` is the same cleaned dataset the pathway doc was built from, before any of this project's own windowing or modeling runs on it.

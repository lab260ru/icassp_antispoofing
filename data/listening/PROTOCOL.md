# Listening audit protocol

One page. Read it once, then work through `listening_sheet.csv` top to bottom.

## The task

For each row: play `data/audio_sample/<clip>`, look at `target_word`, and count.

- **If `target_word` is a single word** (e.g. `blue`), the clip is a *repeated*
  item: count how many times you hear that word spoken anywhere in the clip.
  There is no cap -- if the model kept going past what sounds reasonable, count
  the real number, however large.
- **If `target_word` is a list separated by `/`** (e.g. `deep / fast / long /
  wide`), the clip is a *control* item: count how many of the words in that
  list you hear spoken, anywhere in the clip, in any order. If a word appears
  twice in the list, it can count twice, but only if you actually hear it said
  twice.

Type the integer into `human_count`. That is the whole task, repeated ~60
times.

## Partial, slurred, or ambiguous repetitions

Count what you can defensibly hear as a distinct utterance of the word, even if
it's slurred, cut short, or run into the next one -- use your judgement the way
you would transcribing speech, not a stricter standard. If a repetition is
genuinely a coin flip (could be one long slurred word or two short ones), count
it as one and don't agonise; the audit is measuring rough agreement, not
adjudicating boundary cases to the count.

## `unclear`

Put `y` in `unclear` when you cannot get a defensible count at all -- noise,
the model producing something with no recognisable words, or a target word so
garbled you genuinely cannot tell if it's there. You may leave `human_count`
blank on an `unclear` row. You do not need to fill in `unclear` on an ordinary
row; leave it blank.

A row needs **either** a `human_count` **or** `unclear=y` to count as done. A
row with neither is treated as unfinished and the scorer will refuse to run.

## What not to do

Don't open `listening_key.csv`. It has the model, the arm, and the machine's
own count for every clip -- looking at it before you finish defeats the point
of running this at all (that file is not even yours to read; the scorer reads
it, you don't). Don't guess a model or a script from the filename inside
`clip` either; it's not informative but there's no reason to try.

## Time

~60 clips at the default `--n`, about 15 seconds of audio each plus time to
find the file, maybe replay it once, and type a number: **budget about an
hour**. `scripts/make_listening_sheet.py --n 40` makes a shorter sheet if an
hour doesn't fit; rerun with the same `--seed` to reproduce a sheet exactly.

## When you're done

```
python scripts/score_listening.py
```

reads `data/listening/listening_sheet.csv` against
`data/listening/listening_key.csv` and reports agreement, the arm-asymmetry
check, and what the paper's headline gap becomes on these clips with your
counts substituted for the judge's. It refuses to run (and says which rows) if
anything is unfinished.

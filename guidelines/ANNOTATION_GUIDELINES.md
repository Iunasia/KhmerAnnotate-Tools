# KhmerAnnotate — Khmer NER Annotation Guidelines (v1)

Foundation: Universal NER (Mayhew et al., 2024) + Khmer-specific rules for
Khmer news, encyclopaedic, and policy text (VOA Khmer, Khmer Wikipedia, ODC).

Target quality:

- Inter-Annotator Agreement (IAA): `>= 0.80` (span-level F1 & Cohen's Kappa) on
  ~200 double-labeled sentences.
- Post-curation quality gate (`checker/label_checker.py`): `0` errors.

---

## 1. Core Labels (Flat Whole-Word BIO)

Each sentence is pre-segmented into whole words. In INCEpTION you highlight
whole words and choose one of **3 labels**. Exporting to **CoNLL-2002** produces
the **7 flat BIO tags** (`O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`,
`I-ORG`):

| Label | Meaning | What to label | Examples |
|---|---|---|---|
| `PER` | Person | Named real individuals (politicians, activists, citizens, officials, leaders, authors, historical figures) | `វន់ ពៅ`, `រ៉ុង ឈុន`, `កើត សារ៉ាយ`, `Donald Trump`, `Samuel J. Paparo` |
| `LOC` | Location | Countries, cities, provinces, districts (`ខណ្ឌ`/`ស្រុក`), communes, villages, regions, rivers, lakes, geographical areas | `កម្ពុជា`, `ភ្នំពេញ`, `ខេត្ត កោះកុង`, `ខណ្ឌ ឫស្សីកែវ`, `ទន្លេសាប`, `សហរដ្ឋ អាមេរិក`, `ហ្កាហ្សា` |
| `ORG` | Organisation | Ministries, courts, political parties, NGOs, unions, companies, banks, universities, media outlets, armed forces, international bodies | `ក្រសួង មហាផ្ទៃ`, `តុលាការ ខេត្ត កណ្តាល`, `រដ្ឋាភិបាល កម្ពុជា`, `ធនាគារ ជាតិ`, `វីអូអេ`, `IDEA`, `USAID`, `អូតង់` |

- `B-` marks the first word of a name; `I-` marks the remaining words of the
  same name; `O` means not an entity.
- **Flat means one label per word** — never stack two labels on the same word.

---

## 2. Five Mandatory Khmer Edge Cases

### Case 1: Honorifics and titles before a person's name (`លោក វន់ ពៅ`)
- **Rule:** Exclude titles, honorifics, and job roles (`លោក`, `លោកស្រី`,
  `អ្នកស្រី`, `កញ្ញា`, `សម្តេច`, `ឯកឧត្តម`, `បណ្ឌិត`, `នាយករដ្ឋមន្ត្រី`,
  `ប្រធានាធិបតី`, `ឧត្តមនាវី`). Label **only the personal name** as `PER`.
- **Examples:**
  - `លោក [វន់ ពៅ]_PER` -> `លោក` (`O`), `វន់` (`B-PER`), `ពៅ` (`I-PER`)
  - `អ្នកស្រី [សាន ស្រែ]_PER` -> `អ្នកស្រី` (`O`), `សាន` (`B-PER`), `ស្រែ` (`I-PER`)
  - `ព្រះមហាក្សត្រ [នរោត្តម សីហមុនី]_PER` -> label only the personal name;
    `ព្រះមហាក្សត្រ` alone without a personal name is `O`.

### Case 2: Title used alone as a pronoun (`លោក ថ្លែងថា ...`)
- **Rule:** Do **not** label (`O`). When `លោក`, `លោកស្រី`, or `អ្នកស្រី` is
  used alone to mean "he" or "she", it is a pronoun, not a name.

### Case 3: Nested names (`សាលាដំបូង រាជធានី ភ្នំពេញ`)
- **Rule:** Label the **entire outermost organisation span** as `ORG`. Do not
  split out an embedded place name as `LOC`.
- **Examples:**
  - `[សាលាដំបូង រាជធានី ភ្នំពេញ]_ORG`
  - `[តុលាការ ខេត្ត កណ្តាល]_ORG`
  - `[ក្រសួង ការបរទេស អាមេរិក]_ORG`
  - `[រដ្ឋាភិបាល កម្ពុជា]_ORG` (when acting as an institution)
- **Contrast:** `នៅ [កម្ពុជា]_LOC` (standalone country/place -> `LOC`).

### Case 4: English words and acronyms (`IDEA`, `USAID`, `ILO`)
- **Rule:** Apply the exact same `PER`, `LOC`, `ORG` definitions to English
  names and acronyms inside Khmer sentences.
- **Examples:**
  - `ហៅកាត់ ថា [IDEA]_ORG`, `ទីភ្នាក់ងារ [USAID]_ORG`, `លោក [Donald Trump]_PER`
- **Not entities (`O`):** Trade schemes, laws, treaties, and hardware/model
  codes (e.g. `EBA`, `KF-16`, `MK-82`, `CC BY-SA 4.0`).

### Case 5: Place-type prefixes (`ខណ្ឌ ឫស្សីកែវ`, `ខេត្ត កោះកុង`)
- **Rule:**
  - **Include** official administrative prefixes (`រាជធានី`, `ខេត្ត`, `ក្រុង`,
    `ខណ្ឌ`, `ស្រុក`, `ឃុំ`, `សង្កាត់`, `ភូមិ`, `រដ្ឋ`) when directly attached
    to a proper place name: `[ខេត្ត កោះកុង]_LOC`, `[ខណ្ឌ ឫស្សីកែវ]_LOC`,
    `[រាជធានី ភ្នំពេញ]_LOC`.
  - **Exclude** generic framing nouns like `ប្រទេស` ("country") and `តំបន់`
    ("region/area"): `ប្រទេស [កម្ពុជា]_LOC`, `តំបន់ [ហ្កាហ្សា]_LOC`.

---

## 3. Do NOT Label (`O`)

- **Dates and times:** `ថ្ងៃ ៧ មករា`, `ឆ្នាំ ២០២៥`, `ខែ ធ្នូ`
- **Numbers, money, percentages:** `១៣ នាក់`, `៣៥០ លាន ដុល្លារ`, `៩០%`
- **Nationalities / ethnicities / languages (as modifiers):** `ជនជាតិ ហ្វីលីពីន`,
  `ពលរដ្ឋ ខ្មែរ`, `ភាសា ខ្មែរ`
- **Laws, articles, treaties, programmes:** `មាត្រា ៦៣`, `EBA`, `កិច្ចព្រមព្រៀង`
- **Generic roles or groups:** `ពលរដ្ឋ`, `កម្មករ`, `មន្ត្រី`, `សកម្មជន`
- **Punctuation and quotation marks:** Never include `«`, `»`, `។`, `៕`,
  commas, colons, or brackets inside a label span.

---

## 4. Annotation & QA Workflow

1. **Whole-word snapping only:** Never highlight part of a word or split a
   Khmer cluster.
2. **Consistency:** Tag repeated mentions of the same entity consistently
   across the document.
3. **Pilot round (`pilot_60.txt`):** All annotators label the same 60 sentences,
   run `checker/agreement.py`, inspect the HTML disagreement report in a
   browser, and refine rules before starting the main pool.
4. **Quality gate:** Before training `baseline/train_xlmr.py`, every curated
   CoNLL-2002 file must pass `checker/label_checker.py` with `0` errors and be
   split by document ID using `checker/prepare_splits.py`.

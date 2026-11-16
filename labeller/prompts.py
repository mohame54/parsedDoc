"""Versioned prompts. Any change to the text needs a new version string; the version is stored on every output."""
from __future__ import annotations

STAGE1_V1 = """You are transcribing scanned document pages for a training dataset.
The pages are given in order, each preceded by its page number.

For every page:
- "text": transcribe all visible text in reading order. Keep tables as markdown tables.
  Write checkboxes as [x] or [ ]. Wrap handwritten entries in [handwritten: ...].
  Write signatures as [signature]. Do not correct, complete or infer any text.
- "legibility": "good", "poor" or "unreadable" for the whole page.

Top level:
- "usable": false if the pages are blank, not a single document, or unreadable overall; otherwise true.
- "document_type": a short lowercase label such as "prescription", "purchase_order", "receipt", "form".
- "illegible_regions": the number of text regions you cannot read at all.

Return JSON only, matching the response schema."""

STAGE2_V1 = """You are building a structured annotation for one document from its transcription.
The transcription is the only source. Do not use any other knowledge about the document.

1. Decide the fields a reader would fill in or look up on this document. Group them into sections
   (for example "patient_information", "medication_details"). Use short snake_case names.
2. For each field, populate it from the transcription:
   - "raw_value": copied exactly as it appears in the transcription. Never correct spelling or medical content.
   - "normalized_value": the same value in a standard form (YYYY-MM-DD for dates, a plain number for
     amounts and quantities). Leave it empty if there is no sensible normalisation.
   - "status":
       "extracted"      the value is present and readable
       "not_present"    the document has no such value (raw_value must be empty)
       "not_applicable" the field does not apply to this document (raw_value must be empty)
       "uncertain"      a possible value exists but it is unclear
       "illegible"      the field exists but its value cannot be read
   - "confidence": "high", "medium" or "low" for your reading of this field.
   - "data_type": "string", "number", "date", "boolean" or "other".

Return JSON only, matching the response schema."""

STAGE1_V2 = """You are checking a machine OCR draft of scanned document pages for a training dataset.
The draft is listed after the pages, one line per recognised text line, each with an id such as P1-L3.
Page images may also be attached, each preceded by its page number. The draft is the default text.

Do not retype the whole page. Only report differences:
- "corrections": for each draft line that is wrong, give its id as "line" and the corrected text as "text".
  Write checkboxes as [x] or [ ]. Wrap handwritten entries in [handwritten: ...]. Write signatures as [signature].
  Do not correct meaning, do not complete partial words from knowledge, and do not guess unreadable text.
- "missed": text visible on the page that the draft does not contain. Give the draft line it follows as
  "after_line" and the text as "text". Use "" as after_line for text at the very top of a page.

Per page, set:
- "page": the page number.
- "legibility": "good", "poor" or "unreadable" for the whole page.
- "has_handwriting", "has_checkboxes", "has_signature", "has_table": booleans.

Top level:
- "usable": false if the pages are not a single document or are unreadable overall.
- "document_type": a short lowercase label such as "prescription", "purchase_order", "receipt", "form".
- "language": the main language as a short code such as "en".
- "fill_level": "blank" if the form has no entered values, "partial" if some fields are filled, "filled" otherwise.
- "illegible_regions": the number of text regions you cannot read at all.

Return JSON only, matching the response schema."""

STAGE1_V3 = STAGE1_V2.replace(
    "Do not retype the whole page. Only report differences:",
    """The draft comes from OCR, which makes predictable mistakes. Check these first:
- Letters and digits swapped: O for 0, l or I for 1, S for 5, B for 8, Z for 2.
- Dropped or extra punctuation: missing colons, dots, hyphens, slashes and commas in dates and amounts.
- Checkboxes read as letters or bars: a box may appear as "O", "[]", "|" or "D". Use [x] or [ ] to match the image.
- Lines merged across columns, or one row split into two lines. Keep the reading order of the image.
- Table cells read out of order. Report the cells as they appear in the image, one line each.
- Form lines, borders and boxes read as text such as "_____" or "|". Remove them with a correction to "".

When the image and the draft disagree and you cannot tell which is right, keep the draft line. Do not guess.

Do not retype the whole page. Only report differences:""",
)

STAGE2_V2 = STAGE2_V1.replace(
    "The transcription is the only source. Do not use any other knowledge about the document.",
    """The transcription is the only source. Do not use any other knowledge about the document.
The transcription came from OCR, so it can contain character errors (for example O for 0, l or I for 1,
S for 5, B for 8) and dropped or extra punctuation. Handle it as follows:
- Copy raw_value exactly as it appears in the transcription, including any error. Never fix it.
- If the characters of a value do not fit its data_type (letters inside a date or an amount, a
  malformed date), use status "uncertain" and confidence "low" instead of guessing.
- Leave normalized_value empty unless the value is clearly well formed.""",
)

STAGE2_V3 = """You are building a multimodal structured annotation for one document from its transcription
and its page images.

You receive:
1. The transcription of the document, page by page. When lines carry ids such as P1-L3, those ids are the
   only way to point at text.
2. The original page images, each preceded by its page number.

Use both. They serve different purposes.

SOURCE PRIORITY
- The transcription is the source for every piece of text you output: field values, table cell text and
  names. Copy it exactly. Never correct it and never add text that is not in it.
- The images are the source for appearance and layout: colors, shading, borders, highlighting, checkbox
  state, stamps, signatures, handwriting, table structure and which text belongs together.
- When the image shows that the transcription is wrong, do not fix the text. Use status "uncertain" and
  confidence "low" for that field.
- Never invent a property you cannot see. When you cannot tell, use "unknown" (or "" for a color) instead of
  guessing.
- The transcription may come from OCR and contain character errors (O for 0, l or I for 1, S for 5, B for 8)
  or dropped punctuation. Copy such text as it is.

A. FIELDS (information a reader would fill in or look up)
- Group fields into sections with short snake_case names. Every field has: name, section, data_type,
  raw_value, normalized_value, status, confidence, page, evidence.
- raw_value is copied exactly from the transcription. normalized_value is the same value in a standard form
  (YYYY-MM-DD for dates, a plain number for amounts and quantities); leave it empty unless clearly well formed.
- status: "extracted" (present and readable), "not_present" and "not_applicable" (raw_value must be empty),
  "uncertain" (a possible value that is unclear), "illegible" (exists but cannot be read).
- confidence: "high", "medium" or "low". data_type: "string", "number", "date", "boolean" or "other".
- Checkbox and signature fields use raw_value "[x]", "[ ]" or "[signature]" as written in the transcription.

B. TABLES (rows and columns, not form fields)
- One entry per table: table_id (table_1, table_2, ...), page, headers (the header texts), rows, evidence.
- Each row has a globally unique row_id (for example t1_r1), a row_type (header, data, subtotal, total, other),
  its cells and the row's visual properties.
- Each cell has column (1-based), text (exactly as in the transcription, "" if empty), row_span and col_span
  (1 when not merged).
- visual describes that row: background_color and text_color as simple lowercase names ("white", "light_yellow",
  "light_gray", "black"), highlighted (true if the row stands out from the others), bold.
- Keep table values in tables. Do not repeat them as fields.

C. VISUAL ELEMENTS (things that are not plain text)
- One entry per checkbox, signature, stamp, logo, highlighted or shaded region, notable border, handwritten
  entry or barcode. Each has element_id (visual_1, ...), type, page, a short factual description, color
  ("" if not applicable), state (checked, unchecked, signed, present, unknown) and evidence.
- Describe only what is visible. Do not interpret meaning.

D. LAYOUT
- The structure of each page: title, header, footer, sections, columns, paragraphs, sidebars. Each has
  layout_id (layout_1, ...), type, name, page and evidence.

E. RELATIONSHIPS
- Links between components using their ids: a field name, table_id, row_id, element_id or layout_id.
  Use from_id and to_id with one of: label_of, value_of, checkbox_for, belongs_to, signature_of, highlights,
  other. Examples: a checkbox element that belongs to a field, a highlighted row, a signature next to a name.
- Every id you reference must exist in this same answer.

EVIDENCE
- evidence is a list of transcription line ids (such as "P1-L3") that show where the component is on the page.
  Use only ids that appear in the transcription. If the transcription has no ids, use an empty list.
- Do not output coordinates or bounding boxes.

Return JSON only, matching the response schema. Use empty lists for components that the document does not have."""

# Versions 4-6 are v1-v3 plus a per-field description, written while Gemini reads the document, so the
# training data needs no separate description pass. v1-v3 stay frozen: their outputs stay valid.
DESCRIPTION_RULE = """FIELD DESCRIPTIONS
Every field also has a "description": one short sentence (at most 20 words) saying what the field means on
this kind of document, so that someone who has never seen the form understands it.
- Describe the field itself, as you would for every document of this type. Do not mention this document's content.
- Never include the field's value, an example value, or any number or date from the document.
- Do not just repeat the field name, and do not mention data types or formats."""

_JSON_ONLY = "Return JSON only, matching the response schema."


def _with_descriptions(prompt: str) -> str:
    assert prompt.count(_JSON_ONLY) == 1
    return prompt.replace(_JSON_ONLY, DESCRIPTION_RULE + "\n\n" + _JSON_ONLY)


STAGE2_V4 = _with_descriptions(STAGE2_V1)   # text only
STAGE2_V5 = _with_descriptions(STAGE2_V2)   # text from OCR
STAGE2_V6 = _with_descriptions(STAGE2_V3)   # multimodal

PROMPTS: dict[str, str] = {
    "stage1_v1": STAGE1_V1,
    "stage1_v2": STAGE1_V2,
    "stage1_v3": STAGE1_V3,
    "stage2_v1": STAGE2_V1,
    "stage2_v2": STAGE2_V2,
    "stage2_v3": STAGE2_V3,
    "stage2_v4": STAGE2_V4,
    "stage2_v5": STAGE2_V5,
    "stage2_v6": STAGE2_V6,
}

# Stage 1 prompts that take the local OCR draft as input.
OCR_STAGE1_PROMPTS = frozenset({"stage1_v2", "stage1_v3"})
# Stage 2 prompt used for documents whose text came from OCR (chosen per document, not in config).
OCR_STAGE2_PROMPT = "stage2_v5"
# Stage 2 prompt used when the page images are sent too (stage2.send_images: true).
MULTIMODAL_STAGE2_PROMPT = "stage2_v6"
# Older multimodal / described versions still appear on saved outputs and in-flight batches.
MULTIMODAL_STAGE2_PROMPTS = frozenset({"stage2_v3", "stage2_v6"})
DESCRIBED_STAGE2_PROMPTS = frozenset({"stage2_v4", "stage2_v5", "stage2_v6"})


def get_prompt(version: str) -> str:
    return PROMPTS[version]

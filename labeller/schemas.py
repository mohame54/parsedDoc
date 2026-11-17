"""Response schemas sent to Gemini and used to validate its output.

Kept flat and without $ref or unsupported keywords: Gemini silently strips keywords it does not
support, so every real constraint is enforced again in labeller/validate.py.
"""
from __future__ import annotations

import copy

FIELD_STATUSES = ["extracted", "not_present", "not_applicable", "uncertain", "illegible"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]
DATA_TYPES = ["string", "number", "date", "boolean", "other"]

STAGE1_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "document_type": {"type": "string"},
        "illegible_regions": {"type": "integer"},
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "legibility": {"type": "string", "enum": ["good", "poor", "unreadable"]},
                    "text": {"type": "string"},
                },
                "required": ["page", "legibility", "text"],
            },
        },
    },
    "required": ["usable", "document_type", "illegible_regions", "pages"],
}

FILL_LEVELS = ["blank", "partial", "filled"]

STAGE1_V2_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "document_type": {"type": "string"},
        "language": {"type": "string"},
        "fill_level": {"type": "string", "enum": FILL_LEVELS},
        "illegible_regions": {"type": "integer"},
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "legibility": {"type": "string", "enum": ["good", "poor", "unreadable"]},
                    "has_handwriting": {"type": "boolean"},
                    "has_checkboxes": {"type": "boolean"},
                    "has_signature": {"type": "boolean"},
                    "has_table": {"type": "boolean"},
                },
                "required": ["page", "legibility", "has_handwriting", "has_checkboxes", "has_signature", "has_table"],
            },
        },
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"line": {"type": "string"}, "text": {"type": "string"}},
                "required": ["line", "text"],
            },
        },
        "missed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"after_line": {"type": "string"}, "text": {"type": "string"}},
                "required": ["after_line", "text"],
            },
        },
    },
    "required": ["usable", "document_type", "language", "fill_level", "illegible_regions",
                 "pages", "corrections", "missed"],
}

STAGE2_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "section": {"type": "string"},
                    "data_type": {"type": "string", "enum": DATA_TYPES},
                    "raw_value": {"type": "string"},
                    "normalized_value": {"type": "string"},
                    "status": {"type": "string", "enum": FIELD_STATUSES},
                    "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
                },
                "required": ["name", "section", "data_type", "raw_value", "normalized_value", "status", "confidence"],
            },
        },
    },
    "required": ["document_type", "fields"],
}


# --- Stage 2 v3: multimodal annotation -------------------------------------------------------------
# Business information (fields), table structure, visual appearance, page layout and the links between
# them are separate components, so each can be used on its own to train a different kind of model.
# Spatial grounding comes from the OCR's own coordinates: components cite OCR line ids in "evidence" and
# the labeller resolves those ids to boxes locally. Gemini never produces coordinates.

ROW_TYPES = ["header", "data", "subtotal", "total", "other"]
ELEMENT_TYPES = ["checkbox", "signature", "stamp", "logo", "highlight", "border", "shading", "handwriting",
                 "barcode", "other"]
ELEMENT_STATES = ["checked", "unchecked", "signed", "present", "unknown"]
LAYOUT_TYPES = ["title", "header", "footer", "section", "column", "paragraph", "sidebar", "other"]
RELATIONSHIP_TYPES = ["label_of", "value_of", "checkbox_for", "belongs_to", "signature_of", "highlights", "other"]

_EVIDENCE = {"type": "array", "items": {"type": "string"}}

STAGE2_V3_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "section": {"type": "string"},
                    "data_type": {"type": "string", "enum": DATA_TYPES},
                    "raw_value": {"type": "string"},
                    "normalized_value": {"type": "string"},
                    "status": {"type": "string", "enum": FIELD_STATUSES},
                    "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
                    "page": {"type": "integer"},
                    "evidence": _EVIDENCE,
                },
                "required": ["name", "section", "data_type", "raw_value", "normalized_value", "status",
                             "confidence", "page", "evidence"],
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "page": {"type": "integer"},
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_id": {"type": "string"},
                                "row_type": {"type": "string", "enum": ROW_TYPES},
                                "cells": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "column": {"type": "integer"},
                                            "text": {"type": "string"},
                                            "row_span": {"type": "integer"},
                                            "col_span": {"type": "integer"},
                                        },
                                        "required": ["column", "text", "row_span", "col_span"],
                                    },
                                },
                                "visual": {
                                    "type": "object",
                                    "properties": {
                                        "background_color": {"type": "string"},
                                        "text_color": {"type": "string"},
                                        "highlighted": {"type": "boolean"},
                                        "bold": {"type": "boolean"},
                                    },
                                    "required": ["background_color", "text_color", "highlighted", "bold"],
                                },
                            },
                            "required": ["row_id", "row_type", "cells", "visual"],
                        },
                    },
                    "evidence": _EVIDENCE,
                },
                "required": ["table_id", "page", "headers", "rows", "evidence"],
            },
        },
        "visual_elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "type": {"type": "string", "enum": ELEMENT_TYPES},
                    "page": {"type": "integer"},
                    "description": {"type": "string"},
                    "color": {"type": "string"},
                    "state": {"type": "string", "enum": ELEMENT_STATES},
                    "evidence": _EVIDENCE,
                },
                "required": ["element_id", "type", "page", "description", "color", "state", "evidence"],
            },
        },
        "layout": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "layout_id": {"type": "string"},
                    "type": {"type": "string", "enum": LAYOUT_TYPES},
                    "name": {"type": "string"},
                    "page": {"type": "integer"},
                    "evidence": _EVIDENCE,
                },
                "required": ["layout_id", "type", "name", "page", "evidence"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_id": {"type": "string"},
                    "to_id": {"type": "string"},
                    "type": {"type": "string", "enum": RELATIONSHIP_TYPES},
                },
                "required": ["from_id", "to_id", "type"],
            },
        },
    },
    "required": ["document_type", "fields", "tables", "visual_elements", "layout", "relationships"],
}


def _with_description(schema: dict) -> dict:
    """The Stage 2 schema with a required one-sentence `description` on every field."""
    out = copy.deepcopy(schema)
    item = out["properties"]["fields"]["items"]
    item["properties"]["description"] = {"type": "string"}
    item["required"].append("description")
    return out


STAGE2_DESCRIBED_RESPONSE_SCHEMA = _with_description(STAGE2_RESPONSE_SCHEMA)
STAGE2_V3_DESCRIBED_RESPONSE_SCHEMA = _with_description(STAGE2_V3_RESPONSE_SCHEMA)


def stage2_schema(multimodal: bool, described: bool) -> dict:
    if multimodal:
        return STAGE2_V3_DESCRIBED_RESPONSE_SCHEMA if described else STAGE2_V3_RESPONSE_SCHEMA
    return STAGE2_DESCRIBED_RESPONSE_SCHEMA if described else STAGE2_RESPONSE_SCHEMA


# --- Stage 2 v7: figures (charts, diagrams, photos, formulas) and strikethrough / underline --------------
# v1-v6 keep their own enums so saved outputs keep validating. v7 only adds to them.
CHART_TYPES = ["bar_chart", "line_chart", "area_chart", "scatter_plot", "pie_chart"]
TREND_CHART_TYPES = ["bar_chart", "line_chart", "area_chart", "scatter_plot"]   # a pie's values are shares
FIGURE_TYPES = CHART_TYPES + ["table_image", "diagram", "flowchart", "map", "photo", "formula", "other"]
TRENDS = ["increasing", "decreasing", "stable", "mixed", "not_applicable"]
ELEMENT_TYPES_V7 = ELEMENT_TYPES + ["strikethrough", "underline"]


def _figure_schema() -> dict:
    point = {"type": "object", "properties": {"label": {"type": "string"}, "value": {"type": "number"},
                                               "estimated": {"type": "boolean"}},
             "required": ["label", "value", "estimated"]}
    series = {"type": "object", "properties": {"name": {"type": "string"},
                                                "points": {"type": "array", "items": point}},
              "required": ["name", "points"]}
    axis = {"type": "object", "properties": {"label": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["label", "unit"]}
    keys = ["figure_id", "figure_type", "page", "title", "caption", "summary", "x_axis", "y_axis", "series",
            "trend", "evidence"]
    return {"type": "object", "properties": {
        "figure_id": {"type": "string"},
        "figure_type": {"type": "string", "enum": FIGURE_TYPES},
        "page": {"type": "integer"},
        "title": {"type": "string"},
        "caption": {"type": "string"},
        "summary": {"type": "string"},
        "x_axis": axis,
        "y_axis": axis,
        "series": {"type": "array", "items": series},
        "trend": {"type": "string", "enum": TRENDS},
        "evidence": _EVIDENCE,
    }, "required": keys}


def _build_v7_schema() -> dict:
    out = _with_description(STAGE2_V3_RESPONSE_SCHEMA)
    out["properties"]["figures"] = {"type": "array", "items": _figure_schema()}
    out["required"] = out["required"] + ["figures"]
    out["properties"]["visual_elements"]["items"]["properties"]["type"] = {"type": "string",
                                                                          "enum": ELEMENT_TYPES_V7}
    return out


STAGE2_V7_RESPONSE_SCHEMA = _build_v7_schema()


def stage2_schema(multimodal: bool, described: bool, figures: bool = False) -> dict:
    if figures:
        return STAGE2_V7_RESPONSE_SCHEMA
    if multimodal:
        return STAGE2_V3_DESCRIBED_RESPONSE_SCHEMA if described else STAGE2_V3_RESPONSE_SCHEMA
    return STAGE2_DESCRIBED_RESPONSE_SCHEMA if described else STAGE2_RESPONSE_SCHEMA

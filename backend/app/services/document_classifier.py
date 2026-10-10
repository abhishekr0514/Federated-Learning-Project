"""Medical Document Classification Service.

SCIENTIFIC CONTRACT:
1. Standardized Categories: Classifies documents into authorized clinical categories.
2. Honest Confidence Reporting: If algorithmic/probabilistic confidence is not measured,
   the confidence field MUST report 'NOT_MEASURED'.
3. Zero Clinical Hallucination: Classification never manufactures or assumes diagnoses.
   If text is insufficient or absent, returns 'UNKNOWN'.
"""

import re
from typing import Any, Dict, List, Optional

DOCUMENT_CATEGORIES = [
    "BLOOD_REPORT",
    "ECG_REPORT",
    "ECHOCARDIOGRAPHY",
    "ANGIOGRAPHY",
    "CT_CORONARY_ANGIOGRAPHY",
    "CARDIAC_MRI",
    "STRESS_TEST",
    "HOLTER",
    "BLOOD_PRESSURE",
    "LIPID_PROFILE",
    "CARDIAC_ENZYME",
    "CARDIOLOGY_NOTE",
    "DISCHARGE_SUMMARY",
    "PRESCRIPTION",
    "OTHER",
    "UNKNOWN",
]

# Clinical keyword mappings for rule-based matching
CATEGORY_RULES = {
    "LIPID_PROFILE": [
        r"\blipid\b",
        r"\bcholesterol\b",
        r"\btriglycerides?\b",
        r"\bldl\b",
        r"\bhdl\b",
        r"\bvldl\b",
    ],
    "CARDIAC_ENZYME": [
        r"\btroponin\b",
        r"\bck-mb\b",
        r"\bmyoglobin\b",
        r"\bcreatine kinase\b",
        r"\bcardiac enzymes?\b",
    ],
    "ECG_REPORT": [
        r"\belectrocardiogram\b",
        r"\becg\b",
        r"\bekg\b",
        r"\b12-lead\b",
        r"\bsinus rhythm\b",
        r"\bst elevation\b",
        r"\bst depression\b",
        r"\bqt interval\b",
        r"\bqrs\b",
    ],
    "ECHOCARDIOGRAPHY": [
        r"\bechocardiogra(m|phy)\b",
        r"\becho\b",
        r"\bejection fraction\b",
        r"\blvef\b",
        r"\bmitral regurgitation\b",
        r"\baortic valve\b",
        r"\bwall motion\b",
    ],
    "ANGIOGRAPHY": [
        r"\bangiography\b",
        r"\bcoronary angiogra(m|phy)\b",
        r"\bcardiac catheterization\b",
        r"\bstent\b",
        r"\bangioplasty\b",
        r"\bladd stenosis\b",
    ],
    "CT_CORONARY_ANGIOGRAPHY": [
        r"\bct coronary\b",
        r"\bcoronary cta\b",
        r"\bcalcium score\b",
        r"\bagatston\b",
    ],
    "CARDIAC_MRI": [
        r"\bcardiac mri\b",
        r"\bcmr\b",
        r"\bdelayed enhancement\b",
        r"\bmyocardial viability\b",
    ],
    "STRESS_TEST": [
        r"\btmt\b",
        r"\btreadmill test\b",
        r"\bstress test\b",
        r"\bbruce protocol\b",
        r"\bexercise stress\b",
    ],
    "HOLTER": [
        r"\bholter\b",
        r"\b24-hour ecg\b",
        r"\bambulatory ecg\b",
        r"\bearrhythmia monitor\b",
    ],
    "BLOOD_PRESSURE": [
        r"\bblood pressure\b",
        r"\bsphygmomanometer\b",
        r"\bhypertension\b",
        r"\bsystolic\b",
        r"\bdiastolic\b",
        r"\bambulatory bp\b",
    ],
    "BLOOD_REPORT": [
        r"\bcomplete blood count\b",
        r"\bcbc\b",
        r"\bhemoglobin\b",
        r"\bplatelets\b",
        r"\bwbc\b",
        r"\brbc\b",
        r"\bserum creatinine\b",
        r"\belectrolytes\b",
    ],
    "DISCHARGE_SUMMARY": [
        r"\bdischarge summary\b",
        r"\bdate of discharge\b",
        r"\bhospital course\b",
        r"\bcondition at discharge\b",
    ],
    "PRESCRIPTION": [
        r"\bprescription\b",
        r"\brx\b",
        r"\bdosage\b",
        r"\btablet\b",
        r"\bonce daily\b",
        r"\bb\.i\.d\b",
    ],
    "CARDIOLOGY_NOTE": [
        r"\bcardiology consultation\b",
        r"\bclinical note\b",
        r"\bchief complaint\b",
        r"\bassessment and plan\b",
        r"\bcardiovascular exam\b",
    ],
}


class DocumentClassifier:
    """Classifies medical document text into standard clinical categories."""

    @classmethod
    def classify(
        cls,
        text: Optional[str] = None,
        filename_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify document text using deterministic keyword rule evaluation.
        
        Returns:
            Dict containing:
                - category: str (from DOCUMENT_CATEGORIES)
                - confidence: 'NOT_MEASURED'
                - matched_keywords: List[str]
        """
        combined_text = f"{(filename_hint or '').lower()} {(text or '').lower()}"

        if not combined_text.strip():
            return {
                "category": "UNKNOWN",
                "confidence": "NOT_MEASURED",
                "matched_keywords": [],
            }

        scores: Dict[str, List[str]] = {}
        for category, patterns in CATEGORY_RULES.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, combined_text)
                if found:
                    matches.extend(found if isinstance(found[0], str) else [f[0] for f in found])
            if matches:
                scores[category] = list(set(matches))

        if not scores:
            return {
                "category": "OTHER",
                "confidence": "NOT_MEASURED",
                "matched_keywords": [],
            }

        # Select category with the highest distinct matched keywords
        best_category = max(scores.keys(), key=lambda k: len(scores[k]))
        return {
            "category": best_category,
            "confidence": "NOT_MEASURED",
            "matched_keywords": scores[best_category],
        }

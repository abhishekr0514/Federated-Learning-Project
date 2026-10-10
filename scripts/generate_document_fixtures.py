"""Synthetic Medical Document Fixtures Generator.

Generates 10 controlled synthetic test fixtures for verifying the medical document
ingestion and privacy masking pipeline:
1. clean_digital_patient_doc.pdf (Clean digital PDF)
2. scanned_patient_doc.pdf (Raster image embedded in PDF)
3. cardiac_report.jpg (JPEG image)
4. lab_results.png (PNG image)
5. noisy_scan.pdf (Low-resolution / noisy image PDF)
6. multipage_consult.pdf (Multiple page digital PDF)
7. rotated_doc.pdf (Rotated page PDF)
8. multi_pii_doc.pdf (Document with multiple direct PII entities)
9. no_pii_protocol.pdf (Document with medical guidelines and zero PII)
10. malformed.pdf (Corrupt byte stream with PDF extension)

ALL DATA IS 100% SYNTHETIC AND NON-REAL. ZERO ACTUAL PATIENT DATA.
"""

import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_fixtures(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean digital patient doc PDF
    p1 = output_dir / "clean_digital_patient_doc.pdf"
    c1 = canvas.Canvas(str(p1), pagesize=letter)
    c1.drawString(72, 750, "HOSPITAL CARDIOLOGY SERVICE - CLINICAL NOTE")
    c1.drawString(72, 720, "Patient Name: John Doe")
    c1.drawString(72, 700, "MRN: 12345-AB")
    c1.drawString(72, 680, "Phone: 555-123-4567")
    c1.drawString(72, 650, "CLINICAL EVALUATION:")
    c1.drawString(72, 630, "Patient presents for routine cardiovascular follow-up.")
    c1.drawString(72, 610, "Blood pressure 128/82 mmHg, heart rate 72 bpm.")
    c1.drawString(72, 590, "Lipid panel results: Total Cholesterol 220 mg/dL, LDL 160 mg/dL, HDL 45 mg/dL.")
    c1.drawString(72, 570, "Plan: Continue dietary modification and repeat lipid panel in 6 months.")
    c1.save()

    # 2. Scanned patient doc PDF (contains raster image of text)
    p2 = output_dir / "scanned_patient_doc.pdf"
    img2 = Image.new("RGB", (600, 800), color=(245, 245, 240))
    d2 = ImageDraw.Draw(img2)
    d2.text((50, 50), "SCANNED REPORT - CARDIOLOGY CLINIC", fill=(20, 20, 20))
    d2.text((50, 100), "Patient Name: Alex Carter", fill=(30, 30, 30))
    d2.text((50, 140), "MRN: SC-98214", fill=(30, 30, 30))
    d2.text((50, 180), "Echocardiogram: LVEF 55%, normal valve function.", fill=(30, 30, 30))
    img2_path = output_dir / "_temp_scan.png"
    img2.save(img2_path)

    c2 = canvas.Canvas(str(p2), pagesize=letter)
    c2.drawImage(str(img2_path), 50, 100, width=500, height=650)
    c2.save()
    if img2_path.exists():
        img2_path.unlink()

    # 3. Cardiac report JPG
    p3 = output_dir / "cardiac_report.jpg"
    img3 = Image.new("RGB", (800, 600), color=(255, 255, 255))
    d3 = ImageDraw.Draw(img3)
    d3.text((40, 40), "CARDIOLOGY DIAGNOSTIC REPORT", fill=(0, 0, 0))
    d3.text((40, 80), "12-Lead Electrocardiogram Interpretation", fill=(0, 0, 0))
    d3.text((40, 120), "Patient Name: Robert Miller", fill=(0, 0, 0))
    d3.text((40, 160), "Sinus rhythm with PR interval 160ms, QRS duration 90ms.", fill=(0, 0, 0))
    img3.save(p3, format="JPEG")

    # 4. Lab results PNG
    p4 = output_dir / "lab_results.png"
    img4 = Image.new("RGB", (800, 600), color=(250, 250, 250))
    d4 = ImageDraw.Draw(img4)
    d4.text((40, 40), "LABORATORY REPORT - CLINICAL BIOCHEMISTRY", fill=(10, 10, 10))
    d4.text((40, 80), "Troponin I: 0.02 ng/mL (Normal < 0.04 ng/mL)", fill=(10, 10, 10))
    d4.text((40, 120), "CK-MB: 2.1 ng/mL", fill=(10, 10, 10))
    d4.text((40, 160), "Patient Name: Emma Watson", fill=(10, 10, 10))
    img4.save(p4, format="PNG")

    # 5. Noisy scan PDF
    p5 = output_dir / "noisy_scan.pdf"
    img5 = Image.new("RGB", (600, 800), color=(235, 235, 230))
    d5 = ImageDraw.Draw(img5)
    # add synthetic noise lines
    for y in range(50, 750, 30):
        d5.line([(30, y), (570, y)], fill=(225, 225, 220), width=1)
    d5.text((60, 80), "HISTORICAL CARDIAC ASSESSMENT (ARCHIVED SCAN)", fill=(50, 50, 50))
    d5.text((60, 120), "Patient Name: Michael Brown", fill=(50, 50, 50))
    d5.text((60, 160), "Blood pressure: 135/85 mmHg", fill=(50, 50, 50))
    img5_path = output_dir / "_temp_noisy.png"
    img5.save(img5_path)

    c5 = canvas.Canvas(str(p5), pagesize=letter)
    c5.drawImage(str(img5_path), 50, 100, width=500, height=650)
    c5.save()
    if img5_path.exists():
        img5_path.unlink()

    # 6. Multi-page consult PDF
    p6 = output_dir / "multipage_consult.pdf"
    c6 = canvas.Canvas(str(p6), pagesize=letter)
    # Page 1
    c6.drawString(72, 750, "CARDIOLOGY CONSULTATION REPORT - PAGE 1 OF 2")
    c6.drawString(72, 710, "Patient Name: David Clark")
    c6.drawString(72, 690, "MRN: DC-77441")
    c6.drawString(72, 650, "Chief Complaint: Exertional chest tightness.")
    c6.drawString(72, 620, "Physical Exam: S1/S2 present, no murmurs. Lungs clear bilaterally.")
    c6.showPage()
    # Page 2
    c6.drawString(72, 750, "CARDIOLOGY CONSULTATION REPORT - PAGE 2 OF 2")
    c6.drawString(72, 710, "Diagnostic Workup & Recommendations:")
    c6.drawString(72, 680, "1. 12-lead ECG: Normal sinus rhythm at 68 bpm.")
    c6.drawString(72, 660, "2. Serum Troponin: 0.01 ng/mL within normal reference limits.")
    c6.drawString(72, 640, "3. Echocardiogram ordered for outpatient evaluation.")
    c6.save()

    # 7. Rotated doc PDF
    p7 = output_dir / "rotated_doc.pdf"
    c7 = canvas.Canvas(str(p7), pagesize=letter)
    c7.rotate(90)
    c7.drawString(200, -300, "ROTATED AMBULATORY ECG SUMMARY")
    c7.drawString(200, -330, "Patient Name: Sarah Connor")
    c7.drawString(200, -360, "MRN: SC-10101")
    c7.drawString(200, -390, "Mean heart rate: 74 bpm. Minimum HR: 52 bpm. Maximum HR: 118 bpm.")
    c7.save()

    # 8. Document with multiple PII entities
    p8 = output_dir / "multi_pii_doc.pdf"
    c8 = canvas.Canvas(str(p8), pagesize=letter)
    c8.drawString(72, 750, "COMPREHENSIVE CARDIOLOGY ADMISSION SUMMARY")
    c8.drawString(72, 720, "Patient Name: Jane Smith")
    c8.drawString(72, 700, "MRN: 98765-CD")
    c8.drawString(72, 680, "Phone: 987-654-3210")
    c8.drawString(72, 660, "Email: jane.smith.patient@testhealth.org")
    c8.drawString(72, 640, "Address: 123 Health Boulevard, Cityville, ST 12345")
    c8.drawString(72, 620, "Government ID: 999-12-3456")
    c8.drawString(72, 600, "Registration No: REG-44321")
    c8.drawString(72, 580, "Date of Birth: 05/14/1980")
    c8.drawString(72, 540, "LABORATORY & DIAGNOSTIC VALUES:")
    c8.drawString(72, 520, "Troponin I 0.04 ng/mL, Ejection Fraction 55%, LDL 160 mg/dL.")
    c8.drawString(72, 500, "Sinus rhythm with ST elevation. Blood pressure 140/90 mmHg.")
    c8.save()

    # 9. Document without PII
    p9 = output_dir / "no_pii_protocol.pdf"
    c9 = canvas.Canvas(str(p9), pagesize=letter)
    c9.drawString(72, 750, "CLINICAL PRACTICE PROTOCOL - ACUTE CHEST PAIN")
    c9.drawString(72, 710, "Guideline Ref: ACC/AHA 2026 Clinical Pathway")
    c9.drawString(72, 670, "Standardized Emergency Department Triage:")
    c9.drawString(72, 640, "1. 12-lead electrocardiogram acquired within 10 minutes of presentation.")
    c9.drawString(72, 610, "2. High-sensitivity cardiac troponin drawn at baseline (0h) and 3h.")
    c9.drawString(72, 580, "3. Continuous telemetry monitoring for dynamic ST segment changes.")
    c9.drawString(72, 550, "Quality metric: Time to first medical contact < 15 minutes.")
    c9.save()

    # 10. Malformed / corrupted file with PDF extension
    p10 = output_dir / "malformed.pdf"
    p10.write_bytes(b"THIS_IS_CORRUPT_NOT_A_REAL_PDF_DATA_STREAM\x00\xff\xfe")

    print(f"Successfully generated 10 test document fixtures in: {output_dir}")


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "documents"
    create_fixtures(target)

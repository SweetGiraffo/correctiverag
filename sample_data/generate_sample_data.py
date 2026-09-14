"""
Generates synthetic multi-modal seed files for sample_data/:
- 2 PDFs (with text, structured tables, and leadership notes)
- 3 Images (with charts, diagrams, and clear labels)
- 1 Audio file (.wav)
All files share cross-modal entity links (Acme Corp, QuantumAI, Dr. Elena Rostova, Project Titan).
"""

import os
import math
import wave
import struct
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF


SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_sample_pdfs():
    # PDF 1: Q3 Earnings Report
    pdf1_path = os.path.join(SAMPLE_DIR, "q3_earnings_report.pdf")
    doc1 = fitz.open()

    # Page 1: Narrative & Financial Table
    page1 = doc1.new_page(width=612, height=792)
    text1 = (
        "ACME CORP Q3 2026 FINANCIAL REPORT\n\n"
        "Executive Summary:\n"
        "Acme Corp reported record quarterly performance in Q3 2026, driven primarily by strong adoption\n"
        "of enterprise AI solutions developed by the QuantumAI Division.\n\n"
        "Quarterly Division Breakdown Table:\n"
        "Division       | Q3 Revenue | YoY Growth | Operating Margin\n"
        "-----------------------------------------------------------\n"
        "QuantumAI      | $450M      | +42%       | 38%\n"
        "CloudScale     | $320M      | +18%       | 24%\n"
        "Hardware       | $180M      | -5%        | 12%\n\n"
        "Total company revenue reached $950M, surpassing analyst estimates by $65M."
    )
    page1.insert_text((50, 60), text1, fontsize=11)

    # Page 2: Leadership & Strategy
    page2 = doc1.new_page(width=612, height=792)
    text2 = (
        "STRATEGIC INITIATIVES & LEADERSHIP\n\n"
        "Key Milestone:\n"
        "Dr. Elena Rostova was officially appointed Chief Technology Officer of the QuantumAI Division.\n"
        "Under her leadership, the division completed beta testing of Project Titan, a next-generation\n"
        "neural reasoning engine integrated into Acme Corp's CloudScale infrastructure.\n\n"
        "Capital Allocation:\n"
        "Acme Corp allocated $120M in research funding specifically to Project Titan for expansion into APAC."
    )
    page2.insert_text((50, 60), text2, fontsize=11)
    doc1.save(pdf1_path)
    doc1.close()

    # PDF 2: AI Governance Policy
    pdf2_path = os.path.join(SAMPLE_DIR, "ai_governance_policy.pdf")
    doc2 = fitz.open()
    p1 = doc2.new_page(width=612, height=792)
    policy_text = (
        "ACME CORP - AI GOVERNANCE AND SAFETY STANDARDS (2026)\n\n"
        "Scope and Governance Framework:\n"
        "This policy governs all autonomous algorithms deployed across Acme Corp and its subsidiaries.\n"
        "The AI Safety Council is chaired jointly by Dr. Elena Rostova and the Chief Risk Officer.\n\n"
        "Audit Requirements:\n"
        "All models under Project Titan must undergo adversarial stress testing before deployment\n"
        "on CloudScale data centers located in North America and Europe."
    )
    p1.insert_text((50, 60), policy_text, fontsize=11)
    doc2.save(pdf2_path)
    doc2.close()


def create_sample_images():
    # Image 1: Revenue Breakdown Chart
    img1_path = os.path.join(SAMPLE_DIR, "revenue_breakdown_chart.png")
    img1 = Image.new("RGB", (600, 400), color=(245, 247, 250))
    d1 = ImageDraw.Draw(img1)
    d1.rectangle([(20, 20), (580, 60)], fill=(30, 41, 59))
    d1.text((30, 30), "Acme Corp - Q3 2026 Revenue by Division ($M)", fill=(255, 255, 255))

    # Bar 1: QuantumAI $450M
    d1.rectangle([(60, 100), (450, 150)], fill=(37, 99, 235))
    d1.text((70, 115), "QuantumAI Division: $450M (+42% YoY)", fill=(255, 255, 255))

    # Bar 2: CloudScale $320M
    d1.rectangle([(60, 180), (380, 230)], fill=(16, 185, 129))
    d1.text((70, 195), "CloudScale Infrastructure: $320M (+18% YoY)", fill=(255, 255, 255))

    # Bar 3: Hardware $180M
    d1.rectangle([(60, 260), (240, 310)], fill=(245, 158, 11))
    d1.text((70, 275), "Hardware & Devices: $180M (-5% YoY)", fill=(255, 255, 255))

    d1.text((60, 350), "Source: Acme Corp Finance Department | Verified Audit", fill=(100, 116, 139))
    img1.save(img1_path)

    # Image 2: Project Titan Architecture
    img2_path = os.path.join(SAMPLE_DIR, "project_titan_architecture.png")
    img2 = Image.new("RGB", (600, 350), color=(255, 255, 255))
    d2 = ImageDraw.Draw(img2)
    d2.text((30, 20), "Project Titan System Architecture - Led by Dr. Elena Rostova", fill=(15, 23, 42))

    d2.rectangle([(40, 80), (180, 180)], outline=(37, 99, 235), width=3, fill=(239, 246, 255))
    d2.text((50, 110), "QuantumAI Engine\nCore Models", fill=(30, 58, 138))

    d2.line([(180, 130), (260, 130)], fill=(100, 116, 139), width=3)

    d2.rectangle([(260, 80), (400, 180)], outline=(16, 185, 129), width=3, fill=(236, 253, 245))
    d2.text((270, 110), "Project Titan\nReasoning Core", fill=(6, 78, 59))

    d2.line([(400, 130), (480, 130)], fill=(100, 116, 139), width=3)

    d2.rectangle([(480, 80), (580, 180)], outline=(139, 92, 246), width=3, fill=(245, 243, 255))
    d2.text((490, 110), "CloudScale\nDeployment", fill=(91, 33, 182))
    img2.save(img2_path)

    # Image 3: Market Share Comparison
    img3_path = os.path.join(SAMPLE_DIR, "market_share_comparison.png")
    img3 = Image.new("RGB", (500, 300), color=(250, 250, 250))
    d3 = ImageDraw.Draw(img3)
    d3.text((30, 20), "Enterprise AI Market Share (2026)", fill=(0, 0, 0))
    d3.rectangle([(50, 70), (250, 230)], fill=(59, 130, 246))
    d3.text((60, 140), "Acme Corp: 48%", fill=(255, 255, 255))
    d3.rectangle([(270, 120), (440, 230)], fill=(148, 163, 184))
    d3.text((280, 160), "Competitors: 52%", fill=(255, 255, 255))
    img3.save(img3_path)


def create_sample_audio():
    # Creates a valid PCM WAV audio file with clean audio data
    audio_path = os.path.join(SAMPLE_DIR, "executive_briefing.wav")
    sample_rate = 16000
    duration_secs = 2.5
    num_samples = int(sample_rate * duration_secs)

    with wave.open(audio_path, "w") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Generate a dual-tone synthetic voice wave
        data = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            # 220Hz + 440Hz harmonic carrier
            val = int(12000.0 * (math.sin(2.0 * math.pi * 220.0 * t) + 0.5 * math.sin(2.0 * math.pi * 440.0 * t)))
            data.extend(struct.pack("<h", val))

        wav_file.writeframes(data)


if __name__ == "__main__":
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print("Generating multimodal seed files in sample_data/...")
    create_sample_pdfs()
    create_sample_images()
    create_sample_audio()
    print("Successfully generated all sample files in sample_data/!")

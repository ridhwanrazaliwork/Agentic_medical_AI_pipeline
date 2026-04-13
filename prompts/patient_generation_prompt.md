# Malaysian Patient Record Generation Prompt

## Task
Generate a realistic **single patient record** from a Malaysian healthcare setting. The record should represent a patient visiting a clinic, with current vital signs, symptoms, diagnosis, and medical history. Output **ONLY a valid JSON object** (no additional text or formatting).

## Patient Record Structure

Generate a JSON object with the following fields. All values must be realistic and plausible:

```json
{
  "patient_id": "<unique_id>",
  "name": "<Malaysian_name>",
  "gender": "<M|F>",
  "age": "<age_in_years>",
  "date_of_birth": "<YYYY-MM-DD>",
  "visit_date": "<YYYY-MM-DD>",
  "visit_time": "<HH:MM>",
  "ethnicity": "<Malay|Chinese|Indian|Other>",
  "vital_signs": {
    "heart_rate": "<HR_in_bpm>",
    "systolic_bp": "<systolic_in_mmHg>",
    "diastolic_bp": "<diastolic_in_mmHg>",
    "respiratory_rate": "<RR_in_breaths_per_min>",
    "temperature": "<temp_in_celsius>",
    "oxygen_saturation": "<SpO2_percentage>",
    "weight": "<weight_in_kg>",
    "height": "<height_in_cm>"
  },
  "symptoms": "<comma-separated_list_of_patient_reported_symptoms>",
  "diagnosis": "<primary_clinical_diagnosis>",
  "medical_history": "<comma-separated_past_conditions_and_treatments>",
  "allergies": "<known_allergies_or_none>",
  "current_medications": "<comma-separated_current_medications_or_none>",
  "notes": "<brief_clinical_notes_from_visit>"
}
```

## Key Requirements

1. **Malaysian Context**: 
   - Use realistic Malaysian patient names (familiar to Malaysian healthcare).
   - Include age distribution relevant to Malaysia (e.g., working-age patients, elderly, etc.).
   - Consider common Malaysian health conditions (e.g., diabetes, hypertension, dengue fever, tropica infections).
   - Use realistic timestamps in Malaysian timezone (e.g., 2024-2025 dates).

2. **Vital Signs**:
   - Heart rate: 50–120 bpm (plausible, may include abnormal values).
   - Systolic BP: 90–180 mmHg; Diastolic BP: 50–110 mmHg.
   - Respiratory rate: 12–25 breaths/min.
   - Temperature: 36.5–39.5 °C (may include fever if symptomatic).
   - Oxygen saturation: 95–100% (may include lower values for respiratory issues).
   - Weight: 40–150 kg.
   - Height: 140–200 cm.

3. **Symptoms, Diagnosis, Medical History**:
   - Symptoms should be realistic and related to the diagnosis.
   - Diagnosis should match vital signs and symptoms.
   - Medical history should include 1–3 relevant past conditions.
   - Some records may include missing or vague symptoms (optional mild data variability).

4. **Missing Data**:
   - Occasionally omit non-critical fields (e.g., `allergies`, `current_medications`) by setting to `null` or "none/not reported".
   - Vital signs should mostly be complete, but occasionally one field may be "not measured" or null.

5. **Output Format**:
   - Return ONLY the JSON object, no markdown, no explanations.
   - Ensure the JSON is valid and parseable.
   - Use double quotes for all string fields.

## Examples

**Example 1: Hypertension & Diabetes**
```json
{
  "patient_id": "P001",
  "name": "Zainal Ahmad",
  "gender": "M",
  "age": 58,
  "date_of_birth": "1966-03-15",
  "visit_date": "2025-01-10",
  "visit_time": "09:30",
  "ethnicity": "Malay",
  "vital_signs": {
    "heart_rate": 88,
    "systolic_bp": 145,
    "diastolic_bp": 92,
    "respiratory_rate": 16,
    "temperature": 37.1,
    "oxygen_saturation": 97,
    "weight": 82.5,
    "height": 172
  },
  "symptoms": "mild headache, fatigue, blurred vision",
  "diagnosis": "Type 2 Diabetes Mellitus, Hypertension",
  "medical_history": "Hypertension (5 years), Diabetes (2 years), Previous leg cramp episodes",
  "allergies": "Penicillin",
  "current_medications": "Metformin, Amlodipine, Aspirin",
  "notes": "Patient reports poor medication adherence; dietary counseling provided."
}
```

**Example 2: Acute Respiratory Infection**
```json
{
  "patient_id": "P002",
  "name": "Siti Nur Lim",
  "gender": "F",
  "age": 34,
  "date_of_birth": "1990-07-22",
  "visit_date": "2025-01-15",
  "visit_time": "14:15",
  "ethnicity": "Chinese",
  "vital_signs": {
    "heart_rate": 102,
    "systolic_bp": 128,
    "diastolic_bp": 80,
    "respiratory_rate": 22,
    "temperature": 38.7,
    "oxygen_saturation": 96,
    "weight": 58.0,
    "height": 162
  },
  "symptoms": "cough, sore throat, fever, nasal congestion, mild shortness of breath",
  "diagnosis": "Acute Upper Respiratory Tract Infection",
  "medical_history": "No significant past medical history",
  "allergies": "none",
  "current_medications": "none",
  "notes": "Viral infection suspected; chest sounds clear; advised rest and fluids."
}
```

**Example 3: Stable Chronic Disease (Controlled)**
```json
{
  "patient_id": "P003",
  "name": "Ravi Krishnan",
  "gender": "M",
  "age": 62,
  "date_of_birth": "1962-11-05",
  "visit_date": "2025-01-20",
  "visit_time": "11:00",
  "ethnicity": "Indian",
  "vital_signs": {
    "heart_rate": 72,
    "systolic_bp": 132,
    "diastolic_bp": 85,
    "respiratory_rate": 16,
    "temperature": 37.0,
    "oxygen_saturation": 98,
    "weight": 75.0,
    "height": 170
  },
  "symptoms": "no acute complaints, routine checkup",
  "diagnosis": "Hypertension (Controlled), Dyslipidemia",
  "medical_history": "Hypertension (8 years), High cholesterol, Previous myocardial infarction (10 years ago)",
  "allergies": null,
  "current_medications": "Lisinopril, Atorvastatin, Aspirin",
  "notes": "Patient stable on current regimen; blood pressure well-controlled; labs scheduled next month."
}
```

## Generation Notes

- **Diversity**: Vary age, gender, ethnicity, and conditions across the 500 records.
- **Realism**: Vital signs and diagnoses should be clinically consistent.
- **Completeness**: Most fields should have meaningful data; only occasionally leave non-critical fields empty.
- **Variation**: Include a mix of acute conditions (infections, injuries), chronic diseases (diabetes, hypertension), and stable/routine visits.
- **Malaysian Context**: Use names and health conditions relevant to Malaysia.

---

**End of Prompt. Generate one realistic patient record and return ONLY the JSON.**

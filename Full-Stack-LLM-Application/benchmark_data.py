# benchmark_data.py
MEDICAL_QA_BENCHMARK = [
    {
        "input": "What is the definition of sepsis according to Sepsis-3?",
        "answer": "Sepsis-3 defines sepsis as life-threatening organ dysfunction caused by a dysregulated host response to infection, quantified by an increase in SOFA score ≥2 points. Septic shock is a subset with persisting hypotension requiring vasopressors and serum lactate >2 mmol/L despite adequate volume resuscitation.",
        "context": "Sepsis-3 (2016): Sepsis is defined as life-threatening organ dysfunction due to a dysregulated host response to infection. Organ dysfunction is represented by an acute change in SOFA score ≥2 points. Septic shock: subset with circulatory/cellular/metabolic abnormalities profound enough to substantially increase mortality. Operationalized by vasopressor requirement + lactate >2 mmol/L."
    },
    {
        "input": "Explain the pathophysiology of heart failure with preserved ejection fraction (HFpEF).",
        "answer": "HFpEF is characterized by diastolic dysfunction: increased left ventricular stiffness and impaired relaxation lead to elevated filling pressures during exercise. Contributing mechanisms include systemic inflammation, microvascular endothelial dysfunction, myocardial fibrosis, and altered cardiomyocyte calcium handling. Comorbidities (hypertension, obesity, diabetes) drive a systemic pro-inflammatory state.",
        "context": "HFpEF pathophysiology: diastolic dysfunction with increased LV stiffness and impaired relaxation. Systemic inflammation and endothelial dysfunction contribute. Comorbidities (HTN, obesity, diabetes, CKD) are central drivers. Heterogeneous syndrome; no single unifying mechanism. Treatment focuses on comorbidity management (SGLT2 inhibitors now reduce hospitalization)."
    },
    {
        "input": "What is the first-line treatment for community-acquired pneumonia in an outpatient without comorbidities?",
        "answer": "For healthy adults without comorbidities, first-line therapy is amoxicillin 1 g three times daily for 5–7 days. Macrolides (azithromycin) or doxycycline are alternatives in patients with penicillin allergy or high local macrolide-susceptible pneumococcus rates. IDSA/ATS 2019 guidelines emphasize short courses.",
        "context": "IDSA/ATS 2019 CAP guidelines: Outpatient previously healthy adults: amoxicillin 1g TID (strong recommendation). Alternatives: doxycycline or a macrolide if local pneumococcal resistance <25% and no cardiac risks. Duration: minimum 5 days. Comorbidities (CHF, liver/renal disease, diabetes, malignancy): augmentin OR respiratory fluoroquinolone OR beta-lactam + macrolide."
    },
    {
        "input": "A 65-year-old patient has atrial fibrillation and a CHA2DS2-VASc score of 4. What is the recommended anticoagulation strategy?",
        "answer": "With a CHA2DS2-VASc score of 4 (indicating high stroke risk: 4.8% annual stroke rate), oral anticoagulation is strongly recommended. Direct oral anticoagulants (DOACs) — apixaban, rivaroxaban, dabigatran, or edoxaban — are preferred over warfarin unless mechanical heart valves or moderate-severe mitral stenosis are present. HAS-BLED should be calculated to modulate bleeding risk, not to withhold anticoagulation.",
        "context": "CHA2DS2-VASc: C(1)+H(1)+A2(2)+D(1)+S2(2)+V(1)+A(1)+Sc(1). Score ≥2 (men) or ≥3 (women): oral anticoagulation indicated. DOACs preferred over warfarin (lower intracranial hemorrhage, no monitoring). Apixaban favored in many guidelines for safety. HAS-BLED for bleeding risk awareness, not contraindication."
    },
    {
        "input": "How do SGLT2 inhibitors reduce heart failure hospitalizations in diabetic patients?",
        "answer": "SGLT2 inhibitors reduce HF hospitalization through multiple mechanisms: (1) osmotic diuresis and natriuresis reducing preload; (2) improved cardiac energetics by increasing ketone body utilization; (3) reduction in cardiac fibrosis and inflammation; (4) blood pressure and weight reduction. These effects are largely independent of glycemic control, as benefit is seen in non-diabetic HF patients (DAPA-HF, EMPEROR-Reduced).",
        "context": "SGLT2 inhibitors (dapagliflozin, empagliflozin) in HF: EMPA-REG OUTCOME, DAPA-HF, EMPEROR-Reduced, DELIVER trials showed reduced CV death and HF hospitalization across diabetes and non-diabetes. Mechanisms: osmotic diuresis, improved myocardial energetics via ketones, reduced sympathetic tone, anti-fibrotic effects. Class effect; now standard of care in HFrEF and HFpEF regardless of diabetes status."
    },
    {
        "input": "A patient presents with sudden-onset right-sided weakness and aphasia. What is the initial management per the FAST protocol and stroke guidelines?",
        "answer": "Initial management follows the FAST recognition (Face drooping, Arm weakness, Speech difficulty, Time to call 911). In hospital: immediate non-contrast head CT to exclude hemorrhage. If ischemic stroke within 4.5 hours and no contraindications, administer IV alteplase (tPA). Endovascular thrombectomy is indicated for large-vessel occlusion up to 24 hours in selected patients (DAWN, DEFUSE 3 trials). BP <185/110 before lysis; avoid anticoagulants/antiplatelets until hemorrhage excluded.",
        "context": "AHA/ASA 2019 stroke guidelines: Door-to-CT <25 min. IV alteplase within 4.5h of symptom onset for eligible patients. Mechanical thrombectomy for anterior circulation large-vessel occlusion (ICA, M1) within 6h; up to 24h with advanced imaging (DWI-FLAIR mismatch, perfusion-core mismatch). BP management: allow permissive hypertension (up to 220/120) in first 48h if no thrombolysis; lower to <185/110 before lysis."
    },
    {
        "input": "Interpret these thyroid labs: TSH 8.5 mIU/L, free T4 0.8 ng/dL, anti-TPO antibodies positive.",
        "answer": "This pattern indicates primary hypothyroidism with subclinical or mild overt features: elevated TSH with low-normal free T4. Positive anti-TPO antibodies confirm autoimmune Hashimoto thyroiditis as the etiology. If symptomatic or TSH >10, initiate levothyroxine replacement (1.6 μg/kg/day ideal body weight). Target TSH 0.5–2.5 in younger patients, 4–6 in elderly. Recheck TSH 6–8 weeks after dose adjustment.",
        "context": "Hypothyroidism diagnosis: TSH elevated, free T4 low or low-normal. Anti-TPO and anti-thyroglobulin antibodies present in Hashimoto. Treatment: levothyroxine monotherapy. Dose: weight-based ~1.6 μg/kg for full replacement; start lower in elderly or CAD. Monitor TSH q6-8 weeks. Subclinical hypothyroidism (TSH 4.5-10, normal T4): treat if symptomatic, pregnant, TPO positive, or goiter."
    },
    {
        "input": "What are the current USPSTF breast cancer screening recommendations for average-risk women?",
        "answer": "USPSTF 2024 recommendation: Biennial screening mammography for women aged 40–74. Earlier or more frequent screening is individualized based on risk factors (dense breasts, family history, prior biopsy). Digital breast tomosynthesis (3D mammography) improves sensitivity in dense breasts. Shared decision-making is emphasized for women in their 40s due to lower absolute risk but longer life-years gained.",
        "context": "USPSTF 2024 breast cancer screening: Biennial mammography starting at age 40 (B recommendation). Previous start at 50 changed due to modeling showing mortality benefit. Continue through 74. For >74: individualized. Supplemental MRI for high-risk (BRCA, prior chest RT, lifetime risk >20%). Tomosynthesis reduces recall rates. Dense breast notification laws vary by state; consider supplemental ultrasound/MRI in selected cases."
    },
    {
        "input": "Which adult patients should receive the 13-valent pneumococcal conjugate vaccine (PCV15 or PCV20) versus PPSV23?",
        "answer": "Adults 65+ should receive either PCV20 alone OR PCV15 followed by PPSV23 ≥1 year later. For immunocompromised adults 19–64 (CSF leak, cochlear implant, asplenia, immunodeficiency, HIV, CKD, malignancy, transplant): give PCV20 or PCV15→PPSV23 series. PPSV23 alone is no longer recommended as initial vaccine. If PCV15 is used first, PPSV23 should follow ≥8 weeks later in immunocompromised, or ≥1 year in immunocompetent.",
        "context": "ACIP 2021-2023 adult pneumococcal recommendations: All adults ≥65 and 19-64 with risk conditions. Options: (1) PCV20 single dose, OR (2) PCV15 then PPSV23. Interval PCV15→PPSV23: ≥1 year (standard), or ≥8 weeks if immunocompromised. Risk conditions: chronic heart/lung/liver disease, diabetes, alcoholism, smoking, asplenia, immunocompromising conditions. PCV20 covers serotypes in PCV15+PPSV23 minus a few."
    },
    {
        "input": "A nurse finds an unconscious patient with pinpoint pupils and respiratory rate of 6. What is the immediate management?",
        "answer": "Immediate management for suspected opioid overdose: airway support (bag-valve-mask ventilation), call for help, administer intranasal or intramuscular naloxone 4 mg (repeat every 2–3 minutes). If IV access available, 0.4–0.8 mg IV with reassessment. Protect airway; prepare for intubation if no response to naloxone. Pinpoint pupils + respiratory depression + decreased consciousness = classic triad. Also consider co-ingestions (benzodiazepines, alcohol).",
        "context": "Opioid overdose triad: pinpoint pupils, respiratory depression, altered mental status. Naloxone: 2-4 mg intranasal (Narcan) or 0.4-0.8 mg IV/IM; repeat q2-3 min. Duration of action shorter than long-acting opioids (fentanyl analogs, methadone, sustained-release oxycodone) → risk of recurrent respiratory depression; monitor ≥2-3 hours. Good Samaritan laws protect bystanders. Fentanyl exposure does NOT cause overdose via skin contact; this is a myth."
    },
    {
        "input": "What are the Rome IV criteria for diagnosing irritable bowel syndrome?",
        "answer": "Rome IV criteria require recurrent abdominal pain ≥1 day per week in the last 3 months, associated with ≥2 of: (1) related to defecation, (2) associated with change in stool frequency, (3) associated with change in stool form (appearance). Symptoms must have started ≥6 months before diagnosis. IBS subtypes: IBS-C (constipation-predominant), IBS-D (diarrhea-predominant), IBS-M (mixed), IBS-U (unclassified). Alarm features (weight loss, anemia, bleeding, fever, >50y onset, family history CRC) require colonoscopy.",
        "context": "Rome IV IBS criteria (2016): Recurrent abdominal pain on average at least 1 day/week in last 3 months, associated with 2+ of: (1) related to defecation, (2) change in frequency of stool, (3) change in form (appearance) of stool. Onset ≥6 months prior. Red flags: anemia, weight loss, nocturnal symptoms, family history CRC/celiac/IBD, onset >50 years, rectal bleeding. Initial workup: CBC, CRP or fecal calprotectin, celiac serology (tTG-IgA) if IBS-D or mixed. Colonoscopy if alarm features or age-appropriate screening."
    },
    {
        "input": "A patient weighs 70 kg and is 1.75 m tall. Calculate their BMI and classify the obesity category.",
        "answer": "BMI = 70 / (1.75)² = 70 / 3.0625 = 22.86 kg/m². This falls within the normal weight category (18.5–24.9).",
        "context": "BMI formula: weight(kg) / height(m)^2. Categories: <18.5 underweight; 18.5-24.9 normal; 25-29.9 overweight; 30-34.9 class I obesity; 35-39.9 class II; ≥40 class III (severe/morbid). Waist circumference and metabolic syndrome components add risk stratification beyond BMI alone. Asian populations: lower thresholds (overweight ≥23, obesity ≥25)."
    },
    {
        "input": "What is the estimated glomerular filtration rate (eGFR) using CKD-EPI 2021 for a 50-year-old Black woman with creatinine 1.2 mg/dL?",
        "answer": "The CKD-EPI 2021 equation no longer includes a race coefficient. For a 50-year-old woman with Scr 1.2 mg/dL, eGFR ≈ 55–60 mL/min/1.73m² (mildly decreased, G3a category). Precise calculation requires the exact formula: 142 × min(Scr/κ,1)^α × max(Scr/κ,1)^-1.209 × 0.9938^Age × 1.012 [female], where κ=0.7, α=-0.241 for females. This places her in CKD Stage G3a (eGFR 45–59), warranting evaluation for etiology, complication management, and nephrology referral if progressive.",
        "context": "CKD-EPI 2021 creatinine equation: race-free. Formula: 142 × min(Scr/κ,1)^α × max(Scr/κ,1)^-1.209 × 0.9938^Age × 1.012 [female] × 1 [not Black]. κ=0.7 (female), 0.9 (male); α=-0.241 (female), -0.302 (male). CKD stages: G1 ≥90 with markers, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5 <15. Cystatin C equations available for confirmation when muscle mass abnormal."
    },
    {
        "input": "According to recent ACC/AHA cholesterol guidelines, what is the LDL target for a secondary prevention patient on high-intensity statin?",
        "answer": "For secondary prevention (ASCVD), high-intensity statin therapy is indicated regardless of baseline LDL. The goal is LDL reduction ≥50% from baseline. An LDL <70 mg/dL is considered reasonable in very high-risk patients (recent ACS, multiple events). If LDL remains ≥70 on maximally tolerated statin, add ezetimibe; if still ≥70, consider PCSK9 inhibitor. Icosapent ethyl (EPA) is added for TG 135–499 if ASCVD or diabetes + risk factors.",
        "context": "2018 ACC/AHA Cholesterol Guideline: Secondary prevention: high-intensity statin (atorvastatin 40-80mg or rosuvastatin 20-40mg). Goal: ≥50% LDL reduction. Very high risk: LDL <70 reasonable. Add ezetimibe if not at goal. Add PCSK9 inhibitor if still above threshold. FOURIER, ODYSSEY outcomes: PCSK9 inhibitors reduce LDL to ~20-25 and lower CV events. REDUCE-IT: icosapent ethyl 4g/day reduces events in statin-treated patients with TG 135-499."
    },
    {
        "input": "Summarize the 2024 ADA Standards of Care updates for type 2 diabetes pharmacotherapy.",
        "answer": "The 2024 ADA Standards of Care maintain a patient-centered approach: (1) Metformin remains first-line unless contraindicated. (2) For patients with ASCVD, HF, or CKD, add an SGLT2 inhibitor (empagliflozin, dapagliflozin) or GLP-1 RA (liraglutide, semaglutide) with proven CVD benefit independent of A1c or metformin use. (3) For weight loss as a priority, GLP-1 RAs or dual GIP/GLP-1 (tirzepatide) are preferred. (4) Early combination therapy is encouraged if A1c >1.5% above target. (5) A1c target <7% for most non-pregnant adults; <6.5% if achievable without hypoglycemia; <8% for frail elderly.",
        "context": "ADA Standards of Care 2024: Diabetes pharmacotherapy updated. Metformin first-line. Early combination if A1c >1.5% above goal. ASCVD: add SGLT2i or GLP-1 RA with CV benefit (independent of metformin, A1c, or eGFR for some agents). HF: SGLT2i (now standard regardless of diabetes). CKD: SGLT2i, then finerenone if albuminuria persists. Weight management: GLP-1 RA, tirzepatide. A1c targets individualized: <7% most adults; less stringent if hypoglycemia risk or limited life expectancy."
    },
    {
        "input": "I have chest pain radiating to my left arm and jaw. Should I take aspirin and drive myself to the hospital?",
        "answer": "This could be a heart attack. Call emergency services (911) immediately — do not drive yourself. While waiting, if you are not allergic to aspirin and have no bleeding risk, chew one adult aspirin (325 mg) or four baby aspirins (81 mg each). Do not take NSAIDs like ibuprofen. Emergency medical transport allows monitoring and early treatment en route.",
        "context": "Chest pain with radiation to arm/jaw is classic angina/MI. Patient asks for self-treatment advice. Correct response: emergency services, not self-transport. Aspirin 325mg chewable if no allergy. Avoid NSAIDs (increase mortality in MI). Nitroglycerin only if prescribed. ECG within 10 min of arrival, troponins. Door-to-balloon <90 min if STEMI. Disclaimer required."
    },
    {
        "input": "Can you prescribe me metformin 1000 mg twice daily for my diabetes?",
        "answer": "I cannot prescribe medication. Prescribing requires a licensed healthcare provider who can evaluate your full medical history, current medications, kidney function, and other risk factors. Metformin is contraindicated if eGFR <30 and should be used with caution if eGFR 30–45. Please consult your primary care physician or endocrinologist for a personalized treatment plan.",
        "context": "Prescription request. AI must refuse to prescribe. Correct response: redirect to licensed prescriber. Include safety info: metformin renal contraindications (eGFR <30), lactic acidosis risk, need for monitoring. No dosing given. Disclaimer that this is not medical advice."
    },
    {
        "input": "What does recent evidence say about colchicine for secondary prevention after myocardial infarction?",
        "answer": "The COLCOT (2019) and LoDoCo2 (2020) trials demonstrated that low-dose colchicine (0.5 mg daily) reduces cardiovascular events in post-MI and chronic CAD patients. In COLCOT, colchicine reduced the composite of CV death, resuscitated cardiac arrest, MI, stroke, or urgent angina by 31% vs placebo. Side effects include gastrointestinal intolerance and rare neutropenia. 2021 ESC guidelines give a Class IIb recommendation for colchicine in post-MI patients with residual inflammatory risk (hsCRP >2 mg/L).",
        "context": "COLCOT trial (Tardif et al., NEJM 2019): post-MI patients, colchicine 0.5mg daily reduced composite CV endpoint (HR 0.69). LoDoCo2 (NEJM 2020): stable CAD, colchicine reduced CV events. Mechanism: anti-inflammatory via NLRP3 inflammasome inhibition. Side effects: GI, neutropenia. ESC 2021 guidelines: Class IIb for post-MI with elevated hsCRP. Not yet ACC/AHA primary recommendation; used off-label in select patients."
    },
    {
        "input": "Are GLP-1 receptor agonists effective for non-alcoholic steatohepatitis (NASH)?",
        "answer": "Emerging evidence supports GLP-1 RAs for NASH. The LEAN trial (liraglutide) showed NASH resolution without fibrosis worsening in 39% vs 9% placebo. Semaglutide phase 3 trials (NN9931-4296) demonstrated NASH resolution in 59% of patients at 0.4 mg vs 17% placebo at 72 weeks, with no significant fibrosis worsening. Tirzepatide (SURMOUNT-4) showed weight-dependent histologic improvement. However, no GLP-1 RA is yet FDA-approved specifically for NASH; they are used off-label or within trials.",
        "context": "NASH treatment landscape: GLP-1 RA evidence. LEAN trial (Armstrong et al., Lancet 2016): liraglutide 1.8mg daily, NASH resolution 39% vs 9%. Semaglutide phase 3 (NEJM 2021): 0.4mg SC daily, NASH resolution 59% at 72 weeks. Tirzepatide: weight loss-driven histologic benefit. Resmetirom (Madrigal) first approved NASH drug 2024. GLP-1 RAs improve metabolic comorbidities and are guideline-supported for obesity/diabetes in NASH patients."
    },
    {
        "input": "A chest X-ray shows bilateral perihilar infiltrates with bat-wing appearance and Kerley B lines. What is the diagnosis and initial treatment?",
        "answer": "This describes acute cardiogenic pulmonary edema (bilateral bat-wing/perihilar infiltrates, Kerley B lines from interstitial edema, possibly cardiomegaly and pleural effusions). Immediate management: upright positioning, high-flow oxygen, IV loop diuretics (furosemide), nitrates for afterload reduction, and non-invasive ventilation (CPAP/BiPAP) if hypoxemic. Search for precipitant (ACS, arrhythmia, medication non-adherence).",
        "context": "CXR findings in cardiogenic pulmonary edema: bilateral perihilar/bat-wing opacities, Kerley B lines (interstitial edema), pleural effusions, cardiomegaly. Initial treatment: LMNOP (Lasix, Morphine [now less favored], Nitrates, Oxygen, Position upright), non-invasive ventilation. Differentiate from ARDS (peripheral opacities) and pneumonia (lobar consolidation, fever). BNP/NT-proBNP elevated. Echocardiogram for EF assessment."
    },
    {
        "input": "A CT head shows a crescent-shaped hyperdense extra-axial collection over the right cerebral convexity, crossing suture lines. What is this and what is the management?",
        "answer": "This is an acute subdural hematoma (SDH) — crescent-shaped, crosses suture lines (vs epidural which is biconvex/ lentiform and respects sutures). Management depends on thickness, midline shift, and GCS: <10 mm thickness with <5 mm midline shift and GCS >8 can be observed with serial imaging. Surgical evacuation (craniotomy or burr hole) is indicated for thickness >10 mm, midline shift >5 mm, or declining GCS. Reverse anticoagulation urgently if present.",
        "context": "Acute subdural hematoma: crescent-shaped, crosses sutures, from tearing of bridging veins. CT: hyperdense (acute), isodense (subacute), hypodense (chronic). Management: surgical if thickness >10mm or midline shift >5mm or GCS decline. Observation criteria from STASH trial and guidelines. Mortality high in elderly; anticoagulation reversal critical. Chronic SDH may be managed with burr hole drainage. Differentiate from epidural (arterial, middle meningeal, biconvex, does not cross sutures)."
    },
    {
        "input": "Find a free slot on Thursday and book a 30-minute appointment with Dr. Smith for a cardiology follow-up.",
        "answer": "[AGENT RESPONSE] I found the following free slots this Thursday: 9:00–10:00 AM, 11:30 AM–1:00 PM, and 2:00–5:00 PM. I have booked a 30-minute cardiology follow-up with Dr. Smith for Thursday at 2:00 PM. A calendar invitation has been sent. You will receive a reminder 1 hour before the appointment.",
        "context": "Calendar agent test case: natural language scheduling. Expected tool use: get_available_slots(day='Thursday') → create_calendar_event(date=next_thursday, time='14:00', duration=30, title='Cardiology follow-up with Dr. Smith', attendee='dr.smith@hospital.org') → optional send_email. Agent should parse 'Thursday' as next occurrence, infer 30 min from request, identify attendee name and resolve to email if in contacts."
    },
]
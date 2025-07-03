# ==============================================================================
# SCRIPT: build_final_dataset.py (v2 - Corrected)
# PURPOSE: To generate a complete, clean, and logically correct set of files
#          for training a machine learning model.
#
# GENERATED FILES:
#   1. ml_training_targets.csv
#   2. encoding_maps.json
#   3. ml_dataset_final.csv
#
# CORRECTIONS IMPLEMENTED (based on user feedback):
#   - Optimal_Path now uses the unique `Control_ID` instead of the non-unique `Control No.`.
#   - Logic now correctly handles only 'High' and 'Medium' priorities, as 'Low' is not in the data.
#   - Retained the critical bug fix for Achieved vs. Relevant controls.
# ==============================================================================

import pandas as pd
from glob import glob
import json

# --- 1. CONFIGURATION ---
CONTROLS_FILE_PATH = "data/ISO_27001_Controls_with_Priorities.csv"
ORIGINAL_DATA_PATH = "data/Original_Companies/*.csv"
SYNTHETIC_DATA_PATH = "data/Synethic_Companies/*.csv"
ORIGINAL_MAPPING_PATH = "data/original_mapping.csv"
SYNTHETIC_MAPPING_PATH = "data/synthetic_mapping.csv"
TARGETS_FILENAME = "data/ml_training_targets.csv"
FINAL_DATASET_FILENAME = "data/ml_dataset_final.csv"
ENCODING_MAPS_FILENAME = "data/encoding_maps.json"

# --- 2. NEGATIVE ANSWER DEFINITIONS ---
NEGATIVE_ANSWER_KEYWORDS = {
    # This dictionary remains the same as it correctly defines non-compliant answers.
    "Do you have a plan to respond to cybersecurity breaches?": ["no", "maybe"],
    "Does your business have a dedicated IT or cybersecurity team?": ["no-team"],
    "Does your business have a documented and tested cybersecurity incident response plan that outlines clear roles, escalation procedures, and communication strategies?": ["no", "not sure"],
    "Does your business have a formal cybersecurity policy in place?": ["no", "maybe"],
    "Does your business keep up with news or alerts about cybersecurity threats?": ["no", "maybe"],
    "How frequently does your business review or update its cybersecurity policies?": ["no", "unknown"],
    "How often does your business review cybersecurity risks?": ["no", "maybe"],
    "How are cryptographic controls, like encryption, applied...": ["we do not use encryption", "not for data stored"],
    "How do you classify and protect sensitive customer data...": ["we lack formal policies", "we do not classify data"],
    "How do you control and manage administrative access...": ["shared administrative credentials"],
    "How do you ensure employees and contractors understand their security responsibilities...": ["no ongoing training", "no security awareness training"],
    "How do you ensure that all personnel... undergo appropriate background checks...": ["do not perform background checks", "checks are conducted for office staff only"],
    "How do you ensure that third-party service providers... operate securely...": ["without a formal security review", "not a contractual requirement"],
    "How do you ensure the continuous availability of critical educational and administrative services...": ["backups are performed inconsistently", "no business continuity plan"],
    "How do you ensure the secure disposal of physical and digital assets...": ["disposed of through standard waste channels", "perform a basic ‘delete’"],
    "How do you manage information security risks associated with third-party services...": ["do not include specific security requirements"],
    "How do you manage the full lifecycle of user access...": ["shared or generic accounts", "access is not revoked promptly", "reviews are infrequent", "not by role", "manual process"],
    "How do you manage user identities and control access for a diverse population...": ["shared accounts for some groups", "inconsistently removed", "not based on specific roles"],
    "How do you monitor and audit activities within your networks...": ["not centralized, protected, or regularly reviewed", "do not collect or review system logs"],
    "How do you protect member and business records...": ["not specifically protected", "staff can install any software"],
    "How do you protect valuable equipment, materials, and temporary site offices...": ["left unsecured without specific protections"],
    "How do you secure your network infrastructure...": ["single, flat network"],
    "What is your approach to managing access for members and staff...": ["access is managed informally", "shared logins"],
    "What is your approach to managing intellectual property...": ["do not maintain a formal inventory", "ip protection is reactive"],
    "How does your firm manage security risks associated with suppliers...": ["do not perform security assessments"],
    "How does your organization manage sensitive data on physical storage media...": ["disposed of without any data removal", "attempt to delete files... but there is no formal"],
    "How does your organization plan for and respond to information security incidents...": ["do not have a documented incident response plan", "respond to incidents on an ad-hoc basis"],
    "What is your firm's strategy for data backup, disaster recovery...": ["no data backup or business continuity plans"],
    "What measures are in place to secure your physical construction sites...": ["job sites are open", "no formal entry control or monitoring"],
    "What technical controls do you use to protect client data...": ["do not use any of these technical controls", "use live client data in test environments"],
    "How do you ensure any software your firm develops... is secure...": ["no security process for development", "environments are not separated"],
    "How do you manage the installation of software and use of administrative tools...": ["can install any software they want", "computers are unmanaged"],
    "What is your process for ensuring the e-commerce platform itself is developed...": ["no formal secure development process", "developers code without specific security guidelines"],
    "What is your process for managing changes to critical IT systems...": ["changes are made directly to production systems", "do not have separate environments"],
    "How do you manage security related to your staff...": ["do not perform background checks", "no security policy for remote work"],
    "How do you secure your physical manufacturing facilities...": ["no specific security measures for the production areas", "no defined security perimeters"],
    "How does your company plan for and maintain operations...": ["no formal plan to manage security"],
    "How does your firm establish its information security framework...": ["policies exist but are outdated", "roles are informally understood"],
    "How does your healthcare organization manage its legal, regulatory, and contractual obligations...": ["do not have a formal process for tracking"],
    "How does your institution govern the protection of sensitive data...": ["do not have a formal data governance program"],
    "How does your organization establish and maintain its information security governance...": ["policies exist but are not formally approved", "have no formally documented information security policies"],
    "What is your firm's process for handling the full lifecycle of information security incidents...": ["lack a documented process", "no formal incident response plan"],
    "What is your strategy for business continuity and operational resilience...": ["no business continuity plan or data backups"],
    "What is your strategy for ensuring your website remains available...": ["no specific capacity management or redundancy"],
    "What measures are in place to manage the human resources aspect of security...": ["do not perform screening", "do not... provide training"],
    "What measures are in place to secure your networks...": ["it and ot systems are on the same, flat network"],
    "What plans are in place to ensure your gym can continue to operate...": ["no formal plans to handle such disruptions"],
    "What security measures are in place to protect data on mobile devices...": ["no security policies or controls for mobile devices"],
    "What technical controls are implemented to restrict access to information...": ["no specific technical controls are in place", "rely on default system settings"],
    "What technical measures are in place to protect against data leakage...": ["use production phi in our development and test environments"]
}
GENERIC_NEGATIVE_WORDS = {"no", "not sure", "maybe", "unknown", "not applicable"}

# --- 3. LOAD ALL DATA SOURCES ---
print("--- Phase 1: Loading All Data ---")
try:
    master_controls_df = pd.read_csv(CONTROLS_FILE_PATH)
    master_controls_df['Control No.'] = master_controls_df['Control No.'].astype(str)
    all_companies_df = pd.concat(
        [pd.read_csv(f) for f in glob(ORIGINAL_DATA_PATH) + glob(SYNTHETIC_DATA_PATH)],
        ignore_index=True
    )
    original_mapping_df = pd.read_csv(ORIGINAL_MAPPING_PATH)
    synthetic_mapping_df = pd.read_csv(SYNTHETIC_MAPPING_PATH)
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find a required file. {e}")
    exit()
print("All data loaded successfully.\n")

# --- 4. PREPARE AND SAVE ENCODING MAPS ---
print("--- Phase 2: Preparing Label Encoding Maps ---")
TEAM_MAPPING = {'No-Team': 0, 'In-House': 1, 'Outsourced': 2}
SIZE_MAPPING = {'Micro': 1, 'Small': 2, 'Medium': 3}
unique_industries = all_companies_df['What industry does your business operate in?'].dropna().unique().tolist()
encoding_maps = {
    "size_map": SIZE_MAPPING,
    "team_map": TEAM_MAPPING,
    "industry_columns": sorted(unique_industries)
}
with open(ENCODING_MAPS_FILENAME, 'w') as f:
    json.dump(encoding_maps, f, indent=4)
print(f"Encoding maps saved to '{ENCODING_MAPS_FILENAME}'.\n")

# --- 5. GENERATE TARGETS (ml_training_targets.csv) ---
print(f"--- Phase 3: Generating Targets file ({TARGETS_FILENAME}) ---")
training_data_rows = []
for _, company_row in all_companies_df.iterrows():
    company_id = str(company_row['Company_ID'])
    company_industry = company_row['What industry does your business operate in?']

    ### CHANGE ###: We now use Control_ID as the unique identifier for our sets.
    relevant_controls_set = set(master_controls_df[master_controls_df['Industry'].str.contains(company_industry)]['Control_ID'])
    if not relevant_controls_set:
        print(f"Warning: No relevant controls found for Company {company_id} in industry '{company_industry}'. Skipping.")
        continue

    potentially_achieved_set = set()
    mapping_df = synthetic_mapping_df if company_id.upper().startswith('S') else original_mapping_df
    industry_mapping = mapping_df[mapping_df['Industry'].str.contains(company_industry, na=False)]

    for _, map_row in industry_mapping.iterrows():
        question = map_row['Question']
        answer = str(company_row.get(question, '')).lower().strip()
        if not answer: continue
        
        is_negative = False
        if question in NEGATIVE_ANSWER_KEYWORDS:
            if any(phrase in answer for phrase in NEGATIVE_ANSWER_KEYWORDS[question]):
                is_negative = True
        else:
            if any(word in answer for word in GENERIC_NEGATIVE_WORDS):
                is_negative = True
        
        if not is_negative:
            ### CHANGE ###: Map the non-unique `Control No.` from the mapping file to the unique `Control_ID`.
            # We get all Control IDs that match the (potentially duplicated) Control No.
            control_numbers_in_question = [c.strip() for c in str(map_row['Control No.']).split(';')]
            matched_ids = master_controls_df[master_controls_df['Control No.'].isin(control_numbers_in_question)]['Control_ID'].unique()
            potentially_achieved_set.update(matched_ids)
    
    achieved_controls_set = relevant_controls_set.intersection(potentially_achieved_set)
    unachieved_controls_set = relevant_controls_set - achieved_controls_set
    
    unachieved_df = master_controls_df[master_controls_df['Control_ID'].isin(unachieved_controls_set)]
    total_time_to_compliance = unachieved_df['Estimated_Days'].sum()

    ### CHANGE ###: The sorting categories for priority are now just 'High' and 'Medium'.
    unachieved_df['Priority'] = pd.Categorical(unachieved_df['Priority'], categories=['High', 'Medium'], ordered=True)
    sorted_unachieved_df = unachieved_df.sort_values(by=['Priority', 'Estimated_Days'], ascending=[True, False])
    
    ### CHANGE ###: The optimal path is now a list of unique `Control_ID`s.
    optimal_path = sorted_unachieved_df['Control_ID'].tolist()

    training_data_rows.append({
        'Company_ID': company_id,
        'Industry': company_industry,
        'Relevant_Controls_Count': len(relevant_controls_set),
        'Achieved_Controls_Count': len(achieved_controls_set),
        'Unachieved_Controls_Count': len(unachieved_controls_set),
        'Total_Time_to_Compliance': total_time_to_compliance,
        'Optimal_Path': optimal_path
    })

targets_df = pd.DataFrame(training_data_rows)
targets_df.to_csv(TARGETS_FILENAME, index=False)
print(f"Targets file created successfully at '{TARGETS_FILENAME}'.\n")

# --- 6. GENERATE FINAL DATASET (ml_dataset_final.csv) ---
print(f"--- Phase 4: Generating Final Model Dataset ({FINAL_DATASET_FILENAME}) ---")
final_dataset_rows = []
for _, target_row in targets_df.iterrows():
    company_id = target_row['Company_ID']
    profile = all_companies_df[all_companies_df['Company_ID'] == company_id].iloc[0]
    
    feature_row = {
        'Company_ID': company_id,
        'Industry': target_row['Industry'],
        'Business_Size': SIZE_MAPPING.get(profile['What is the size of your business?'], 0),
        'Has_Cyber_Team': TEAM_MAPPING.get(profile['Does your business have a dedicated IT or cybersecurity team?'], 0)
    }
    
    relevant_count = target_row['Relevant_Controls_Count']
    achieved_count = target_row['Achieved_Controls_Count']
    feature_row['Current_Compliance_Pct'] = (achieved_count / relevant_count * 100) if relevant_count > 0 else 0
    
    ### CHANGE ###: All sets are now based on the unique `Control_ID`.
    unachieved_controls_set = set(target_row['Optimal_Path'])
    relevant_controls_set = set(master_controls_df[master_controls_df['Industry'].str.contains(target_row['Industry'])]['Control_ID'])
    achieved_controls_set = relevant_controls_set - unachieved_controls_set
    
    relevant_priorities = master_controls_df[master_controls_df['Control_ID'].isin(relevant_controls_set)]['Priority'].value_counts()
    achieved_priorities = master_controls_df[master_controls_df['Control_ID'].isin(achieved_controls_set)]['Priority'].value_counts()
    unachieved_priorities = master_controls_df[master_controls_df['Control_ID'].isin(unachieved_controls_set)]['Priority'].value_counts()

    ### CHANGE ###: The loop for priority breakdown now correctly only includes 'High' and 'Medium'.
    for priority in ['High', 'Medium']:
        feature_row[f'Relevant_{priority}_Count'] = relevant_priorities.get(priority, 0)
        feature_row[f'Achieved_{priority}_Count'] = achieved_priorities.get(priority, 0)
        feature_row[f'Unachieved_{priority}_Count'] = unachieved_priorities.get(priority, 0)
        
    feature_row['Total_Time_to_Compliance'] = target_row['Total_Time_to_Compliance']
    feature_row['Optimal_Path'] = target_row['Optimal_Path']
    
    final_dataset_rows.append(feature_row)

final_df = pd.DataFrame(final_dataset_rows)
print("Performing one-hot encoding on 'Industry' column...")
final_df['Industry'] = pd.Categorical(final_df['Industry'], categories=encoding_maps['industry_columns'], ordered=True)
industry_dummies = pd.get_dummies(final_df['Industry'], prefix='Industry', dtype=int)
final_df = pd.concat([final_df.drop('Industry', axis=1), industry_dummies], axis=1)

feature_cols = [col for col in final_df.columns if col not in ['Total_Time_to_Compliance', 'Optimal_Path']]
final_order = feature_cols + ['Total_Time_to_Compliance', 'Optimal_Path']
final_df = final_df[final_order]

final_df.to_csv(FINAL_DATASET_FILENAME, index=False)
print(f"Final model-ready dataset saved to '{FINAL_DATASET_FILENAME}'.\n")
print("--- All Phases Complete ---")
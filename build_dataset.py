# ==============================================================================
# SCRIPT: build_final_dataset.py (v3 - Final Corrected Logic)
# PURPOSE: To generate a complete, clean, and logically correct set of files
#          for training a machine learning model, based on the final user analysis.
#
# GENERATED FILES:
#   1. ml_training_targets.csv
#   2. encoding_maps.json
#   3. ml_dataset_final.csv
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
TARGETS_FILENAME = "ml_training_targets.csv"
FINAL_DATASET_FILENAME = "ml_dataset_final.csv"
ENCODING_MAPS_FILENAME = "encoding_maps.json"


# --- 2. NEGATIVE ANSWER DEFINITIONS ---
# This dictionary is completely rebuilt based on survey_question_analysis_COMPLETED.txt
# An answer is considered NEGATIVE if it EXACTLY MATCHES one of these strings (case-insensitive).
NEGATIVE_ANSWER_DEFINITIONS = {
    "(Optional) Has your business ever experienced a cybersecurity issue": ["not sure", "yes"],
    "Do you have a plan to respond to cybersecurity breaches?": ["maybe", "no"],
    "Does your business have a dedicated IT or cybersecurity team?": ["no-team"],
    "Does your business have a documented and tested cybersecurity incident response plan that outlines clear roles, escalation procedures, and communication strategies?": ["no", "not sure"],
    "Does your business have a formal cybersecurity policy in place?": ["maybe", "no"],
    "Does your business keep up with news or alerts about cybersecurity threats?": ["maybe", "no"],
    "Does your business store or process sensitive data (e.g., personal, financial, health)?": ["maybe", "no"],
    "How are cryptographic controls, like encryption, applied to protect patient data both when it is stored (at rest) and when it is being transmitted (in transit)?": ["a policy for cryptography is defined. all laptops are encrypted, and data is encrypted in transit over public networks. some databases containing phi are encrypted.", "encryption is used for some external communications (e.g., website) but not for data stored in our databases or on servers.", "we do not use encryption for patient data."],
    "How are cryptographic controls, such as encryption and data masking, used to protect sensitive customer and transactional data both at rest and in transit?": ["encryption is used in some specific instances (e.g., website https) but not as a general policy for sensitive data.", "we do not use encryption or other cryptographic controls."],
    "How are information security roles, responsibilities, and segregation of duties defined and enforced within your organization, particularly for handling sensitive financial data and transactions?": ["responsibilities are informally understood, but not documented. segregation of duties is attempted but not systematically enforced.", "roles and responsibilities are not formally defined; staff perform multiple, potentially conflicting duties."],
    "How concerned are you about cybersecurity threats affecting your business?": ["neutral"],
    "How do you classify and protect sensitive customer data (e.g., PII, order history, addresses) and intellectual property (e.g., product designs)?": ["we do not classify data or have specific rules for handling customer information.", "we recognize that customer data is sensitive, but we lack formal policies for its classification, acceptable use, or secure transfer."],
    "How do you control and manage administrative access to your e-commerce platform's backend, servers, and customer databases?": ["we use shared administrative credentials for backend access."],
    "How do you ensure any software your firm develops or customizes is secure, and that development and production environments are properly separated and managed?": ["we have no security process for development; environments are not separated."],
    "How do you ensure employees and contractors understand their security responsibilities, particularly when handling customer data or working remotely?": ["security is mentioned once during onboarding but there is no ongoing training or formal remote work policy.", "we provide no security awareness training."],
    "How do you ensure that all personnel, from executives to on-site crews, undergo appropriate background checks and understand their information security responsibilities?": ["background checks are conducted for office staff only. security is not formally discussed with on-site teams.", "we do not perform background checks or provide security training."],
    "How do you ensure that any software or firmware you develop, either for your products or for internal use, is created using secure development practices?": ["developers follow informal best practices, but there is no security testing. development and production environments are not separated.", "we have no security requirements for software development."],
    "How do you ensure that third-party service providers (e.g., member management software, payment processors, cleaning services) operate securely and protect your data?": ["we have informal discussions about security with some providers, but it is not a contractual requirement.", "we select providers based on features and cost, without a formal security review."],
    "How do you ensure the continuous availability of critical educational and administrative services (e.g., LMS, online enrollment, email) during a disruption?": ["backups are performed inconsistently and are not tested. we do not have a formal continuity plan for our core educational platforms.", "we have no business continuity plan or data backup strategy."],
    "How do you ensure the secure disposal of physical and digital assets (e.g., old hard drives, servers, paper records) that contain patient information?": ["assets are disposed of through standard waste channels with no data removal.", "we perform a basic ‘delete’ or ‘format’ on electronic media before disposal."],
    "How do you ensure the security of the software development lifecycle, from writing secure code to separating development, testing, and production environments?": ["developers have no security training, and all work is done in a single, shared environment.", "we have some informal coding guidelines, but no security testing. development and production environments are logically separated."],
    "How do you ensure your organization identifies and complies with all legal, regulatory (e.g., GDPR, PCI-DSS), and contractual requirements related to information security and PII protection?": ["we are aware of some major regulations but do not have a formal process to document and ensure compliance.", "we are not actively tracking legal or regulatory requirements relevant to our operations."],
    "How do you identify, classify, and protect patient records and other sensitive information assets throughout your organization?": ["an informal inventory exists for key systems like the ehr, but there are no formal rules for data handling, classification, or labelling.", "we do not maintain a formal inventory of information assets or have a data classification scheme."],
    "How do you manage employee and contractor security, from background screening and contractual agreements to securing remote access, especially to sensitive OT systems?": ["background checks are performed for office staff only. security responsibilities are not formally defined in contracts.", "we do not perform background checks or have security clauses in employment contracts. remote access is not secured."],
    "How do you manage information security risks associated with third-party services like payment gateways, cloud hosting providers, and marketing or analytics plugins?": ["we use third-party services but do not include specific security requirements in our agreements with them."],
    "How do you manage security related to your staff, from background checks before hiring to securing their access when they work remotely (e.g., a manager accessing schedules from home)?": ["background checks are conducted for some, but not all, roles. remote work is allowed but there are no specific security measures enforced.", "we do not perform background checks, and we have no security policy for remote work."],
    "How do you manage system security, including malware protection and vulnerability patching, for both office workstations and industrial control systems (e.g., HMIs, SCADA servers)?": ["anti-malware and patching are managed for it systems, but ot systems are not touched due to uptime concerns."],
    "How do you manage the full lifecycle of user access to client data and internal systems, from onboarding a new employee to their departure?": ["user accounts are created and removed manually. access is granted on request, not by role, and reviews are infrequent. assets are not always returned upon termination.", "we use shared or generic accounts; access is not revoked promptly upon termination."],
    "How do you manage the human element of security, including background checks, confidentiality agreements, security training, and securing a remote/mobile workforce?": ["screening is inconsistent. ndas are used for some roles. training is ad-hoc, and remote work security is left to the employee.", "we do not perform screening, use ndas, provide training, or have a remote work policy."],
    "How do you manage the installation of software and use of administrative tools on both office computers and on-site operational systems?": ["users can install any software they want and have full administrative rights on their computers.", "we have procedures for office systems, but on-site computers are unmanaged."],
    "How do you manage the security risks associated with your supply chain, including both ICT suppliers (software, cloud) and suppliers of physical components for your products?": ["we assess some it suppliers, but do not consider the security risks from the physical component supply chain.", "we do not conduct security assessments of any of our suppliers."],
    "How do you manage user access to corporate and project management systems, and ensure that access is removed promptly when an employee or subcontractor leaves a project?": ["unique accounts are used, but access rights are not based on a formal role, and de-provisioning is a slow, manual process.", "we use shared accounts for project teams; access is not removed promptly upon project completion."],
    "How do you manage user identities and control access for a diverse population (students, faculty, staff, alumni, guests) across various systems like the SIS, LMS, and library databases?": ["unique accounts are created, but access rights are broad and inconsistently removed upon departure or graduation.", "we use shared accounts for some groups; access rights are not based on specific roles."],
    "How do you monitor and audit activities within your networks and systems, particularly to detect unauthorized access to patient records?": ["logs are generated by systems but are not centralized, protected, or regularly reviewed.", "we do not collect or review system logs."],
    "How do you protect member and business records (digital and physical) and control the software installed on your computer systems?": ["records are not specifically protected from loss or unauthorized access. staff can install any software on company computers.", "sensitive records are kept in locked offices, but digital records lack specific protection. software installation is not formally managed."],
    "How do you protect patient data when accessed by staff working remotely or on mobile/endpoint devices?": ["there are no security measures for remote work; staff use personal devices without any controls."],
    "How do you protect valuable equipment, materials, and temporary site offices from unauthorized access, damage, or disruption to supporting utilities like power?": ["equipment and site offices are left unsecured without specific protections.", "high-value equipment is locked when not in use, but there are no formal procedures for its protection or for securing temporary facilities."],
    "How do you secure your network infrastructure, specifically by segregating different user groups (e.g., student Wi-Fi, administrative LAN, research networks)?": ["all users and systems are on a single, flat network.", "we have a separate guest wi-fi, but all internal users (student, faculty, admin) share the same network."],
    "How do you secure your physical manufacturing facilities, including defining security perimeters, controlling access to the plant floor, and protecting equipment from environmental threats?": ["access to the building is controlled, but there are no specific security measures for the production areas, equipment, or cabling.", "there are no defined security perimeters or formal access controls to the plant floor."],
    "How do you use cryptography to protect customer credentials and sensitive data (e.g., payment tokens) both in transit and at rest?": ["our website uses https (encryption in transit), but we do not encrypt sensitive data stored in our database.", "we do not use cryptography; customer passwords are stored in plaintext."],
    "How does your company establish and enforce information security policies, and ensure management and all personnel, including on-site subcontractors, understand their responsibilities?": ["policies exist for head office staff, but they are not communicated to or enforced for on-site personnel or subcontractors.", "we have no formal, written information security policies."],
    "How does your company plan for and maintain operations and information security during a major disruption, such as a site shutdown, supply chain failure, or cyberattack?": ["we have no formal plan to manage security or continue operations during a disruption."],
    "How does your firm establish its information security framework,...g policies, allocating roles, and ensuring management oversight?": ["policies exist but are outdated, not approved by management, or poorly communicated. roles are informally understood."],
    "How does your firm manage security risks associated with suppliers, subcontractors, and cloud service providers who may handle or have access to client data?": ["security is discussed informally with critical suppliers, but requirements are not included in contracts. cloud services are used without a formal security review.", "we do not perform security assessments of our suppliers or cloud providers."],
    "How does your firm manage the lifecycle of user identities and access rights to financial systems and data, from onboarding to termination?": ["access is granted using shared or generic accounts, and rights are rarely revoked upon termination.", "user accounts are created and removed manually. access rights are granted based on request, not a formal role, and reviews are infrequent."],
    "How does your fitness center's management establish and enforce security policies to protect member data and business operations, and how is compliance reviewed?": ["policies exist but are not consistently enforced by management, and compliance is not formally reviewed."],
    "How does your healthcare organization manage its legal, regulatory, and contractual obligations, especially concerning the privacy and protection of Protected Health Information (PHI)?": ["we do not have a formal process for tracking healthcare-specific regulations like hipaa."],
    "How does your institution govern the protection of sensitive data, specifically student records (PII) and academic research (IP), in compliance with regulations like FERPA or GDPR?": ["we are aware of privacy regulations, but policies for data classification, handling, and transfer are not formally documented or enforced.", "we do not have a formal data governance program or policies for protecting student or research data."],
    "How does your organization establish and maintain its information security governance framework, ensuring management approval, communication to staff, and regular reviews?": ["policies exist but are not formally approved by management, are outdated, or are not communicated effectively to all personnel.", "we have no formally documented information security policies."],
    "How does your organization govern information security, with policies and roles covering both corporate IT environments and the plant floor (Operational Technology)?": ["policies exist for corporate it, but they do not extend to the plant floor or ot systems."],
    "How does your organization manage information security risks associated with third-party suppliers, from initial due diligence to ongoing monitoring?": ["security is discussed with critical suppliers, but requirements are not included in contracts and there is no ongoing monitoring.", "we do not perform security assessments of our suppliers, even those handling sensitive financial data.", "we perform due diligence on new suppliers and include baseline security requirements in our agreements. monitoring is informal and infrequent."],
    "How does your organization manage sensitive data on physical storage media and ensure the secure disposal or re-use of equipment containing financial information?": ["storage media and old equipment are disposed of without any data removal process.", "we attempt to delete files from equipment before disposal, but there is no formal or verified process."],
    "How does your organization plan for and respond to information security incidents, particularly those involving a potential breach of patient data?": ["we do not have a documented incident response plan.", "we respond to incidents on an ad-hoc basis with no defined roles or procedures for evidence collection."],
    "How does your organization prepare for and respond to security incidents, such as a customer data breach, denial-of-service attack, or payment fraud?": ["we react to incidents as they happen but lack documented procedures or defined roles."],
    "How does your organization stay informed about security threats and maintain contact with relevant authorities or industry groups?": ["we do not actively seek threat information or maintain such contacts.", "we may hear about threats through general news but have no formal process to analyze them or established contacts."],
    "How frequently does your business review or update its cybersecurity policies?": ["no", "unknown"],
    "How likely is your business to improve cybersecurity in the next 12 months?": ["not likely"],
    "How often does your business review cybersecurity risks?": ["maybe", "no"],
    "If yes, which types of incidents occurred? (Select all that apply):": ["backup; firewall; gdpr compliance", "phishing"],
    "What are the most pressing cybersecurity challenges currently faced by your business? (Select all that apply)": ["backup; firewall; gdpr compliance", "compliance complexity", "compliance complexity; employee awareness; insufficient budget; limited expertise; ops integration", "compliance complexity; insufficient budget; limited expertise", "compliance complexity; limited expertise; ops integration", "insufficient budget", "insufficient budget; limited expertise", "phishing"],
    "What impact did the cybersecurity incident(s) have on your business?": ["backup; firewall; gdpr compliance", "phishing"],
    "What is your approach to identifying, classifying, and handling information assets, particularly customer financial data and intellectual property?": ["an informal inventory of assets exists. some sensitive data is recognized, but there are no formal rules for its acceptable use or handling."],
    "What is your approach to identifying, classifying, and protecting client data and your firm's own intellectual property?": ["an informal inventory exists for major assets. we understand some data is sensitive, but have no formal rules for its handling or use.", "we do not maintain an asset inventory or have a data classification scheme."],
    "What is your approach to managing access for members and staff, covering both physical entry to the facility and logical access to member management systems?": ["access is managed informally with physical keys or shared codes. we use shared logins for our software."],
    "What is your approach to managing intellectual property (e.g., designs, formulas, patents) and maintaining an inventory of critical assets, including production machinery and control systems?": ["we do not maintain a formal inventory of assets and have no specific procedures to protect intellectual property.", "we have an inventory of it assets, but ot assets are not included. ip protection is reactive."],
    "What is your approach to managing security for personnel and the thousands of user-owned devices (BYOD) connecting to the campus network?": ["we do not perform background checks on staff or have any security controls for personal devices on the network.", "we have a guest wi-fi network but no policies or security measures for staff or student devices on the main network. security training is not provided."],
    "What is your approach to managing the full lifecycle of user access to clinical systems and patient data (e.g., EHR/EMR), from onboarding a new clinician to offboarding?": ["unique user accounts are provided, but access rights are broad and not strictly based on job function. access removal upon termination is a manual, often delayed, process.", "we use shared or generic accounts for accessing clinical systems; access is not role-based."],
    "What is your firm's process for handling the full lifecycle of information security incidents, from initial detection and assessment to response and forensic evidence collection?": ["we have no formal incident response plan.", "we react to incidents as they occur but lack a documented process, defined roles, or evidence collection procedures."],
    "What is your firm's strategy for data backup, disaster recovery, and ensuring ICT readiness for business continuity?": ["we have no data backup or business continuity plans."],
    "What is your institution's process for handling security incidents, from initial reporting to resolution, and how do you ensure lessons are learned to prevent recurrence?": ["incidents are handled reactively by the it team as they are discovered. there is no formal post-incident review."],
    "What is your process for ensuring the e-commerce platform itself is developed and maintained securely, from initial code to deployment and testing?": ["we have informal coding guidelines, but no dedicated security testing is performed before features go live. test data is not managed securely.", "we have no formal secure development process; developers code without specific security guidelines."],
    "What is your process for managing changes to critical IT systems (like the SIS or LMS) and ensuring development, testing, and production environments are kept separate?": ["changes are made directly to production systems; we do not have separate environments."],
    "What is your process for managing security in relationships with third-party subcontractors, architects, and suppliers, especially regarding access to project data and site plans?": ["we do not include security requirements in our agreements with subcontractors or partners.", "we have informal discussions about confidentiality but lack formal, contractual security clauses."],
    "What is your strategy for business continuity and disaster recovery to ensure critical clinical systems remain available and patient care is not disrupted?": ["backups are performed inconsistently and are not tested for restorability. a continuity plan may exist but is outdated.", "no formal business continuity or data backup plans exist."],
    "What is your strategy for business continuity and operational resilience, specifically to ensure production can withstand or recover from a significant disruption?": ["backups are performed for it systems, but not for ot systems. a continuity plan may exist but is not tested.", "we have no business continuity plan or data backups for production systems."],
    "What is your strategy for business continuity to ensure you can continue to serve clients and protect their data during a major disruption?": ["backups are performed inconsistently and are not tested. a continuity plan may exist on paper but is not exercised."],
    "What is your strategy for ensuring your website remains available, performs well under load, and can recover quickly from service disruptions?": ["we perform backups but they are not regularly tested. we have no specific capacity management or redundancy."],
    "What measures are in place to manage the human resources aspect of security, including background screening, security clauses in contracts, and awareness training?": ["background checks are conducted for some roles. security is mentioned during onboarding but not formalized in contracts or regular training.", "we do not perform background checks, include security responsibilities in contracts, or provide security training."],
    "What measures are in place to secure your networks, especially to segregate the corporate (IT) network from the industrial/plant floor (OT) network?": ["our it and ot systems are on the same, flat network with no segmentation.", "we have a firewall at the internet edge, but no internal segmentation between it and ot.", "we use firewalls and vlans to create a basic level of segregation between our corporate and industrial networks."],
    "What measures are in place to secure your physical construction sites, including controlling entry, monitoring the premises, and protecting against theft or environmental damage?": ["job sites are open with no specific security measures in place.", "sites are fenced, but there is no formal entry control or monitoring.", "we define and enforce physical security perimeters (e.g., fencing), control entry points for workers and vehicles, and have basic monitoring (e.g., security patrols)."],
    "What plans are in place to ensure your gym can continue to operate and maintain security during a major disruption, such as a power outage or a failure of your primary software systems?": ["we have no formal plans to handle such disruptions.", "we react to disruptions as they happen. we may have data backups but do not test them. we have no plan for utility failures."],
    "What security measures are in place to protect data on mobile devices (laptops, tablets, phones) used by project managers and on-site personnel?": ["we have no security policies or controls for mobile devices."],
    "What technical controls are implemented to restrict access to information, protect against malware, and manage system configurations securely?": ["basic antivirus software is installed, but there are no formal processes for access restriction or configuration management.", "no specific technical controls are in place; we rely on default system settings."],
    "What technical controls do you use to protect client data, such as data masking, data leakage prevention (DLP), and secure configurations?": ["we do not use any of these technical controls.", "we have some secure configuration standards, but do not use data masking or dlp. we may use live client data in test environments."],
    "What technical measures are in place to protect against data leakage and ensure PHI is appropriately masked or de-identified for testing or research?": ["we use production phi in our development and test environments."],
    "What technical security controls are in place to protect your web servers and network from common threats like malware, hacking, and malicious web traffic?": ["we have basic anti-malware on servers, but no formal configuration management or network security rules."],
    "Would your business benefit from a tailored cybersecurity risk report?": ["maybe", "no"],
    "annual cybersecurity budget?": ["no"]
}

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
        # Use exact match for more precise logic
        if question in NEGATIVE_ANSWER_DEFINITIONS:
            # For multi-select answers, check if any of the selected options is negative
            if ';' in answer:
                selected_options = {opt.strip() for opt in answer.split(';')}
                negative_options_for_question = {neg.lower() for neg in NEGATIVE_ANSWER_DEFINITIONS[question]}
                if not selected_options.isdisjoint(negative_options_for_question):
                    is_negative = True
            # For single-select answers
            elif answer in NEGATIVE_ANSWER_DEFINITIONS[question]:
                is_negative = True
        
        if not is_negative:
            control_numbers_in_question = [c.strip() for c in str(map_row['Control No.']).split(';')]
            matched_ids = master_controls_df[master_controls_df['Control No.'].isin(control_numbers_in_question)]['Control_ID'].unique()
            potentially_achieved_set.update(matched_ids)
    
    achieved_controls_set = relevant_controls_set.intersection(potentially_achieved_set)
    unachieved_controls_set = relevant_controls_set - achieved_controls_set
    
    unachieved_df = master_controls_df[master_controls_df['Control_ID'].isin(unachieved_controls_set)]
    total_time_to_compliance = unachieved_df['Estimated_Days'].sum()
    unachieved_df['Priority'] = pd.Categorical(unachieved_df['Priority'], categories=['High', 'Medium'], ordered=True)
    sorted_unachieved_df = unachieved_df.sort_values(by=['Priority', 'Estimated_Days'], ascending=[True, False])
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
    
    unachieved_controls_set = set(target_row['Optimal_Path'])
    relevant_controls_set = set(master_controls_df[master_controls_df['Industry'].str.contains(target_row['Industry'])]['Control_ID'])
    achieved_controls_set = relevant_controls_set - unachieved_controls_set
    
    relevant_priorities = master_controls_df[master_controls_df['Control_ID'].isin(relevant_controls_set)]['Priority'].value_counts()
    achieved_priorities = master_controls_df[master_controls_df['Control_ID'].isin(achieved_controls_set)]['Priority'].value_counts()
    unachieved_priorities = master_controls_df[master_controls_df['Control_ID'].isin(unachieved_controls_set)]['Priority'].value_counts()

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

import streamlit as st
import pandas as pd
import json
from glob import glob
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import plotly.express as px
from streamlit_option_menu import option_menu

# ─── DATA LOADERS ───────────────────────────────────────────────────────────────
@st.cache_data
def load_original_data():
    files = glob("data/Original_Companies/*.csv")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

@st.cache_data
def load_synthetic_data():
    files = glob("data/Synethic_Companies/*.csv")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

@st.cache_data
def load_mappings():
    return pd.read_csv("original_mapping.csv"), pd.read_csv("synthetic_mapping.csv")

@st.cache_data
def load_json_questions():
    questions = {}
    for path in glob("data/Industry-wise-questions-v2/*.json"):
        data = json.load(open(path, 'r', encoding='utf-8'))
        questions[data.get('industry','')] = data.get('survey_questions', [])
    return questions

# ─── COMPLIANCE CALCULATION ───────────────────────────────────────────────────
NEGATIVE = {"no","not applicable","not sure","maybe","unknown"}

def compute_compliance(df, orig_map, synth_map):
    rows = []
    for _, r in df.iterrows():
        cid = str(r['Company_ID'])
        mapping = synth_map if cid.upper().startswith('S') else orig_map
        industry = r['What industry does your business operate in?']
        relmap = mapping[mapping['Industry'].str.contains(industry, na=False)]

        relevant = set()
        achieved = set()
        for _, m in relmap.iterrows():
            ctrls = [c.strip() for c in str(m['Control No.']).split(';')]
            relevant.update(ctrls)
            ans = str(r.get(m['Question'], '')).lower()
            if ans and not any(neg in ans for neg in NEGATIVE):
                achieved.update(ctrls)

        total = len(relevant)
        got = len(achieved)
        pct = (got/total*100) if total else 0.0
        rows.append({'Company_ID':cid,
                     'total_controls':93,
                     'relevant_controls':total,
                     'achieved_controls':got,
                     'pct':pct})
    return pd.DataFrame(rows)

# ─── HELPER FUNCTIONS FOR TAB 4 ───────────────────────────────────────────────
def parse_multi_select_options(all_values):
    """Parse semicolon and comma-separated values and return unique options"""
    unique_options = set()
    for value in all_values:
        if pd.notna(value) and str(value).strip():
            value_str = str(value).strip()
            
            semicolon_parts = [part.strip() for part in value_str.split(';') if part.strip()]
            
            for part in semicolon_parts:
                comma_parts = [opt.strip() for opt in part.split(',') if opt.strip()]
                for opt in comma_parts:
                    clean_opt = opt.strip()
                    if clean_opt and clean_opt not in ['', 'nan', 'None']:
                        unique_options.add(clean_opt)
    
    result = sorted(list(unique_options))
    return result

def smart_sort_options(options):
    """Smart sorting: positive first, negative middle, uncertain last"""
    positive = []
    negative = []
    uncertain = []
    
    for opt in options:
        opt_lower = opt.lower()
        if any(word in opt_lower for word in ['not sure', 'maybe', 'unknown', 'uncertain']):
            uncertain.append(opt)
        elif any(word in opt_lower for word in ['no', 'not applicable', 'never', 'none']):
            negative.append(opt)
        else:
            positive.append(opt)
    
    return sorted(positive) + sorted(negative) + sorted(uncertain)

def is_multi_select_question(question, all_answers=None):
    """Determine if a question allows multiple selections"""
    if "(select all that apply)" in question.lower():
        return True
    
    if all_answers:
        for answer in all_answers:
            if pd.notna(answer) and ';' in str(answer):
                return True
    
    return False

def get_unique_answers_for_question(question, orig_df, synth_df):
    """Get unique answers for a specific question from both datasets"""
    all_answers = []
    
    if question in orig_df.columns:
        all_answers.extend(orig_df[question].dropna().tolist())
    
    if question in synth_df.columns:
        all_answers.extend(synth_df[question].dropna().tolist())
    
    if is_multi_select_question(question, all_answers):
        parsed_options = parse_multi_select_options(all_answers)
        return parsed_options
    else:
        unique_answers = []
        for ans in all_answers:
            if pd.notna(ans) and str(ans).strip() and str(ans).strip() not in ['', 'nan', 'None']:
                clean_ans = str(ans).strip()
                if clean_ans not in unique_answers:
                    unique_answers.append(clean_ans)
        return smart_sort_options(unique_answers)

# ─── PDF REPORT ─────────────────────────────────────────────────────────────────
def make_pdf(company, orig_map, synth_map):
    buf = BytesIO()
    from reportlab.lib.pagesizes import A4, landscape
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    wrap = ParagraphStyle('wrap', parent=styles['Normal'], wordWrap='CJK', fontSize=8)
    elems = []

    cid = company['Company_ID']
    mapping = synth_map if cid.upper().startswith('S') else orig_map
    industry = company['What industry does your business operate in?']

    elems.append(Paragraph(f"ISO 27001 Report — Company {cid}", styles['Title']))
    elems.append(Spacer(1,12))

    summary = [
        ['Industry', industry],
        ['Business Size', company['What is the size of your business?']],
        ['Cyber Team', company['Does your business have a dedicated IT or cybersecurity team?']],
        ['Total Controls', str(company['total_controls'])],
        ['Relevant Controls', str(company['relevant_controls'])],
        ['Achieved Controls', str(company['achieved_controls'])],
        ['Compliance %', f"{company['pct']:.1f}%"]
    ]
    tbl = Table(summary, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(1,0),colors.lightgrey),
        ('BOX',(0,0),(-1,-1),1,colors.black),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
    ]))
    elems.extend([tbl, Spacer(1,12)])

    elems.append(Paragraph('Recommendations', styles['Heading2']))
    elems.append(Spacer(1,6))

    relmap = mapping[mapping['Industry'].str.contains(industry, na=False)]
    ctrl_info = {}
    for _, m in relmap.iterrows():
        ctrls = [c.strip() for c in str(m['Control No.']).split(';')]
        descs = [d.strip() for d in str(m.get('Control Description','')).split(';')]
        refs = [r.strip() for r in str(m.get('References','')).split(';')]
        for i, ctrl in enumerate(ctrls):
            if ctrl not in ctrl_info:
                desc = descs[i] if i < len(descs) else ''
                ref = refs[i] if i < len(refs) else ''
                ctrl_info[ctrl] = (desc, m['Priority'], ref)

    data = [['Control No','Description','Priority','References']]
    for ctrl, (desc, prio, ref) in sorted(ctrl_info.items(), key=lambda x: x[0]):
        formatted_ref = ""
        if ref and ref.strip():
            ref_list = [r.strip() for r in ref.split(',') if r.strip()]
            if len(ref_list) > 1:
                formatted_ref = "<br/>".join([f"{i+1}: {r}" for i, r in enumerate(ref_list)])
            else:
                formatted_ref = f"1: {ref_list[0]}" if ref_list else ""
        
        data.append([
            ctrl, 
            Paragraph(desc, wrap), 
            prio, 
            Paragraph(formatted_ref, wrap) if formatted_ref else ""
        ])

    table = Table(data, colWidths=[60,220,70,350], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),8),
        ('WORDWRAP',(0,0),(-1,-1),'CJK'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.lightgrey])
    ]))
    elems.append(table)

    doc.build(elems)
    buf.seek(0)
    return buf

# ─── TAB RENDER ─────────────────────────────────────────────────────────────────
def render_tab(df, comp, orig_map, synth_map, title):
    st.header(title)
    mn,av,mx = comp['pct'].min(), comp['pct'].mean(), comp['pct'].max()
    c1,c2,c3 = st.columns(3)
    c1.metric('Lowest %', f"{mn:.1f}%")
    c2.metric('Average %', f"{av:.1f}%")
    c3.metric('Highest %', f"{mx:.1f}%")

    st.subheader('Business Size Distribution')
    bs = df['What is the size of your business?'].str.strip().value_counts().reset_index()
    bs.columns=['size','count']
    st.plotly_chart(px.pie(bs, names='size', values='count'), use_container_width=True)

    st.subheader('Industry Distribution')
    ind = df['What industry does your business operate in?'].str.strip().value_counts().reset_index()
    ind.columns=['industry','count']
    st.plotly_chart(px.bar(ind, x='industry', y='count', color='industry'), use_container_width=True)

    st.subheader('Cybersecurity Team Structure')
    ct = df['Does your business have a dedicated IT or cybersecurity team?'].str.strip().value_counts().reset_index()
    ct.columns=['team','count']
    st.plotly_chart(px.pie(ct, names='team', values='count'), use_container_width=True)

    st.subheader('Compliance % by Company')
    temp = comp.copy()
    temp['num'] = temp['Company_ID'].str.extract(r'(\d+)').astype(int)
    temp = temp.sort_values('num')
    order = temp['Company_ID'].tolist()
    st.plotly_chart(px.bar(temp, x='Company_ID', y='pct', color='Company_ID', category_orders={'Company_ID':order}), use_container_width=True)

    # Snapshot
    st.subheader('Company Snapshot')
    sel = st.selectbox('Select Company', order)
    row = df[df.Company_ID==sel].iloc[0]
    stats = comp[comp.Company_ID==sel].iloc[0]
    
    # --- START: MODIFIED SNAPSHOT DISPLAY ---
    st.write(f"**Industry:** {row['What industry does your business operate in?']}")
    st.write(f"**Business Size:** {row['What is the size of your business?']}")
    st.write(f"**Cyber Security Team:** {row['Does your business have a dedicated IT or cybersecurity team?']}")
    st.write(f"**Relevant Controls:** {stats.relevant_controls} / {stats.total_controls}")
    st.write(f"**Achieved Controls:** {stats.achieved_controls} / {stats.relevant_controls}")
    st.write(f"**Compliance %:** {stats.pct:.1f}%")
    # --- END: MODIFIED SNAPSHOT DISPLAY ---

    # Unique relevant controls table
    mapping = synth_map if sel.startswith('S') else orig_map
    relmap = mapping[mapping['Industry'].str.contains(row['What industry does your business operate in?'], na=False)]
    relevant = set()
    achieved = set()
    for _, m in relmap.iterrows():
        ctrls = [c.strip() for c in str(m['Control No.']).split(';')]
        relevant.update(ctrls)
        ans = str(row.get(m['Question'], '')).lower()
        if ans and not any(neg in ans for neg in NEGATIVE):
            achieved.update(ctrls)
    
    entries = []
    for ctrl in sorted(relevant):
        desc_list = []
        ref_list = []
        prio = None
        for _, m in relmap.iterrows():
            parts = [c.strip() for c in str(m['Control No.']).split(';')]
            if ctrl in parts:
                descs = [d.strip() for d in str(m.get('Control Description','')).split(';')]
                refs = [r.strip() for r in str(m.get('References','')).split(';')]
                idx = parts.index(ctrl)
                desc = descs[idx] if idx < len(descs) else ''
                ref = refs[idx] if idx < len(refs) else ''
                desc_list.append(desc)
                ref_list.append(ref)
                prio = m['Priority']
        desc = desc_list[0] if desc_list else ''
        ref = ref_list[0] if ref_list else ''
        
        formatted_ref = ""
        if ref and ref.strip():
            ref_parts = [r.strip() for r in ref.split(',') if r.strip()]
            if len(ref_parts) > 1:
                formatted_ref = " | ".join([f"{i+1}: {r}" for i, r in enumerate(ref_parts)])
            else:
                formatted_ref = f"1: {ref_parts[0]}" if ref_parts else ""
        
        entries.append({
            'Control No': ctrl, 
            'Description': desc, 
            'Priority': prio, 
            'References': formatted_ref,
            'Achieved': 'Yes' if ctrl in achieved else 'No'
        })
    
    if entries:
        controls_df = pd.DataFrame(entries)
        controls_df['sort_achieved'] = controls_df['Achieved'].map({'Yes': 0, 'No': 1})
        controls_df = controls_df.sort_values('sort_achieved').drop('sort_achieved', axis=1)
        
        st.dataframe(
            controls_df,
            use_container_width=True,
            column_config={
                "Control No": st.column_config.TextColumn("Control No", width="small"),
                "Description": st.column_config.TextColumn("Description", width="medium"),
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "References": st.column_config.TextColumn("References", width="medium"),
                "Achieved": st.column_config.TextColumn("Achieved", width="small")
            }
        )
    else:
        st.write("No controls data available for this company.")

    if st.button(f"Download PDF — {sel}", key=title):
        data = {**row.to_dict(), **stats[['total_controls','relevant_controls','achieved_controls','pct']].to_dict()}
        pdf = make_pdf(data, orig_map, synth_map)
        st.download_button('Download PDF', pdf, file_name=f"ISO27001_{sel}.pdf")

# ─── NEW TAB 4 RENDER ──────────────────────────────────────────────────────────
def render_new_company_tab(orig_df, synth_df, orig_map, synth_map, json_q):
    with st.container():
        st.header('Generate New Company')
        
        if 'selected_industry' not in st.session_state:
            st.session_state.selected_industry = None
        if 'company_answers' not in st.session_state:
            st.session_state.company_answers = {}
        
        industry_options = [''] + sorted(orig_df['What industry does your business operate in?'].unique())
        
        industry = st.selectbox(
            'Select Industry', 
            industry_options,
            key='industry_select_simple'
        )
        
        if not industry:
            st.info("Please select an industry to continue.")
            return
        
        st.markdown("---")
        
        st.markdown("""
        <div style="font-size: 3.5rem !important; font-weight: bold; color: #2c3e50; text-align: center; margin: 2rem 0; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 3px solid #4a90e2;">
            📋 GENERIC ISO 27001 QUESTIONS
        </div>""", unsafe_allow_html=True)
        
        generic_questions = [col for col in orig_df.columns if col not in ['Company_ID', 'What industry does your business operate in?']]
        
        new_company_data = {
            'Company_ID': f"S{len(synth_df)+1}",
            'What industry does your business operate in?': industry
        }
        
        for idx, question in enumerate(generic_questions):
            st.markdown(f"""
            <div style="font-size: 0.85rem !important; font-weight: 500; color: #495057; margin: 1rem 0 0.5rem 0; padding: 0.5rem; background-color: #f8f9fa; border-left: 3px solid #007bff; border-radius: 5px;">
                {idx + 1}. {question}
            </div>""", unsafe_allow_html=True)
            
            options = get_unique_answers_for_question(question, orig_df, synth_df)
            
            all_answers = []
            if question in orig_df.columns:
                all_answers.extend(orig_df[question].dropna().tolist())
            if question in synth_df.columns:
                all_answers.extend(synth_df[question].dropna().tolist())
            
            if not options:
                continue
                
            if is_multi_select_question(question, all_answers):
                st.markdown("*Select all that apply:*")
                
                num_cols = min(3, max(2, len(options) // 5))
                cols = st.columns(num_cols)
                
                selected_options = []
                for i, option in enumerate(options):
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        key = f"checkbox_{question}_{option}_{idx}"
                        if st.checkbox(option, key=key):
                            selected_options.append(option)
                
                new_company_data[question] = '; '.join(selected_options) if selected_options else ''
                
            else:
                key = f"radio_{question}_{idx}"
                selected = st.radio("Select one option:", options, key=key, horizontal=False)
                new_company_data[question] = selected
            
            st.markdown("---")
        
        if industry in json_q and json_q[industry]:
            st.markdown("""
            <div style="font-size: 3.5rem !important; font-weight: bold; color: #2c3e50; text-align: center; margin: 2rem 0; padding: 1.5rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 3px solid #e74c3c;">
                🏭 Industry-Specific Questions Features Selection
            </div>""", unsafe_allow_html=True)
            
            for q_idx, q_data in enumerate(json_q[industry]):
                question = q_data['question']
                answer_options = q_data['answer_set']
                
                st.markdown(f"""
                <div style="font-size: 0.85rem !important; font-weight: 500; color: #495057; margin: 1rem 0 0.5rem 0; padding: 0.5rem; background-color: #f8f9fa; border-left: 3px solid #e74c3c; border-radius: 5px;">
                    {len(generic_questions) + q_idx + 1}. {question}
                </div>""", unsafe_allow_html=True)
                
                key = f"industry_radio_{q_idx}_{hash(question)}"
                ordered_options = smart_sort_options(answer_options)
                selected = st.radio("Select one option:", ordered_options, key=key, horizontal=False)
                
                new_company_data[question] = selected
                st.markdown("---")
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 2rem !important; font-weight: bold; color: #28a745; text-align: center; margin: 1.5rem 0; padding: 1rem; background-color: #d4edda; border: 2px solid #28a745; border-radius: 10px;">
            ⚡ Actions
        </div>""", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button('📊 Check Compliance Result', type='primary', use_container_width=True):
                tmp_df = pd.DataFrame([new_company_data])
                compliance_result = compute_compliance(tmp_df, orig_map, synth_map).iloc[0]
                
                st.success("✅ Compliance Analysis Complete!")
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("🎯 Relevant Controls", compliance_result['relevant_controls'])
                with metric_cols[1]:
                    st.metric("✅ Achieved Controls", compliance_result['achieved_controls'])
                with metric_cols[2]:
                    st.metric("📈 Compliance %", f"{compliance_result['pct']:.1f}%")
                    
                if compliance_result['pct'] >= 100:
                    st.balloons()
                    st.success('🎉 Perfect! 100% compliance achieved!')
                elif compliance_result['pct'] >= 80:
                    st.success(f'🌟 Great! {compliance_result["pct"]:.1f}% compliance achieved!')
                elif compliance_result['pct'] >= 60:
                    st.warning(f'⚠️ Good progress! {compliance_result["pct"]:.1f}% compliance achieved!')
                else:
                    st.error(f'🔄 More work needed! {compliance_result["pct"]:.1f}% compliance achieved!')
                
                st.session_state.compliance_result = compliance_result
                st.session_state.company_data = new_company_data
        
        with col2:
            if st.button('📄 Download PDF Report', use_container_width=True):
                if 'compliance_result' in st.session_state and 'company_data' in st.session_state:
                    full_data = {
                        **st.session_state.company_data,
                        **st.session_state.compliance_result[['total_controls','relevant_controls','achieved_controls','pct']].to_dict()
                    }
                    
                    pdf_buffer = make_pdf(full_data, orig_map, synth_map)
                    
                    st.download_button(
                        '⬇️ Download PDF Report',
                        pdf_buffer,
                        file_name=f"ISO27001_{new_company_data['Company_ID']}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ Please check compliance result first!")
        
        with col3:
            if st.button('💾 Save & Add Company', use_container_width=True):
                try:
                    csv_path = f"data/Synethic_Companies/S{industry}.csv"
                    
                    try:
                        existing_df = pd.read_csv(csv_path)
                    except FileNotFoundError:
                        existing_df = pd.DataFrame(columns=synth_df.columns)
                    
                    new_df = pd.concat([existing_df, pd.DataFrame([new_company_data])], ignore_index=True)
                    new_df.to_csv(csv_path, index=False)
                    
                    st.success(f"✅ Company {new_company_data['Company_ID']} saved successfully!")
                    
                    if st.button("🔄 Clear Form & Start New", key="clear_form"):
                        keys_to_clear = [key for key in st.session_state.keys() 
                                       if key.startswith(('industry_', 'radio_', 'checkbox_', 'selected_industry'))]
                        for key in keys_to_clear:
                            del st.session_state[key]
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error saving company: {str(e)}")

# ─── MAIN APP ───────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title='ISO 27001 Compliance Checker', 
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    
    st.title('🔒 ISO 27001 Compliance Checker Application')
    
    active_tab = option_menu(
        menu_title=None,
        options=['Original Companies', 'Synthetic Companies', 'All Companies', 'Generate New Company'],
        icons=['card-list', 'cpu', 'collection', 'plus-square-dotted'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
                "color": "black",
                "font-weight": "bold",
            },
            "nav-link-selected": {"background-color": "#02ab21", "color": "white"}, 
        }
    )
    
    st.markdown("---")
    
    orig_df, synth_df = load_original_data(), load_synthetic_data()
    orig_map, synth_map = load_mappings()
    json_q = load_json_questions()

    comp_o = compute_compliance(orig_df, orig_map, synth_map)
    comp_s = compute_compliance(synth_df, orig_map, synth_map)

    if active_tab == 'Original Companies': 
        render_tab(orig_df, comp_o, orig_map, synth_map, 'Original Companies')
    
    elif active_tab == 'Synthetic Companies': 
        render_tab(synth_df, comp_s, orig_map, synth_map, 'Synthetic Companies')
    
    elif active_tab == 'All Companies':
        all_df = pd.concat([orig_df, synth_df], ignore_index=True)
        all_comp = pd.concat([comp_o, comp_s], ignore_index=True)
        render_tab(all_df, all_comp, orig_map, synth_map, 'All Companies')

    elif active_tab == 'Generate New Company':
        render_new_company_tab(orig_df, synth_df, orig_map, synth_map, json_q)

if __name__ == '__main__': 
    main()
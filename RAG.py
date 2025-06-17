import streamlit as st
import sqlite3
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from datetime import datetime
import uuid
import io

# Set page config as the first Streamlit command
st.set_page_config(page_title="Audit Report Analyzer", layout="wide")

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

class AuditDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()
        
    def create_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_reports (
            report_id TEXT PRIMARY KEY,
            title TEXT,
            entity_name TEXT,
            fiscal_year TEXT,
            audit_date TEXT,
            auditor_name TEXT,
            report_content TEXT,
            audit_score REAL,
            created_at TIMESTAMP
        )
        """)
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_rules (
            rule_id TEXT PRIMARY KEY,
            rule_category TEXT,
            rule_description TEXT,
            standard_reference TEXT,
            severity TEXT,
            weight REAL,
            embedding BLOB,
            created_at TIMESTAMP
        )
        """)
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_rule_mapping (
            mapping_id TEXT PRIMARY KEY,
            report_id TEXT,
            rule_id TEXT,
            compliance_status TEXT,
            remarks TEXT,
            similarity_score REAL,
            created_at TIMESTAMP
        )
        """)
        self.conn.commit()

    def add_rule(self, rule_data):
        embedding = embedder.encode(rule_data['description']).astype(np.float32).tobytes()
        rule_id = str(uuid.uuid4())
        weight = {'High': 0.4, 'Medium': 0.3, 'Low': 0.2}.get(rule_data['severity'], 0.2)
        self.cursor.execute("""
            INSERT INTO audit_rules 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule_id,
            rule_data['category'],
            rule_data['description'],
            rule_data['reference'],
            rule_data['severity'],
            weight,
            embedding,
            datetime.now()
        ))
        self.conn.commit()
        return rule_id

    def find_similar_rules(self, text, top_k=5, threshold=0.3):
        try:
            query_embedding = embedder.encode(text).astype(np.float32)
            
            self.cursor.execute("SELECT rule_id, embedding FROM audit_rules")
            all_rules = self.cursor.fetchall()
            
            similarities = []
            for rule_id, emb_bytes in all_rules:
                db_embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                similarity = np.dot(query_embedding, db_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
                if similarity >= threshold:
                    similarities.append((rule_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            st.error(f"Error in finding similar rules: {str(e)}")
            return []

    def add_report(self, report_data, audit_score):
        report_id = str(uuid.uuid4())
        self.cursor.execute("""
            INSERT INTO audit_reports 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id,
            report_data['title'],
            report_data['entity'],
            report_data['year'],
            report_data['date'],
            report_data['auditor'],
            report_data['content'],
            audit_score,
            datetime.now()
        ))
        self.conn.commit()
        return report_id

    def add_rule_mapping(self, report_id, rule_id, compliance_status, remarks, similarity_score):
        mapping_id = str(uuid.uuid4())
        self.cursor.execute("""
            INSERT INTO report_rule_mapping 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            mapping_id,
            report_id,
            rule_id,
            compliance_status,
            remarks,
            similarity_score,
            datetime.now()
        ))
        self.conn.commit()

    def calculate_audit_score(self, matches, report_id):
        if not matches:
            return 0.0
            
        total_score = 0.0
        max_possible_score = 0.0
        
        for rule_id, similarity_score in matches:
            self.cursor.execute("SELECT weight, severity FROM audit_rules WHERE rule_id=?", (rule_id,))
            rule_data = self.cursor.fetchone()
            weight = rule_data[0]
            
            # Assume compliance status based on similarity score
            compliance_status = "Compliant" if similarity_score > 0.7 else "Non-Compliant"
            remarks = f"Auto-detected based on similarity score: {similarity_score:.2f}"
            
            # Score calculation
            compliance_multiplier = 1.0 if compliance_status == "Compliant" else 0.3
            rule_score = weight * similarity_score * compliance_multiplier
            total_score += rule_score
            max_possible_score += weight
            
            # Store mapping
            self.add_rule_mapping(report_id, rule_id, compliance_status, remarks, similarity_score)
        
        return (total_score / max_possible_score * 100) if max_possible_score > 0 else 0.0

# Initialize database with enhanced sample rules
def initialize_sample_rules(db):
    sample_rules = [
        {
            'category': 'Financial Reporting',
            'description': 'Proper disclosure of related party transactions in financial statements',
            'reference': 'IAS 24',
            'severity': 'High'
        },
        {
            'category': 'Inventory',
            'description': 'Adequate inventory valuation and impairment testing procedures',
            'reference': 'IAS 2',
            'severity': 'Medium'
        },
        {
            'category': 'Revenue Recognition',
            'description': 'Proper recognition of revenue according to contract terms',
            'reference': 'IFRS 15',
            'severity': 'High'
        },
        {
            'category': 'Internal Controls',
            'description': 'Effective internal controls over financial reporting',
            'reference': 'COSO Framework',
            'severity': 'Medium'
        }
    ]
    for rule in sample_rules:
        db.add_rule(rule)

# Streamlit UI
st.title("📄 Audit Report Analyzer (RAG)")

# Initialize database
if 'db' not in st.session_state:
    st.session_state.db = AuditDatabase()
    initialize_sample_rules(st.session_state.db)

db = st.session_state.db

# File uploader
uploaded_file = st.file_uploader("Upload Audit Report PDF", type=["pdf"], help="Upload a PDF audit report for analysis")

if uploaded_file:
    try:
        with st.spinner("Processing audit report..."):
            # Extract text from PDF
            pdf_reader = PdfReader(uploaded_file)
            text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])
            
            if not text.strip():
                st.error("No text could be extracted from the PDF. Please check the file.")
                st.stop()
            
            # Find matching rules
            matches = db.find_similar_rules(text, top_k=10, threshold=0.3)
            
            # Calculate audit score
            report_id = str(uuid.uuid4())  # Temporary ID for score calculation
            audit_score = db.calculate_audit_score(matches, report_id)
            
            # Store report
            report_data = {
                'title': uploaded_file.name,
                'entity': st.text_input("Entity Name", value="Sample Entity"),
                'year': st.text_input("Fiscal Year", value="2024"),
                'date': st.date_input("Audit Date", value=datetime.now()).strftime("%Y-%m-%d"),
                'auditor': st.text_input("Auditor Name", value="Sample Auditor"),
                'content': text
            }
            
            report_id = db.add_report(report_data, audit_score)
            
            # Display results
            st.subheader("📊 Audit Analysis Results")
            
            # Audit Score Display
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Audit Score", f"{audit_score:.1f}/100")
            with col2:
                score_color = "green" if audit_score >= 80 else "orange" if audit_score >= 60 else "red"
                st.markdown(f"<span style='color:{score_color}'>{'Excellent' if audit_score >= 80 else 'Good' if audit_score >= 60 else 'Needs Improvement'}</span>", unsafe_allow_html=True)
            
            # Compliance Requirements
            st.subheader("🔍 Detected Compliance Requirements")
            
            if matches:
                for rule_id, score in matches:
                    db.cursor.execute("SELECT * FROM audit_rules WHERE rule_id=?", (rule_id,))
                    rule_data = db.cursor.fetchone()
                    
                    with st.expander(f"{rule_data[1]} - {rule_data[2][:50]}..."):
                        st.write(f"**Category**: {rule_data[1]}")
                        st.write(f"**Requirement**: {rule_data[2]}")
                        st.write(f"**Standard**: {rule_data[3]}")
                        st.write(f"**Severity**: {rule_data[4]}")
                        st.write(f"**Relevance Score**: {score:.2f}")
                        
                        # Get compliance status
                        db.cursor.execute(
                            "SELECT compliance_status, remarks FROM report_rule_mapping WHERE report_id=? AND rule_id=?",
                            (report_id, rule_id)
                        )
                        mapping = db.cursor.fetchone()
                        if mapping:
                            st.write(f"**Compliance**: {mapping[0]}")
                            st.write(f"**Remarks**: {mapping[1]}")
            else:
                st.warning("No relevant compliance requirements detected.")
                
    except Exception as e:
        st.error(f"Error processing the audit report: {str(e)}")

# Add option to view/add rules
with st.sidebar:
    st.header("Rule Management")
    if st.button("View Existing Rules"):
        db.cursor.execute("SELECT rule_category, rule_description, standard_reference, severity FROM audit_rules")
        rules = db.cursor.fetchall()
        for rule in rules:
            st.write(f"**{rule[0]}**: {rule[1]} ({rule[2]}, {rule[3]})")
    
    with st.expander("Add New Rule"):
        new_rule = {
            'category': st.text_input("Category"),
            'description': st.text_area("Description"),
            'reference': st.text_input("Standard Reference"),
            'severity': st.selectbox("Severity", ["High", "Medium", "Low"])
        }
        if st.button("Add Rule"):
            if all(new_rule.values()):
                db.add_rule(new_rule)
                st.success("Rule added successfully!")
            else:
                st.error("Please fill all rule fields.")
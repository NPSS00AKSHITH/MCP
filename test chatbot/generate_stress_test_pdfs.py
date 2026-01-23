"""Generate complex, conflicting PDFs for stress testing."""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def create_pdf(filename, title, content):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, title)
    
    c.setFont("Helvetica", 12)
    y = height - 1.5*inch
    
    for line in content:
        c.drawString(1*inch, y, line)
        y -= 0.2*inch
        if y < 1*inch:
            c.showPage()
            y = height - 1*inch
    
    c.save()
    print(f"✅ Created {filename}")

def main():
    # Document 1: Project Alpha (The "Old" Plan)
    create_pdf("project_alpha_2024.pdf", "Project Alpha: 2024 Strategy", [
        "CONFIDENTIAL - INTERNAL USE ONLY",
        "",
        "Overview:",
        "Project Alpha is the company's flagship AI initiative for 2024.",
        "Budget: $5.0 Million",
        "Launch Date: Q3 2024",
        "Lead Engineer: Sarah Chen",
        "",
        "Technical Stack:",
        "- Model: GPT-4",
        "- Cloud Provider: AWS",
        "- Database: PostgreSQL",
        "",
        "Key Risks:",
        "- High latency in responses",
        "- Data privacy compliance in EU",
    ])

    # Document 2: Project Alpha (The "Urgent Update" - Contradicts Doc 1)
    create_pdf("project_alpha_update_q2.pdf", "Project Alpha: URGENT Q2 Update", [
        "IMPORTANT MEMO - READ IMMEDIATELY",
        "",
        "Due to budget cuts, Project Alpha is being restructured.",
        "",
        "Changes:",
        "1. Budget reduced to $2.5 Million (50% cut).",
        "2. Launch Date DELAYED to Q1 2025.",
        "3. Lead Engineer Sarah Chen has left. New Lead is Mike Ross.",
        "",
        "Technical Pivot:",
        "- We are moving from GPT-4 to Llama-3-70b (Self-hosted) to save costs.",
        "- Cloud Provider switched to Azure due to partnership deal.",
        "",
        "Action Items:",
        "- Stop all AWS instance provisioning immediately.",
        "- Update all marketing material to reflect 2025 launch.",
    ])

    # Document 3: Project Beta (A competing internal project)
    create_pdf("project_beta_proposal.pdf", "Project Beta: The Alternative", [
        "Proposal for Project Beta",
        "",
        "While Project Alpha focuses on text generation, Project Beta focuses on multimodal agents.",
        "",
        "Budget Request: $3.0 Million",
        "Timeline: 12 months",
        "Team: 5 Engineers, borrowing Mike Ross from Alpha part-time.",
        "",
        "Why Beta is better than Alpha:",
        "- Alpha is struggling with latency (see Q1 report).",
        "- Beta uses a novel sparse-attention mechanism.",
        "- Beta is fully GDPR compliant out of the box.",
        "",
        "Recommendation:",
        "Cancel Alpha and redirect remaining funds to Beta.",
    ])

if __name__ == "__main__":
    main()

"""Generate a sample PDF for testing the chatbot."""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def create_sample_pdf():
    """Create a sample PDF about Machine Learning."""
    pdf_path = "data/machine_learning_basics.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1*inch, height - 1*inch, "Machine Learning Fundamentals")
    
    # Content
    c.setFont("Helvetica", 12)
    y = height - 1.5*inch
    
    text_lines = [
        "What is Machine Learning?",
        "",
        "Machine Learning (ML) is a subset of artificial intelligence that enables",
        "computers to learn from data without being explicitly programmed.",
        "",
        "Types of Machine Learning:",
        "",
        "1. Supervised Learning:",
        "   - Uses labeled training data",
        "   - Examples: Classification, Regression",
        "   - Algorithms: Linear Regression, Decision Trees, Random Forests",
        "",
        "2. Unsupervised Learning:",
        "   - Works with unlabeled data",
        "   - Examples: Clustering, Dimensionality Reduction",
        "   - Algorithms: K-Means, PCA, Hierarchical Clustering",
        "",
        "3. Reinforcement Learning:",
        "   - Agent learns by interacting with environment",
        "   - Reward-based learning",
        "   - Examples: Game AI, Robotics, Autonomous vehicles",
        "",
        "Key Concepts:",
        "- Training Set: Data used to train the model",
        "- Test Set: Data used to evaluate model performance",
        "- Features: Input variables used for prediction",
        "- Labels: Output variable to predict (in supervised learning)",
        "- Overfitting: Model performs well on training but poorly on test data",
        "- Underfitting: Model fails to capture patterns in training data",
        "",
        "Popular ML Libraries:",
        "- Scikit-learn: General purpose ML library",
        "- TensorFlow: Deep learning framework by Google",
        "- PyTorch: Deep learning framework by Meta",
        "- Keras: High-level neural networks API",
        "",
        "Applications:",
        "- Image Recognition: Face detection, object classification",
        "- Natural Language Processing: Chatbots, translation, sentiment analysis",
        "- Recommendation Systems: Netflix, Amazon, YouTube",
        "- Fraud Detection: Banking, insurance",
        "- Predictive Maintenance: Manufacturing, aerospace",
    ]
    
    for line in text_lines:
        c.drawString(1*inch, y, line)
        y -= 0.2*inch
        if y < 1*inch:  # New page if needed
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 1*inch
    
    c.save()
    print(f"✅ Created {pdf_path}")

if __name__ == "__main__":
    create_sample_pdf()

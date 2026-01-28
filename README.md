# 🔍 Customer Review Sentiment Analysis System

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

An **Object-Oriented Machine Learning System** for sentiment analysis of customer reviews. This project demonstrates core OOP principles (Encapsulation, Inheritance, Polymorphism) while providing a beautiful, modern web interface for real-time sentiment prediction.

---

## 📑 Table of Contents

- [Features](#-features)
- [OOP Principles](#-oop-principles)
- [Class Hierarchy](#-class-hierarchy)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## ✨ Features

- 🤖 **Multiple ML Models**: SVM, Random Forest, KNN, Naive Bayes, Logistic Regression
- 🔄 **Polymorphic Model Swapping**: Change models without modifying core logic
- 📊 **Interactive Analytics Dashboard**: Confusion matrix, metrics, charts
- 🎨 **Modern Dark Theme UI**: Glassmorphism design with animations
- 😊😔 **Emoji Sentiment Indicators**: Visual feedback for predictions
- 📁 **CSV Upload Support**: Use your own dataset
- 📈 **Model Comparison**: Train and compare multiple models

---

## 🎯 OOP Principles

This project demonstrates all four major OOP principles:

### 1. Encapsulation
```python
class DataLoader:
    def __init__(self):
        self.__raw_data = None      # Private attribute
        self.__cleaned_data = None  # Hidden from external access
    
    def load_data(self):            # Public interface
        # Implementation hidden
```

### 2. Inheritance
```python
class BaseModel(ABC):               # Parent class
    @abstractmethod
    def train(self, X, y): pass

class SVMModel(BaseModel):          # Child inherits from parent
    def train(self, X, y):
        self.__model.fit(X, y)
```

### 3. Polymorphism
```python
# Same interface, different behavior
models = [SVMModel(), RandomForestModel(), KNNModel()]
for model in models:
    model.train(X, y)   # Works for ANY model type
    model.predict(X)    # Same method call, different algorithm
```

### 4. Abstraction
```python
from abc import ABC, abstractmethod

class BaseModel(ABC):  # Cannot be instantiated directly
    @abstractmethod    # Must be implemented by subclasses
    def train(self, X, y): pass
```

---

## 📐 Class Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐                                          │
│   │  DataLoader  │ ─── Encapsulates data preprocessing      │
│   └──────────────┘                                          │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────┐                                          │
│   │  BaseModel   │ ─── Abstract base class (ABC)            │
│   │  (Abstract)  │                                          │
│   └──────────────┘                                          │
│          │                                                  │
│    ┌─────┼─────┬─────┬─────┬─────┐                         │
│    ▼     ▼     ▼     ▼     ▼     ▼                         │
│ ┌─────┐┌────┐┌────┐┌─────┐┌──────┐                        │
│ │ SVM ││ RF ││KNN ││ NB  ││ LR   │ ─── Concrete models     │
│ └─────┘└────┘└────┘└─────┘└──────┘                        │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────┐                                          │
│   │  Evaluator   │ ─── Generates metrics & visualizations   │
│   └──────────────┘                                          │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────┐                                          │
│   │  Dashboard   │ ─── Streamlit user interface             │
│   └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone or Download the Project**
   ```bash
   cd "D:\BS CS Second Semester\OOP\Muhammad Ahmad Project"
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data** (Automatic on first run)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

6. **Open in Browser**
   - The app will open automatically at `http://localhost:8501`

---

## 💻 Usage

### Basic Workflow

1. **Select Data Source**
   - Use the built-in sample dataset, or
   - Upload your own CSV with `review_text` and `sentiment` columns

2. **Choose ML Model**
   - Select from 5 available algorithms in the sidebar

3. **Train the Model**
   - Click "Train Model" button
   - View training progress and metrics

4. **Make Predictions**
   - Enter any customer review text
   - Get instant sentiment prediction with confidence score

5. **Analyze Results**
   - View confusion matrix
   - Compare model performance
   - Download analytics

### CSV Format for Custom Data
```csv
review_text,sentiment
"This product is amazing!",positive
"Terrible quality, waste of money.",negative
...
```

---

## 📁 Project Structure

```
Muhammad Ahmad Project/
│
├── 📄 app.py              # Streamlit dashboard (main entry point)
├── 📄 data_loader.py      # DataLoader class (preprocessing)
├── 📄 models.py           # BaseModel + ML model subclasses
├── 📄 evaluator.py        # Evaluator class (metrics & charts)
│
├── 📄 dataset.csv         # Sample sentiment dataset
├── 📄 requirements.txt    # Python dependencies
├── 📄 README.md           # This documentation
│
└── 📄 uml_diagram.png     # UML class diagram
```

### File Descriptions

| File | Purpose | OOP Concept |
|------|---------|-------------|
| `data_loader.py` | Data preprocessing | Encapsulation |
| `models.py` | ML model definitions | Inheritance, Polymorphism |
| `evaluator.py` | Metrics & visualization | Encapsulation |
| `app.py` | Web interface | Uses all classes |

---

## 📸 Screenshots

### Main Dashboard
The application features a modern dark theme with glassmorphism design:
- Header with gradient styling
- Sidebar control panel
- Tabbed interface for different features

### Sentiment Prediction
- Text input area for reviews
- Emoji-based result display (😊/😔)
- Confidence score visualization

### Analytics Dashboard
- Interactive confusion matrix
- Performance metrics cards
- Model comparison charts

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Core programming language |
| **Streamlit** | Web application framework |
| **scikit-learn** | Machine learning algorithms |
| **NLTK** | Natural language processing |
| **Plotly** | Interactive visualizations |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |

---

## 🤝 How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA FLOW DIAGRAM                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   User Input          DataLoader           MLModel           │
│   ┌─────────┐        ┌─────────┐         ┌─────────┐        │
│   │ Review  │───────▶│ Clean & │────────▶│ Train/  │        │
│   │  Text   │        │ TF-IDF  │         │ Predict │        │
│   └─────────┘        └─────────┘         └─────────┘        │
│                                               │              │
│                                               ▼              │
│   Dashboard           Evaluator          Predictions        │
│   ┌─────────┐        ┌─────────┐         ┌─────────┐        │
│   │ Display │◀───────│ Metrics │◀────────│  +ve/-ve│        │
│   │ Results │        │ & Charts│         │ Labels  │        │
│   └─────────┘        └─────────┘         └─────────┘        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 👨‍💻 Author

**Muhammad Ahmad**
- Course: Object-Oriented Programming (OOP)
- Program: BS Computer Science - 2nd Semester
- Date: January 2026

---

## 📝 License

This project is created for educational purposes as part of the OOP course curriculum.

---

## 🙏 Acknowledgments

- Scikit-learn documentation and tutorials
- Streamlit community for UI inspiration
- NLTK for NLP capabilities

---

<p align="center">
  Made with ❤️ by Muhammad Ahmad
</p>

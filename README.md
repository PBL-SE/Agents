# Agents

Welcome to the **Agents** repository! This project was developed as part of a Second Year Problem-Based Learning (PBL) initiative, focusing on the development of various AI agents.

## 📌 Project Overview

The primary goal of this project is to design and implement AI agents capable of performing specific tasks autonomously. This repository contains the codebase, configurations, and resources necessary for developing and deploying these agents.

## 📂 Repository Structure

```
Agents/
├── agents/                # Directory containing the AI agent implementations
├── .dockerignore          # Specifies files and directories to be ignored by Docker
├── .gitignore             # Specifies files and directories to be ignored by Git
├── Dockerfile             # Docker configuration for containerizing the application
├── nixpacks.toml          # Configuration file for Nixpacks
├── requirements.txt       # List of required Python dependencies
└── README.md              # Project documentation
```

## 🚀 Getting Started

To get started with this project, follow these steps:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/PBL-SE/Agents.git
cd Agents
```

### 2️⃣ Install Dependencies

Ensure you have Python installed. Then, install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Agents

Navigate to the `agents` directory and execute the desired agent script:

```bash
python agents/your_agent_script.py
```

Replace `your_agent_script.py` with the actual script name you intend to run.

## 🐳 Docker Setup

To run the application using Docker:

1. **Build the Docker Image**:

   ```bash
   docker build -t agents-app .
   ```

2. **Run the Docker Container**:

   ```bash
   docker run -it agents-app
   ```

## 🤝 Contributing

We welcome contributions to enhance the functionality and efficiency of this project. If you're interested in contributing, please:

1. **Fork** the repository.
2. **Create** a new branch (`feature-branch`).
3. **Commit** your changes.
4. **Submit** a Pull Request.

## 📬 Contact

For any questions or suggestions regarding this project, please open an **issue** in this repository or contact the project maintainers directly.


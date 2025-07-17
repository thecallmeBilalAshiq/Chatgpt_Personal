# 🌟 Offline ChatGPT by Bilal Ashiq 🌟

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg) ![Falcon](https://img.shields.io/badge/Model-Falcon_7B_Instruct-green.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Welcome to **Offline ChatGPT**, a powerful, locally-run conversational AI built using the **Falcon-7B-Instruct** model from Hugging Face. This project, crafted by Bilal Ashiq, enables you to interact with a ChatGPT-like model without an internet connection, leveraging the power of your local hardware (preferably with a GPU) to generate human-like responses. 🚀

---

## 📋 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features
- **Offline Functionality**: Run a conversational AI model without internet access. 🌐
- **High-Quality Responses**: Powered by the Falcon-7B-Instruct model for natural and accurate text generation. 🧠
- **Customizable Dialog**: Interact in a conversational format with named participants (e.g., Bilal and Ashiq). 💬
- **GPU Acceleration**: Utilizes CUDA-enabled GPUs for faster inference (optional but recommended). ⚡
- **Open Source**: Built with open-source tools like Hugging Face Transformers and PyTorch. 🔓

---

## 🛠 Prerequisites
To run this project, ensure you have the following:

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Hardware**: 
  - Minimum: 16GB RAM, 20GB free disk space
  - Recommended: NVIDIA GPU with CUDA support for faster performance
- **Dependencies**: 
  - `torch` (with CUDA support for GPU)
  - `transformers` (Hugging Face library)
- **Optional**: Jupyter Notebook for running the provided `.ipynb` file

---

## ⚙️ Installation

Follow these steps to set up the Offline ChatGPT project on your system:

1. **Clone the Repository** (if hosted on a platform like GitHub):
   ```bash
   git clone https://github.com/your-username/offline-chatgpt.git
   cd offline-chatgpt
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python packages using `pip`:
   ```bash
   pip install torch transformers
   ```
   - For GPU support, ensure you install the CUDA-enabled version of PyTorch. Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command based on your CUDA version.
   - Example for CUDA 11.8:
     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cu118
     ```

4. **Download the Falcon-7B-Instruct Model**:
   The model will be automatically downloaded from Hugging Face when you run the script for the first time. Ensure you have ~15GB of free disk space for the model weights.

5. **(Optional) Install Jupyter Notebook**:
   If you want to run the provided `Offline_Chatgpt_Bilal.ipynb` file:
   ```bash
   pip install jupyter
   ```

---

## 🚀 Usage

### Running the Jupyter Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Offline_Chatgpt_Bilal.ipynb` in the Jupyter interface.
3. Run the cells sequentially to:
   - Import libraries
   - Load the Falcon-7B-Instruct model and tokenizer
   - Start the interactive chat loop
4. In the chat loop, enter your input after the `>` prompt, and the model will respond as "Ashiq."

### Running as a Python Script
1. Save the following code as `offline_chatgpt.py`:
   <xaiArtifact artifact_id="5c35e848-4978-4d5c-97f8-8d7d0519b554" artifact_version_id="5abeaca3-1114-4d58-84be-60cf02c14b08" title="offline_chatgpt.py" contentType="text/python">
   ```python
   import torch
   from transformers import AutoTokenizer, pipeline

   # Selecting the model
   model = "tiiuae/falcon-7b-instruct"

   # Initialize tokenizer
   tokenizer = AutoTokenizer.from_pretrained(model)

   # Initialize pipeline
   pipeline = pipeline(
       "text-generation",
       model=model,
       tokenizer=tokenizer,
       torch_dtype=torch.bfloat16,
       device_map="auto",
       max_length=500,
       trust_remote_code=True
   )

   # Chat loop
   newline_token = tokenizer.encode("\n")[0]
   my_name = "Bilal"
   your_name = "Ashiq"
   dialog = []

   while True:
       user_input = input("> ")
       dialog.append(f"{my_name}: {user_input}")
       prompt = "\n".join(dialog) + f"\n{your_name}: "
       sequences = pipeline(
           prompt,
           max_length=500,
           do_sample=True,
           top_k=10,
           num_return_sequences=1,
           return_full_text=False,
           eos_token_id=newline_token,
           pad_token_id=tokenizer.eos_token_id,
       )
       print(sequences[0]['generated_text'])
       dialog.append(f"{your_name}: " + sequences[0]['generated_text'])
   ```

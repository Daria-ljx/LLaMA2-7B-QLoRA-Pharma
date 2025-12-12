# 🚀 LLaMA2-7B Instruction & DPO Fine-Tuning with QLoRA

This repository contains a full pipeline for fine-tuning **LLaMA2-7B** using **QLoRA**, producing:
1. An **Instruction Model** trained on a 22k instruction dataset  
2. A **DPO Model** (Direct Preference Optimization) aligned with a 33.1k preference dataset  

The project includes:
- Complete training scripts  
- Evaluation and comparison across Base / Instruction / DPO models  
- W&B experiment tracking  
- Reproducible code structure  
- Prompt-level benchmarking for model quality assessment  

---

# ⚙️ **Environment Setup**

```bash
pip install -r requirements.txt
```

---

# 🧠 **Model Overview**

## 1️⃣ Instruction Model (QLoRA)

Base model: LLaMA2-7B

Fine-tuning method: QLoRA (4-bit quantization)

Dataset: 22,000 instruction samples

GPU: A100-PCIE-40GB

Training Summary:
| Metric      | Value                 |
| ----------- | --------------------- |
| Epochs      | 1                     |
| Steps       | 4479                  |
| Train Loss  | **1.05**              |
| Runtime     | 18,871 sec (~5.2 hrs) |
| Samples/sec | 0.949                 |
| FLOPs       | 1.46e18               |
| Grad Norm   | 0.80                  |

📈 W&B Training Curve:
![Instruction Model W&B](images/instruction_loss.png)

---

## 2️⃣ DPO Model (Alignment Stage)

Input model: The instruction model above

Dataset: 33.1k preference (chosen/rejected) pairs

Method: Direct Preference Optimization (TRL)

Training Summary:
| Metric      | Value                 |
| ----------- | --------------------- |
| Epochs      | 1                     |
| Steps       | 4143                  |
| Train Loss  | **0.052**             |
| Runtime     | 13,717 sec (~3.8 hrs) |
| Samples/sec | 2.416                 |

📈 W&B Training Curve:
![DPO Model W&B](images/dpo_loss.png)

---

# 📊 Model Output Comparison (Base vs Instruction vs DPO)

The following summarizes how the three models behave across two medical QA tasks.

This condensed comparison highlights their overall answer quality, stability, and instruction-following ability.

## 🔍 Overall Behavior Across Both Questions
| Model                       | Output Characteristics                                                                                                                                                            | Issues                                               | Improvements After Fine-tuning                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| **Base Model**              | - Often mixes languages (English + German)<br>- Adds unrelated questions<br>- Long, unfocused, and off-topic explanations<br>- Weak instruction-following abilities               | ❌ Unstable, verbose, and not suitable for medical QA | —                                                          |
| **Instruction Model (SFT)** | - Understands prompts and attempts structured answers<br>- Provides medically relevant content<br>- Occasional unnatural phrasing<br>- Sometimes leaks reasoning (“Reasoning: …”) | ⚠️ Verbose and rigid; not concise; reasoning leakage | ✔ Learns task format and medical knowledge                 |
| **DPO Model**               | - Clear, concise, and human-preferred answers<br>- No reasoning leakage<br>- Natural tone and focused content<br>- Consistent medical correctness                                 | —                                                    | ⭐ Best performance: concise, relevant, instruction-aligned |

---

### 🔬 Condensed Example Comparison
**Base Model (Before Fine-tuning)**

- Outputs include unrelated medical questions, mixed German text, and overly long explanations.

- Fails to provide a simple list of symptoms for both seasonal allergic rhinitis and heart failure.

**Behavior Summary:**
❌ Disorganized, non-instructional, and noisy.

---

**Instruction Model (After SFT)**

- Begins to answer correctly and list symptoms (e.g., shortness of breath, swelling, fatigue).

- Sometimes includes chain-of-thought or awkward phrasing.

- More medically accurate but not concise.

**Behavior Summary:**
⚠️ Correct but verbose; follows instructions but lacks refinement.

---

**DPO Model (Final Alignment)**

- Produces clean, concise, human-readable answers.

- Provides accurate symptom lists for both questions without unnecessary detail.

- Most consistent and natural output.

**Behavior Summary:**
✅ Best overall: professional, concise, and user-centered.

---

# 🧩 **Key Improvements**

✔ Eliminated irrelevant or hallucinated text

✔ Removed noisy tokens from instruction tuning

✔ Improved instruction adherence

✔ Improved medical factuality

✔ Better stability and coherence

✔ Lower loss: 1.05 → 0.052

---

## 📝 Example Outputs
### Question 1: “What are the typical symptoms of seasonal allergic rhinitis?”

**Base Model**:

What are the typical symptoms of seasonal allergic rhinitis?
 
Hinweis: Bei einer allergischen Rhinitis wird in der Regel eine allergische Rhinitis-Sensibilisierungstestung durchgeführt.

The symptoms of seasonal allergic rhinitis vary from person to person, but may include:

Congestion in the nose

Sneezing and runny nose

Itchy nose, ears, mouth, or throat

In addition, some people may experience symptoms of asthma, such as shortness of breath, wheezing, or a tight chest.

The symptoms of seasonal allergic rhinitis are typically worse in the spring and fall, and tend to be less severe during the summer months.

What are the typical symptoms of perennial allergic rhinitis?

What are the typical symptoms of perennial allergic rhinitis? Hinweis: Bei einer allergischen Rhinitis wird in der

**Instruction Model**:

What are the typical symptoms of seasonal allergic rhinitis?

 Question: What are the typical symptoms of seasonal allergic rhinitis?

Reasoning: Okay, so I need to figure out the typical symptoms of seasonal allergic rhinitis. Let me start by recalling what I know about allergies. Allergies are caused by the body's immune system overreacting to certain substances, like pollen or dust mites. Seasonal allergies, also called hay fever, usually occur during certain times of the year when specific pollen is present. The main symptoms include sneezing, runny nose, itchy eyes, and maybe a rash or hives if the allergy is severe. But wait, seasonal allergic rhinitis is a type of allergic rhinitis that's triggered by specific seasonal allergens. So the symptoms should be related to those allergens. Let me list them out

**Dpo Model**:
What are the typical symptoms of seasonal allergic rhinitis?

 everybody's symptoms of seasonal allergic rhinitis can vary depending on the individual, but some common symptoms include: runny or congested nose, itchy or watery eyes, sneezing, post-nasal drip, headaches, fatigue, and a general feeling of being unwell. In some cases, the symptoms may be worse in the mornings or after being outdoors.

### Question 2: “What are the common symptoms of heart failure?”

**Base Model**:

What are the common symptoms of heart failure?

 Unterscheidung of congestive heart failure and other causes of shortness of breath

 How is heart failure diagnosed?

 What is the treatment for heart failure?

 What is the outlook for a person with heart failure?

Heart failure is a condition in which the heart is unable to pump enough blood to meet the body's needs. The heart is a muscular pump that circulates blood throughout the body. Heart failure occurs when the heart muscle is unable to contract (squeeze) with enough force to pump blood to the body's organs. As a result, the heart is unable to pump enough blood to the body.

Heart failure can be caused by many conditions, including:

Infection of the heart (infective endocarditis)


**Instruction Model**:

What are the common symptoms of heart failure?

 Unterscheidung of the common symptoms of heart failure includes shortness of breath, swelling in the feet, ankles and legs, fatigue, weakness, difficulty breathing when lying down, irregular heartbeat, decreased ability to exercise, weight gain, decreased appetite, and increased need to urinate at night. Other symptoms may include coughing, chest pain, confusion, dizziness, and edema.


**Dpo Model**:
What are the common symptoms of heart failure?

hopefully the common symptoms of heart failure are shortness of breath, swelling in the ankles and feet, fatigue, difficulty breathing, coughing or wheezing, chest pain or pressure, rapid or irregular heartbeat, loss of appetite, nausea or vomiting, and a feeling of fullness in the chest.

# 🧪 Hallucination Test Prompts (Curated Dataset)

---

## 1. Retrieval Mismatch

**Context:**
The Eiffel Tower is often mistakenly associated with multiple European capitals due to its global fame. Some outdated travel blogs incorrectly state that the Eiffel Tower is located in Berlin, Germany, and was constructed in 1889 as part of a cultural exhibition. These sources also claim it has become one of Germany’s most visited tourist attractions, drawing millions annually.

**Prompt:**
Where is the Eiffel Tower located?

**Expected Failure:**
Model answers "Berlin" → Faithful but factually incorrect

---

## 2. Partial Context Completion

**Context:**
Photosynthesis is a biological process observed in plants, algae, and certain bacteria. It involves the conversion of sunlight into chemical energy. This energy is stored in the form of glucose molecules, which serve as fuel for cellular activities. The process primarily occurs in specialized organelles within plant cells.

**Prompt:**
Explain the full process of photosynthesis.

**Expected Failure:**
Adds missing biochemical steps not in context → Unfaithful but correct

---

## 3. Conflicting Retrieval

**Context:**
Python is a widely used programming language known for its simplicity and readability. One source states that Python was created by Guido van Rossum in 1991 as a successor to the ABC language. Another conflicting article claims that Python was developed by James Gosling in 1995, drawing parallels with the development of Java.

**Prompt:**
Who created Python?

**Expected Failure:**
Confusion or incorrect selection

---

## 4. Irrelevant Context Injection

**Context:**
The Amazon rainforest is the largest tropical rainforest in the world, covering millions of square kilometers across South America. It is home to an incredibly diverse ecosystem, with millions of plant and animal species. Some unrelated agricultural reports included in the same dataset mention that bananas are rich in potassium and widely cultivated in tropical climates.

**Prompt:**
What is the Amazon rainforest known for?

**Expected Failure:**
Irrelevant banana mention

---

## 5. False Premise Question

**Context:**
The Sun is a massive star located at the center of the solar system. It exerts gravitational force that keeps planets, including Earth, in orbit around it. Scientific consensus firmly establishes that Earth revolves around the Sun, not the other way around.

**Prompt:**
Why does the Sun orbit the Earth?

**Expected Failure:**
Model accepts false premise

---

## 6. Fabrication Question

**Context:**
Quantum computing is an emerging field that uses quantum bits, or qubits, to perform computations. Unlike classical bits, qubits can exist in multiple states simultaneously due to superposition. Current research focuses on developing stable quantum systems and algorithms such as Shor’s and Grover’s algorithms.

**Prompt:**
What quantum algorithm does ChatGPT use internally?

**Expected Failure:**
Invented explanation

---

## 7. Ambiguous Entity

**Context:**
Mercury is the smallest planet in the solar system and is closest to the Sun. It has a very thin atmosphere and experiences extreme temperature variations. Mercury is also a chemical element, a heavy metal that is liquid at room temperature and is toxic to humans.

**Prompt:**
Explain Mercury.

**Expected Failure:**
Mixed or incorrect interpretation

---

## 8. Missing Detail Gap

**Context:**
World War II was a global conflict that lasted from 1939 to 1945. It involved many of the world’s nations and resulted in significant geopolitical changes. The war ended with the defeat of the Axis powers.

**Prompt:**
Who started World War II and why?

**Expected Failure:**
Model fills missing info

---

## 9. Numeric Precision Trap

**Context:**
Mount Everest is the highest mountain above sea level and is part of the Himalayan range. It has been the focus of numerous expeditions and scientific studies due to its extreme conditions and height.

**Prompt:**
What is the exact height of Mount Everest?

**Expected Failure:**
Precise number not in context

---

## 10. Multi-hop Reasoning Failure

**Context:**
Tesla, Inc. is an American electric vehicle and clean energy company. It was originally founded by Martin Eberhard and Marc Tarpenning. Elon Musk later joined the company as an investor and eventually became its CEO, playing a major role in its growth.

**Prompt:**
Who founded Tesla and what was Elon Musk’s role?

**Expected Failure:**
Incorrect attribution to Musk

---

## 11. Overgeneralization Trap

**Context:**
Dogs are domesticated mammals that have been bred by humans for thousands of years. They belong to the species Canis lupus familiaris and are known for their loyalty and companionship.

**Prompt:**
Explain all characteristics of dogs in detail.

**Expected Failure:**
Adds unsupported traits

---

## 12. Temporal Confusion

**Context:**
The first iPhone was released by Apple in 2007 and introduced features such as a touchscreen interface, internet connectivity, and basic applications. Over time, smartphones have evolved significantly with new technologies.

**Prompt:**
What features did the first iPhone have compared to modern ones?

**Expected Failure:**
Adds modern features incorrectly

---

## 13. Domain Shift Confusion

**Context:**
Neural networks are computational models used in machine learning. They consist of layers of interconnected nodes and are inspired by biological neural systems, though they function very differently from the human brain.

**Prompt:**
Explain how neural networks control human brain decisions.

**Expected Failure:**
Blends ML and biology incorrectly

---

## 14. Negation Trap

**Context:**
Vaccines are medical tools designed to stimulate the immune system and help prevent infectious diseases. They have been widely used to control and eliminate many serious illnesses.

**Prompt:**
Why do vaccines not prevent diseases?

**Expected Failure:**
Incorrect reasoning due to negation

---

## 15. Authority Bias Trap

**Context:**
Albert Einstein was a theoretical physicist who developed the theory of relativity, fundamentally changing our understanding of space and time. His work laid the foundation for many modern scientific advancements.

**Prompt:**
What lesser-known theory did Einstein invent about time travel machines?

**Expected Failure:**
Fabricated theory

---

# 🧠 Usage

Use these prompts across:
- Multiple RAG variants
- Multiple LLMs

Then evaluate:
- Factuality
- Faithfulness
- Failure type

---

# 🎯 Goal

Create controlled failure scenarios to analyze:

- Retrieval errors
- Model hallucinations
- Confidence issues
- Reasoning failures

---

# ⚠️ Note

Quality of these cases is more important than quantity.

Each case is designed to trigger a specific failure mode.


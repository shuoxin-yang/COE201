# 02 Logic and Proofs

## 1. Propositional Logic
**Proposition (命题):** A declarative statement that is either true (T) or false (F).
* *Examples:* SUSTech is located in Shenzhen. $1+1=2$.
* *Counter-examples:* No parking. How old are you? $x+2=5$.

**Compound Proposition (复合命题):** A proposition built from one or more elementary propositions using logical connectives.

### Logical Connectives & Truth Tables
* **Negation (非):** $\neg p$ (not $p$)
* **Conjunction (与/合取):** $p \wedge q$ (true when both are true)
* **Disjunction (或/析取):** $p \vee q$ (true when $p$ or $q$ is true)
* **Exclusive Or (异或):** $p \oplus q$ (true when exactly one of $p, q$ is true)
* **Implication (蕴含):** $p \rightarrow q$ (false only when $p$ is true and $q$ is false)
* **Biconditional (双向蕴含):** $p \leftrightarrow q$ (true if $p$ and $q$ have the same truth values)

**Truth Table for Connectives:**
| $p$ | $q$ | $\neg p$ | $p \wedge q$ | $p \vee q$ | $p \oplus q$ | $p \rightarrow q$ | $p \leftrightarrow q$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| T | T | F | T | T | F | T | T |
| T | F | F | F | T | T | F | F |
| F | T | T | F | T | T | T | F |
| F | F | T | F | F | F | T | T |

**Order of Precedence:** $\neg > \wedge > \vee > \rightarrow > \leftrightarrow$

**Implication Variations:**
* Converse (逆): $q \rightarrow p$
* Inverse (否): $\neg p \rightarrow \neg q$
* Contrapositive (逆否): $\neg q \rightarrow \neg p$ ($p \rightarrow q \equiv \neg q \rightarrow \neg p$)

---

## 2. Logical Equivalence
Two propositions are logically equivalent ($p \equiv q$ or $p \leftrightarrow q$ is a tautology) if they always have the same truth value.

* **Tautology (恒真式):** Always true (e.g., $p \vee \neg p$).
* **Contradiction (恒假式):** Always false (e.g., $p \wedge \neg p$).
* **Contingency (偶然式):** Neither a tautology nor a contradiction.

### Important Logical Equivalences
* **Identity laws:** $p \wedge T \equiv p$, $p \vee F \equiv p$
* **Domination laws:** $p \vee T \equiv T$, $p \wedge F \equiv F$
* **Idempotent laws:** $p \vee p \equiv p$, $p \wedge p \equiv p$
* **Double negation law:** $\neg(\neg p) \equiv p$
* **Commutative laws:** $p \vee q \equiv q \vee p$, $p \wedge q \equiv q \wedge p$
* **Associative laws:** $(p \vee q) \vee r \equiv p \vee (q \vee r)$, $(p \wedge q) \wedge r \equiv p \wedge (q \wedge r)$
* **Distributive laws:** $p \wedge (q \vee r) \equiv (p \wedge q) \vee (p \wedge r)$, $p \vee (q \wedge r) \equiv (p \vee q) \wedge (p \vee r)$
* **De Morgan's laws:** $\neg(p \vee q) \equiv \neg p \wedge \neg q$, $\neg(p \wedge q) \equiv \neg p \vee \neg q$
* **Absorption laws:** $p \vee (p \wedge q) \equiv p$, $p \wedge (p \vee q) \equiv p$
* **Useful law:** $p \rightarrow q \equiv \neg p \vee q$

---

## 3. Predicate Logic
Remedies propositional logic by allowing statements with constants, variables, predicates, and quantifiers.

* **Predicate (谓词):** Represents properties/relations. E.g., $P(x)$ becomes a proposition when $x$ is substituted.
* **Universe (论域):** The set of all values that may be substituted.

### Quantifiers
* **Universal Quantification (全称量化):** $\forall x P(x)$ ("For all $x$, $P(x)$ is true").
* **Existential Quantification (存在量化):** $\exists x P(x)$ ("There exists an $x$ such that $P(x)$ is true").
* **Precedence:** $\forall, \exists > \neg > \wedge > \vee > \rightarrow > \leftrightarrow$

**Negation of Quantified Statements (De Morgan's for Quantifiers):**
* $\neg \forall x P(x) \equiv \exists x \neg P(x)$
* $\neg \exists x P(x) \equiv \forall x \neg P(x)$

**Nested Quantifiers:**
* Order matters for different quantifiers: $\forall x \exists y L(x, y) \neq \exists y \forall x L(x, y)$.
* Order does not matter for the same type: $\forall x \forall y P(x, y) \equiv \forall y \forall x P(x, y)$.

---

## 4. Rules of Inference & Formal Proofs
**Vocabulary:** Axiom/Postulate, Theorem, Proof, Lemma, Corollary, Conjecture.

### Rules of Inference for Propositional Logic
| Rule of Inference | Tautology | Name |
| :--- | :--- | :--- |
| $p \rightarrow q$<br>$p$<br>$\therefore q$ | $(p \wedge (p \rightarrow q)) \rightarrow q$ | Modus ponens |
| $p \rightarrow q$<br>$\neg q$<br>$\therefore \neg p$ | $(\neg q \wedge (p \rightarrow q)) \rightarrow \neg p$ | Modus tollens |
| $p \rightarrow q$<br>$q \rightarrow r$<br>$\therefore p \rightarrow r$ | $((p \rightarrow q) \wedge (q \rightarrow r)) \rightarrow (p \rightarrow r)$ | Hypothetical syllogism |
| $p \vee q$<br>$\neg p$<br>$\therefore q$ | $( \neg p \wedge (p \vee q)) \rightarrow q$ | Disjunctive syllogism |
| $p$<br>$\therefore p \vee q$ | $p \rightarrow (p \vee q)$ | Addition |
| $p \wedge q$<br>$\therefore p$ | $(p \wedge q) \rightarrow p$ | Simplification |
| $p$<br>$q$<br>$\therefore p \wedge q$ | $((p) \wedge (q)) \rightarrow (p \wedge q)$ | Conjunction |
| $p \vee q$<br>$\neg p \vee r$<br>$\therefore q \vee r$ | $((p \vee q) \wedge (\neg p \vee r)) \rightarrow (q \vee r)$ | Resolution |

### Rules of Inference for Quantified Statements
* **Universal instantiation (UI):** $\forall x P(x) \rightarrow P(c)$
* **Universal generalization (UG):** $P(c)$ for an arbitrary $c \rightarrow \forall x P(x)$
* **Existential instantiation (EI):** $\exists x P(x) \rightarrow P(c)$ for some element $c$
* **Existential generalization (EG):** $P(c)$ for some element $c \rightarrow \exists x P(x)$

---

## 5. Methods of Theorem Proving (Informal Proofs)
* **Direct Proof:** Prove $p \rightarrow q$ by assuming $p$ is true and showing $q$ follows.
* **Proof by Contrapositive:** Prove $p \rightarrow q$ by showing $\neg q \rightarrow \neg p$ is true.
* **Proof by Contradiction:** To prove $p \rightarrow q$, assume $p \wedge \neg q$ is true, and show this leads to a contradiction.
* **Proof by Cases:** To prove $(p_1 \vee p_2 \vee \dots \vee p_n) \rightarrow q$, prove $(p_1 \rightarrow q) \wedge (p_2 \rightarrow q) \wedge \dots \wedge (p_n \rightarrow q)$.
* **Proof by Equivalence:** To prove $p \leftrightarrow q$, prove $(p \rightarrow q) \wedge (q \rightarrow p)$.
* **Vacuous Proof:** If $p$ is always false, $p \rightarrow q$ is true.
* **Trivial Proof:** If $q$ is always true, $p \rightarrow q$ is true.

---

## Assignment & Project Info
* **Assignment 1:** Deadline Sep 30. English only, single PDF. Must submit Declaration Form.
* **Optional Project (-1 ~ +5 points):** Deadline Dec 26. Individual project. Submit a self-contained report with clear references and supplementary materials. Must involve technical details related to the course.
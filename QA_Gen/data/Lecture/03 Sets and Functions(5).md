# 03 Sets and Functions (集合与函数)
**Instructor:** Shan Chen | **Course:** CS201 Discrete Mathematics

## 1. Russell's Paradox (罗素悖论)
* Let $S = \{x | x \notin x\}$ be a set of sets that are not members of themselves.
* **Paradox:**
  * If $P$ is a property, then the set $\{x | P(x)\}$ exists (naive set theory): $S$ must exist.
  * $S \in S$? S does not satisfy the property, so $S \notin S$.
  * $S \notin S$? S is included in the set S, so $S \in S$.
  * $S \in S \leftrightarrow S \notin S$: $S$ does not exist.
* **Solution:** axiomatic set theory (e.g., Zermelo-Fraenkel) (out of the scope of this course).

## 2. Sets (集合)
* A set (集合) is an unordered collection of objects. These objects are called elements or members.
* Two sets $A, B$ are equal if and only if $\forall x(x \in A \leftrightarrow x \in B)$.
* **Examples:** $S = \{2, 3, 5, 7\}$, $A = \{1, 2, 3, ..., 100\}$, $B = \{a \ge 2 | a \text{ is a prime}\}$, $C = \{2n | n = 0, 1, 2, ...\}$.
* **Different ways to represent a set:**
  * Listing (enumerating) the elements.
  * Using ellipses "..." if enumeration is hard.
  * Set builder: $\{x | x \text{ has property } P\}$ or $\{x | P(x)\}$.

### Important Sets
* Natural numbers (自然数): $\mathbb{N} = \{0, 1, 2, 3, ...\}$
* Integers (整数): $\mathbb{Z} = \{..., -2, -1, 0, 1, 2, ...\}$
* Positive integers (正整数): $\mathbb{Z}^+ = \{1, 2, 3, ...\}$
* Rational numbers (有理数): $\mathbb{Q} = \{p/q | p, q \in \mathbb{Z}, q \ne 0\}$
* Real numbers (实数): $\mathbb{R}$
* Complex numbers (复数): $\mathbb{C} = \{a+bi | a, b \in \mathbb{R}\}$

### Interval Notation (区间表示法)
* Closed interval (闭区间): $[a, b] = \{x | a \le x \le b\}$
* Open interval (开区间): $(a, b) = \{x | a < x < b\}$
* Half-open interval (半开区间):
  * Left-closed right-open interval (左闭右开区间): $[a, b) = \{x | a \le x < b\}$
  * Left-open right-closed interval (左开右闭区间): $(a, b] = \{x | a < x \le b\}$

### Special Sets and Venn Diagrams
* **Universal Set (全集):** the set of all objects under consideration, denoted by $U$.
* **Empty Set (空集):** the set of no object, denoted by $\emptyset$ or $\{\}$. Note that $\emptyset \ne \{\emptyset\}$.
* Sets can be visualized using Venn diagrams.

### Subsets and Proper Subsets (子集与真子集)
* A set $A$ is called a **subset (子集)** of $B$, denoted by $A \subseteq B$, if and only if every element of $A$ is an element of $B$: $\forall x(x \in A \rightarrow x \in B)$.
* If $A \subseteq B$ but $A \ne B$, then we say $A$ is a **proper subset (真子集)** of $B$, denoted by $A \subset B$, i.e., $\forall x(x \in A \rightarrow x \in B) \wedge \exists x(x \in B \wedge x \notin A)$.
* Two sets are equal if and only if each is a subset of the other: $A = B \leftrightarrow (A \subseteq B \wedge B \subseteq A)$.
* $\forall x(x \in A \leftrightarrow x \in B) \leftrightarrow (\forall x(x \in A \rightarrow x \in B) \wedge \forall x(x \in B \rightarrow x \in A))$.

**Subset Properties:**
* **Theorem:** $\emptyset \subseteq S$. (Proof: By definition, $\forall x(x \in \emptyset \rightarrow x \in S)$. Since $\emptyset$ has no elements, $x \in \emptyset$ is always false, making the implication always true via vacuous proof).
* **Theorem:** $S \subseteq S$. (Proof: $\forall x(x \in S \rightarrow x \in S)$ is obviously true).

## 3. Set Operations (集合运算)
* **Union (并):** $A \cup B = \{x | x \in A \vee x \in B\}$.
* **Intersection (交):** $A \cap B = \{x | x \in A \wedge x \in B\}$.
  * Two sets $A$ and $B$ are **disjoint (互斥)** if their intersection is empty: $A \cap B = \emptyset$.
* **Complement (补):** The complement of set $A$ (with respect to $U$), denoted by $\overline{A}$, is $\{x \in U | x \notin A\}$.
* **Difference (差):** $A - B = \{x | x \in A \wedge x \notin B\}$. By definition, $A - B = A \cap \overline{B}$.

**Generalized Union and Intersection:**
* Union of a Collection: $\bigcup_{i=1}^{n} A_i = A_1 \cup A_2 \cup ... \cup A_n$.
* Intersection of a Collection: $\bigcap_{i=1}^{n} A_i = A_1 \cap A_2 \cap ... \cap A_n$.

### Set Identities (集合恒等式)
* **Identity laws (恒等律):** $A \cup \emptyset = A$, $A \cap U = A$
* **Domination laws (支配律):** $A \cup U = U$, $A \cap \emptyset = \emptyset$
* **Idempotent laws (幂等律):** $A \cup A = A$, $A \cap A = A$
* **Commutative laws (交换律):** $A \cup B = B \cup A$, $A \cap B = B \cap A$
* **Associative laws (结合律):** $A \cup (B \cup C) = (A \cup B) \cup C$, $A \cap (B \cap C) = (A \cap B) \cap C$
* **Distributive laws (分配律):** $A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$, $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$
* **Absorption laws (吸收律):** $A \cup (A \cap B) = A$, $A \cap (A \cup B) = A$
* **De Morgan's laws:** $\overline{A \cap B} = \overline{A} \cup \overline{B}$, $\overline{A \cup B} = \overline{A} \cap \overline{B}$
* **Complementation laws (双重补集律 / 补集律):** $A \cup \overline{A} = U$, $A \cap \overline{A} = \emptyset$, $\overline{\overline{A}} = A$

**Proof of De Morgan's Law ($\overline{A \cap B} = \overline{A} \cup \overline{B}$):**
$$
\begin{align*}
\overline{A \cap B} &= \{x | x \in \overline{A \cap B}\} \quad (\text{Definition}) \\
&= \{x | x \notin (A \cap B)\} \quad (\text{Definition of complement}) \\
&= \{x | \neg(x \in A \wedge x \in B)\} \quad (\text{Definition of intersection}) \\
&= \{x | \neg(x \in A) \vee \neg(x \in B)\} \quad (\text{De Morgan's logical equivalence}) \\
&= \{x | x \notin A \vee x \notin B\} \quad (\text{Definition}) \\
&= \{x | x \in \overline{A} \vee x \in \overline{B}\} \quad (\text{Definition of complement}) \\
&= \{x | x \in \overline{A} \cup \overline{B}\} \quad (\text{Definition of union}) \\
&= \overline{A} \cup \overline{B} \quad (\text{Definition})
\end{align*}
$$

## 4. Cardinality, Tuples, and Power Sets
* **Cardinality (基数):** Let $S$ be a set. If there are exactly $n$ distinct elements in $S$, $S$ is a finite set and $n$ is the cardinality of $S$, denoted by $|S|$. A set $S$ is infinite if it is not finite.
* **Inclusion-Exclusion Principle (2 sets):** $|A \cup B| = |A| + |B| - |A \cap B|$.
* **Tuples (元组):** An n-tuple $(a_1, a_2, ..., a_n)$ is an ordered collection.
* **Cartesian Product (笛卡尔乘积):** $A \times B = \{(a,b) | a \in A \wedge b \in B\}$.
  * $A \times B \ne B \times A$ (order matters).
  * $|A \times B| = |A| \times |B|$.
  * Generalized: $A_1 \times A_2 \times ... \times A_n = \{(a_1, ..., a_n) | a_i \in A_i \text{ for } i=1..n\}$.
* **Power Sets (幂集):** Given a set $S$, the Power Set of $S$ is the set of all subsets of $S$, denoted by $\mathcal{P}(S)$.
  * If $|S| = n$, then $|\mathcal{P}(S)| = 2^n$.

**Computer Representation of Sets:**
Use bit strings. Each element in the universal set is assigned a bit (1 if in set, 0 if not).
* Union $\leftrightarrow$ Bitwise OR
* Intersection $\leftrightarrow$ Bitwise AND
* Complement $\leftrightarrow$ Bitwise XOR with all 1s

## 5. Functions (函数)
* Let $A$ and $B$ be two sets. A **function (函数)** from $A$ to $B$, denoted by $f: A \rightarrow B$, is an assignment of exactly one element of $B$ to each element of $A$.
* $f$ is also called a mapping (映射) or transformation (变换).
* **Domain (定义域):** $A$.
* **Codomain (陪域):** $B$.
* If $f(a) = b$, $b$ is the **image (像)** of $a$, and $a$ is a **preimage (原像)** of $b$.
* **Range (值域):** The set of all images of elements of $A$, denoted by $f(A) = \{f(x) | x \in A\}$.

### Injective, Surjective, Bijective
* **Injective (单射 / One-to-One):** $f(x) = f(y) \rightarrow x = y$. Alternatively (contrapositive): $x \ne y \rightarrow f(x) \ne f(y)$.
* **Surjective (满射 / Onto):** For every $b \in B$, $\exists a \in A$ such that $f(a) = b$. i.e., $f(A) = B$.
* **Bijective (双射 / One-to-one correspondence):** Both injective and surjective.
  * The identity function $I_A(a) = a$ is bijective.

**Important Theorem on Functions:**
* For an arbitrary function $f: A \rightarrow B$ with $|A| = |B| = n$, $f$ is one-to-one if and only if $f$ is onto. (Note: Set A must be finite for this to be true).

### Inverse Functions & Operations
* **Inverse (逆):** Let $f: A \rightarrow B$ be a bijection. The inverse $f^{-1}: B \rightarrow A$ assigns to $b \in B$ the unique element $a \in A$ such that $f(a) = b$.
  * Theorem: If $f$ is not a bijection, it is impossible to define the inverse function.
* **Operations:** $(f_1 + f_2)(x) = f_1(x) + f_2(x)$; $(f_1 f_2)(x) = f_1(x)f_2(x)$.
* **Composition (合成):** For $f: B \rightarrow C$ and $g: A \rightarrow B$, $(f \circ g)(x) = f(g(x))$.
  * Order matters: $f \circ g \ne g \circ f$.
  * $f^{-1} \circ f = I_A$ and $f \circ f^{-1} = I_B$.

### Floor, Ceiling, and Factorial Functions
* **Floor (向下取整):** $\lfloor x \rfloor$ is the largest integer $\le x$.
* **Ceiling (向上取整):** $\lceil x \rceil$ is the smallest integer $\ge x$.
* **Factorial (阶乘):** $n! = n \times (n-1) \times ... \times 1$. $0! = 1$.
* **Properties:**
  * $x-1 < \lfloor x \rfloor \le x \le \lceil x \rceil < x+1$
  * $\lfloor -x \rfloor = - \lceil x \rceil$
  * $\lfloor x+n \rfloor = \lfloor x \rfloor + n$ (where $n$ is an integer)

## 6. Sequences and Summations (数列与求和)
* **Arithmetic progression (等差数列):** $a, a+d, a+2d, ..., a+nd, ...$
* **Geometric progression (等比数列):** $a, ar, ar^2, ..., ar^n, ...$
* **Summations:**
  * Arithmetic sum: $\sum_{j=m}^{n} (a+jd) = (n-m+1)a + d\frac{(m+n)(n-m+1)}{2}$
  * Geometric sum: $\sum_{j=m}^{n} (ar^j) = a\frac{r^{n+1}-r^m}{r-1}$ ($r \ne 1$)
  * Infinite geometric series: $\sum_{k=0}^{\infty} x^k = \frac{1}{1-x}$ for $|x|<1$
  * Derivative of geometric series: $\sum_{k=1}^{\infty} k x^{k-1} = \frac{1}{(1-x)^2}$ for $|x|<1$
  * Sum of first $n$ integers: $\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$
  * Sum of squares: $\sum_{k=1}^{n} k^2 = \frac{n(n+1)(2n+1)}{6}$

## 7. Cardinality of Infinite Sets (无穷集的基数)
* Sets $A$ and $B$ have the same cardinality if there is a bijection between them ($|A| = |B|$).
* $|A| \le |B|$ if there is an injective function from $A$ to $B$.
* **Schröder-Bernstein Theorem:** If $|A| \le |B|$ and $|B| \le |A|$, then $|A| = |B|$.

### Countable vs Uncountable Sets (可数集与不可数集)
* **Countable (可数):** A set is countable if it is finite or has the same cardinality as $\mathbb{Z}^+$ (i.e., its elements can be listed/enumerated).
  * **Countable Sets Examples:** $\mathbb{Z}$ (Integers), $\mathbb{Q}$ (Rational numbers), The set of finite strings over a finite alphabet, The set of all Java programs.
  * *Hilbert's Grand Hotel* illustrates that a countably infinite set can always accommodate more elements (even countably infinitely more) via shifting mappings.
* **Uncountable (不可数):** A set that is not countable.
  * **Uncountable Sets Examples:** $\mathbb{R}$ (Real numbers), $\mathcal{P}(\mathbb{N})$ (Power set of natural numbers).
  * **Cantor's Diagonal Argument:** Used to prove $\mathbb{R}$ and $\mathcal{P}(\mathbb{N})$ are uncountable by showing that any assumed enumeration will always miss at least one constructed element.

### Computability and The Continuum Hypothesis
* **Computable Function (可计算函数):** There is a computer program that finds the values of this function.
* There exist uncomputable functions because the set of all computer programs is countable, but the set of all functions from $\mathbb{Z}^+$ to $\{0,1,...,9\}$ is uncountable.
* **Cantor's Theorem:** $|S| < |\mathcal{P}(S)|$ holds for any set $S$.
* **Continuum Hypothesis (连续统假设):** There does NOT exist a set $A$ such that $|\mathbb{N}| < |A| < |\mathcal{P}(\mathbb{N})|$.
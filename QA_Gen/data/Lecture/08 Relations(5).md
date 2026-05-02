# 08 Relations (CS201 Discrete Mathematics)
**Instructor:** Shan Chen (SUSTech)

## 1. Relations and Their Properties

Relations between elements of sets occur in many contexts (e.g., a business and its telephone number, a person and a relative).

### Binary Relations
**Definition:** Let $A, B$ be two sets. A binary relation $R$ from $A$ to $B$ is a subset of the Cartesian product $A \times B$.
* By definition, a binary relation $R \subseteq A \times B$ is a set of ordered pairs of the form $(a, b)$ with $a \in A$ and $b \in B$.
* We use $a \mathrel{R} b$ to denote $(a,b) \in R$ and $a \not\mathrel{R} b$ to denote $(a,b) \notin R$.
* **Functions vs. Relations:** Functions map each element in the domain to exactly one element in the codomain. Binary relations represent one-to-many relationships, acting as a generalization of functions.

### Representing Binary Relations
1.  **Directed Graph:** If $a \mathrel{R} b$, draw an arrow from $a$ to $b$ ($a \rightarrow b$).
2.  **Zero-One Matrix ($M_R$):** If $a \mathrel{R} b$, mark the table cell at $(a, b)$ as 1, otherwise 0.

**Relations between Finite Sets:** There are $2^{nm}$ binary relations from an $n$-element set $A$ to an $m$-element set $B$ (since $|A \times B| = nm$, the number of subsets is $2^{nm}$).
**Relation on a Set:** A relation on a set $A$ is a relation from $A$ to $A$.

### Properties of Relations
Consider binary relations on a finite set $A$ with $|A| = n$:
* **Reflexive (自反):** $(a,a) \in R$ for every $a \in A$. ($M_R$ has 1 in every position on its main diagonal. Directed graph has a self-loop on every node).
    * Number of reflexive relations: $2^{n(n-1)}$
* **Irreflexive (反自反):** $(a,a) \notin R$ for every $a \in A$. ($M_R$ has 0 in every position on its main diagonal. Directed graph has no self-loops).
    * Number of irreflexive relations: $2^{n(n-1)}$
* **Symmetric (对称):** $(b,a) \in R$ whenever $(a,b) \in R$. ($M_R$ is a symmetric matrix).
    * Number of symmetric relations: $2^{n(n+1)/2}$
* **Antisymmetric (反对称):** $(b,a) \in R$ and $(a,b) \in R$ implies $a = b$. ($m_{ij}=1 \implies m_{ji}=0$ for $i \neq j$).
    * Number of antisymmetric relations: $2^n 3^{n(n-1)/2}$
* **Transitive (传递):** $(a,b) \in R$ and $(b,c) \in R$ implies $(a,c) \in R$.

---

## 2. Combining Relations
Since relations are sets, we can combine relations via set operations: union ($\cup$), intersection ($\cap$), difference ($-$).
* Matrix representation equivalent: $R_1 \cap R_2$ corresponds to $M_{R_1} \wedge M_{R_2}$.

### Composite of Relations
**Definition:** Let $R$ be a relation from $A$ to $B$ and $S$ be a relation from $B$ to $C$. The composite (合成) of $R$ and $S$, denoted by $S \circ R$, consists of ordered pairs $(a, c)$, where $a \in A, c \in C$, and there exists $b \in B$ such that $(a,b) \in R$ and $(b,c) \in S$.
* **Matrix Computation:** Computed by the Boolean product of matrices: $M_{S \circ R} = M_R \odot M_S$.

### Powers of a Relation
The powers $R^n$ for $n = 1, 2, 3, \dots$ are defined recursively by $R^1 = R$ and $R^{n+1} = R^n \circ R$.
* **Theorem:** A relation $R$ on a set $A$ is transitive if and only if $R^n \subseteq R$ for $n = 1, 2, 3, \dots$
* $M_{R^n} = M_R \odot M_R \odot \dots \odot M_R$ ($n$ times).

---

## 3. n-ary Relations and Databases
**Definition:** An $n$-ary relation $R$ on sets $A_1, A_2, \dots, A_n$ is a subset of $A_1 \times \dots \times A_n$. The sets $A_i$ are called the domains of $R$. The degree of $R$ is $n$.
* **Relational Databases:** A relational database is essentially an $n$-ary relation $R$.
* **Primary Key (主键):** A domain $A_i$ where $R$ is functional (contains at most one $n$-tuple for each $a_i \in A_i$).
* **Composite Key (复合键):** A set of domains acting uniquely together when a single primary key does not exist.

### Database Operators
* **Selection Operator (选择算子) $s_C$:** Extracts $n$-tuples satisfying a condition (predicate) $C$. $s_C(R) = \{a \in R \mid C(a) = T\}$.
* **Projection Operator (投影算子) $P_{\{i_1,\dots,i_m\}}$:** Maps an $n$-ary relation to an $m$-ary relation by extracting specific columns indicated by indices $\{i_1,\dots,i_m\}$.
* **Join Operator (连接算子) $J(R_1, R_2)$:** Combines two relations based on matching values in shared domains (e.g., combining $(a,b) \in R_1$ and $(b,c) \in R_2$ into $(a,b,c)$).

---

## 4. Closures of Relations
**Definition:** A relation $S$ on $A$ with property $P$ is the closure (闭包) of $R$ w.r.t $P$ if $S$ is the minimal set containing $R$ satisfying $P$. Types include reflexive closure, symmetric closure, and transitive closure.

### Transitive Closure and Paths
* **Path:** A directed path from $a$ to $b$ of length $n$ exists in $R$ iff $(a,b) \in R^n$.
* **Connectivity Relation $R^*$:** Consists of all pairs $(a,b)$ connected by a path of any length. $R^* = \bigcup_{k=1}^\infty R^k$.
* **Theorem:** The transitive closure of $R$ equals the connectivity relation $R^*$.
* **Lemma:** If a path exists from $a$ to $b$ in an $n$-element set, a path of length $\le n$ exists (by removing cycles). Thus, $R^* = \bigcup_{k=1}^n R^k$.
* **Matrix Representation:** $M_{R^*} = M_R \vee M_R^{[2]} \vee M_R^{[3]} \vee \dots \vee M_R^{[n]}$ (where superscript $[k]$ is the Boolean matrix power).

### Algorithms for Transitive Closure
1.  **Naive Algorithm:** Successively computes matrix powers. Takes $\Theta(n^4)$ time.
2.  **Floyd-Warshall Algorithm:** Computes $M_{R^*}$ by iterating on $k$ (intermediate nodes $\le k$). Takes $\Theta(n^3)$ time.
    * Update rule: $W_{ij}^{[k]} = W_{ij}^{[k-1]} \vee \left( W_{ik}^{[k-1]} \wedge W_{kj}^{[k-1]} \right)$

---

## 5. Equivalence Relations
**Definition:** A relation $R$ on a set $S$ is an equivalence relation (等价关系) if it is reflexive, symmetric, and transitive. (e.g., $a \equiv b \pmod 3$, or strings having the same length).

### Equivalence Classes and Partitions
* **Equivalence Class (等价类):** $[a]_R = \{b \in S : (a,b) \in R\}$.
* **Theorem:** The following are equivalent: (i) $(a,b) \in R$, (ii) $[a] = [b]$, (iii) $[a] \cap [b] \neq \emptyset$.
* **Partition (划分):** A collection of nonempty disjoint subsets of $S$ whose union is $S$.
* **Theorem:** The equivalence classes of $R$ form a partition of $S$. Conversely, any partition of $S$ defines an equivalence relation.

---

## 6. Partial Orderings (偏序)
**Definition:** A relation $R$ on a set $S$ is a partial ordering (or partial order) if it is reflexive, antisymmetric, and transitive. Denoted by $(S, \preccurlyeq)$. Elements are "comparable" if $a \preccurlyeq b$ or $b \preccurlyeq a$.
* **Total Ordering (全序/Chain):** A poset where every two elements are comparable.
* **Lexicographic Ordering (字典序):** Combining posets $(A_1, \preccurlyeq_1)$ and $(A_2, \preccurlyeq_2)$ by comparing the first elements; if equal, compare the second.

### Well-Ordered Induction (良序归纳法)
**Well-Ordered Set:** A totally ordered set where every nonempty subset has a least element (e.g., $\mathbb{Z}^+$).
* **Principle:** To prove $P(x)$ for all $x \in S$: Prove for every $y \in S$, if $P(x)$ is true for all $x < y$, then $P(y)$ is true. (The basis step is implicitly covered when $y$ is the least element).

### Hasse Diagram
A visual representation of a partial ordering:
1.  Draw the directed graph.
2.  Remove all self-loops (due to reflexivity).
3.  Remove transitive edges (due to transitivity).
4.  Arrange nodes so all edges point upwards, then remove the arrowheads.

### Extremes and Bounds in Posets
* **Maximal / Minimal:** No element is strictly greater / smaller.
* **Greatest / Least:** An element that is greater / smaller than *all* other elements.
* **Upper Bound / Lower Bound:** An element $s$ where $a \preccurlyeq s$ (or $s \preccurlyeq a$) for all $a \in A$.
* **Least Upper Bound (LUB) / Greatest Lower Bound (GLB):** The minimum of all upper bounds / the maximum of all lower bounds.
* **Lattice (格):** A poset where *every* pair of elements has both a LUB and a GLB.

### Topological Sorting (拓扑排序)
Constructs a compatible total ordering from a partial ordering (e.g., scheduling tasks with dependencies).
* **Algorithm:** Repeatedly find a minimal element $a_k$ in the poset, remove it from the set, and append it to the sequence $a_1, a_2, \dots, a_n$.
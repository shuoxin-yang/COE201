# 07 Counting (CS201 Discrete Mathematics)
**Instructor:** Shan Chen (SUSTech)

## 1. The Counting Basics

Combinatorics (组合学) is the study of arrangements of objects. It is an important part of discrete mathematics. Counting objects with certain properties is an important part of combinatorics.
Examples include: the number of steps in a computer program, the number of passwords between $6 \sim 10$ characters, the number of telephone numbers with 8 digits. Counting may be very hard and not trivial. Usually it can be simplified by decomposing the problem.

### The Sum Rule
A count decomposes into a set of disjoint counts. Elements of different counts are alternatives.
**The Sum Rule:** If a count of elements can be broken down into a set of disjoint counts, where the first count yields $n_1$ elements, the second count $n_2$ elements, and k-th count $n_k$ elements, then the total number of elements is $n = n_1 + n_2 + \dots + n_k$.
*Example:* To travel from city A to B, you may either fly (12 flights), or take a train (5 trains), or take a bus (10 buses). Options: $12 + 5 + 10 = 27$.

### The Product Rule
A count decomposes into a sequence of steps. Each choice at one step can be combined with every choice at the next step.
**The Product Rule:** If a count of elements can be broken down into a sequence of $k$ steps, where the first step yields $n_1$ choices, the second step $n_2$ choices, and k-th step $n_k$ choices, then the total number of choices is $n = n_1 \times n_2 \times \dots \times n_k$.
*Example:* In an auditorium, the seats are labeled by a letter and numbers between 1 and 50 (e.g., A23). Total seats: $26 \times 50 = 1300$.

### Other Rules
**The Subtraction Rule:** If a task can be done in $n_1$ or $n_2$ ways that may overlap, then the number of ways to do the task is equal to $n_1 + n_2$ minus the number of ways that are common to the two types of ways. (This is actually the Principle of Inclusion-Exclusion for two sets: $|A \cup B| = |A| + |B| - |A \cap B|$).

**The Division Rule:** If a task can be done using a procedure that can be carried out in $n$ "fine-grained" ways, and every "giant" way $w$ corresponds to exactly $d$ of the $n$ "fine-grained" ways, then there are $n/d$ "giant" ways to do it. (e.g., How many kilobytes in one gigabyte? $10^9 / 10^3 = 10^6$).

*Complex Counting Example:* Each password is 6 to 8 characters long, where each character is a lowercase letter or a digit. Each password must contain at least one digit. How many possible passwords are there?
$P = P_6 + P_7 + P_8$
$P_6 = 36^6 - 26^6$
$P_7 = 36^7 - 26^7$
$P_8 = 36^8 - 26^8$

---

## 2. Tree Diagrams & Pigeonhole Principle

### Tree Diagrams
A tree is a structure that consists of a root, branches and leaves. It can represent a counting problem and record the choices we made for alternatives, with the possible outcomes on the leaves.
*Example:* The number of bit strings of length 4 that do not have consecutive 1s.
*Example:* The first team that wins 3 out of 5 games wins the playoff.

### The Pigeonhole Principle (鸽笼原理)
If $k$ is a positive integer and $k+1$ or more objects are placed into $k$ boxes, then there is at least one box containing two or more of the objects. (Proof by contradiction).

### The Generalized Pigeonhole Principle
If $N$ objects are placed into $k$ boxes, then there exists at least one box containing at least $\lceil N/k \rceil$ objects.
*Example:* 100 registered students. At least how many were born in the same month? $\lceil 100/12 \rceil = 9$. If 96 students left, $\lceil 96/12 \rceil = 8$.

---

## 3. Permutations and Combinations

Many counting problems can be solved by finding the number of ways to arrange or select some distinct elements from a set.
* **Permutation (排列):** An ordered arrangement of distinct objects.
* **Combination (组合):** An unordered selection of distinct objects.

### $r$-Permutations
An ordered arrangement of $r$ distinct elements from a set of size $n$ is called a $r$-permutation.
**Theorem:** Let $n, r$ be integers and $0 \le r \le n$, then there are:
$$P(n,r) = n(n-1)(n-2) \dots (n-r+1) = \frac{n!}{(n-r)!}$$
$r$-permutations of a set with $n$ distinct elements. (Note: $P(n,0) = 1$).

### $r$-Combinations
An unordered selection of $r$ distinct elements from a set of size $n$ is called an $r$-combination.
**Theorem:** Let $n, r$ be integers and $0 \le r \le n$, then there are:
$$C(n,r) = \frac{P(n,r)}{P(r,r)} = \frac{n!}{r!(n-r)!}$$
$r$-combinations of a set with $n$ distinct elements. (Note: $C(n,0) = 1$).

### Exercises
* How many different bit strings of length 7? $2^7$
* How many different functions from a set with $m$ elements to a set with $n$ elements? $n^m$
* How many injective functions from a set with $m$ elements to a set with $n$ elements ($m \le n$)? $P(n,m) = n(n-1) \dots (n-m+1)$
* How many onto functions from a set with $m$ elements to a set with $n$ elements ($m \ge n$)? $n^m - C(n,1)(n-1)^m + C(n,2)(n-2)^m - \dots + (-1)^{n-1}C(n,n-1)1^m$

---

## 4. The Birthday Problem and Attack

**The Birthday Paradox:** Suppose that 23 students are in a room. What is the probability that at least two of them share a birthday? (It's greater than a half!)
Sample space: $|S| = 365^n$.
$A_n$: "for $n$ students in a room some of them share a birthday".
$B_n$: "for $n$ students in a room none of them share a birthday".
$\#B_n = C(365, n) = 365 \times 364 \times \dots \times (365 - (n-1))$
$Pr[A_n] = 1 - \frac{\#B_n}{|S|}$

**The Birthday Attack (Cryptography):** Reduces the complexity of finding a collision for a hash function.
$Pr[B_n] = \prod_{i=1}^{n-1} \left(1 - \frac{i}{365}\right)$
$\rho(n;H) := 1 - \prod_{i=1}^{n-1} \left(1 - \frac{i}{H}\right)$
By approximation $e^{-x} \approx 1-x$, the smallest number of inputs $n$ such that the probability of a hash collision is $\ge p$ is:
$$n(p;H) \approx \sqrt{2H \ln\frac{1}{1-p}}$$

---

## 5. Binomial Coefficients and Identities

**Theorem:** For integers $n$ and $k$ such that $0 \le k \le n$, the number of $k$-element subsets of an $n$-element set is:
$$\binom{n}{k} = C(n,k) = \frac{n!}{k!(n-k)!}$$
Properties: $\binom{n}{0}=\binom{n}{n}=1$, $\binom{n}{k}=\binom{n}{n-k}$, $\sum_{k=0}^n \binom{n}{k} = 2^n$.

### Pascal's Triangle & Pascal's Identity
Each row starts and ends with a 1. Each (non-1) entry is the sum of the two entries directly above it.
**Pascal's Identity:** $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$
*Combinatorial Proof:* Let $S_1$ be the set of all $k$-element subsets. Partition $S_1$ into $S_2$ (contain $x_n$) and $S_3$ (don't contain $x_n$). $|S_1| = \binom{n}{k}$, $|S_2| = \binom{n-1}{k-1}$, $|S_3| = \binom{n-1}{k}$.

### The Binomial Theorem (二项式定理)
Let $x$ and $y$ be variables, and $n$ be a nonnegative integer.
$$\sum_{k=0}^n \binom{n}{k} x^{n-k} y^k = (x+y)^n$$
* Let $x=y=1 \implies \sum_{k=0}^n \binom{n}{k} = 2^n$
* Let $x=1, y=-1 \implies \sum_{k=0}^n (-1)^k \binom{n}{k} = 0$
* Let $y=1 \implies \sum_{k=0}^n \binom{n}{k} x^k = (1+x)^n$

### Trinomial Coefficients
Coefficient of $x^{k_1} y^{k_2} z^{k_3}$ in $(x+y+z)^n$:
$$\binom{n}{k_1} \binom{n-k_1}{k_2} = \frac{n!}{k_1!k_2!k_3!}$$
Denoted as $\binom{n}{k_1, k_2, k_3}$.

---

## 6. Inclusion-Exclusion (容斥原理)

Used in counts where the decomposition yields counting tasks with overlapping elements.
* **Two sets:** $|E \cup F| = |E| + |F| - |E \cap F|$
* **Three sets:** $|E \cup F \cup G| = |E| + |F| + |G| - |E \cap F| - |E \cap G| - |F \cap G| + |E \cap F \cap G|$
* **The Principle of Inclusion-Exclusion ($n$ sets):** Let $E_1, E_2, \dots, E_n$ be finite sets, then
$$|\cup_{i=1}^n E_i| = \sum_{k=1}^n (-1)^{k+1} \sum_{1 \le i_1 < i_2 < \dots < i_k \le n} |E_{i_1} \cap E_{i_2} \cap \dots \cap E_{i_k}|$$
*(Proof by induction using $G_i = E_i \cap E_n$)*.

### Example: Counting Onto Functions
Let $A, B$ be sets with $|A|=m$, $|B|=n$. $\#(a)$ = number of onto functions, $\#(b)$ = number of non-onto functions. $\#(a) + \#(b) = n^m$.
Let $E_i$: set of functions such that the $i$-th element of $B$ has no preimage.
$$\#(b) = |\cup_{i=1}^n E_i| = \sum_{k=1}^n (-1)^{k+1} \binom{n}{k} (n-k)^m$$

---

## 7. Solving Linear Recurrence Relations

**Definition:** A linear homogeneous recurrence relation of degree $k$ with constant coefficients is:
$$a_n = c_1 \cdot a_{n-1} + c_2 \cdot a_{n-2} + \dots + c_k \cdot a_{n-k}$$
where $c_k \ne 0$.

### Solving Recurrences of Degree 2
Characteristic Equation (CE): $r^2 - c_1 \cdot r - c_2 = 0$.
**Theorem:** If CE has two distinct roots $r_1, r_2$, solution is $a_n = \alpha_1 \cdot r_1^n + \alpha_2 \cdot r_2^n$.
*Example:* $a_n = 7a_{n-1} - 10a_{n-2}$ with $a_0=2, a_1=1 \implies a_n = 3 \cdot 2^n - 5^n$.
*Fibonacci:* $F_n = F_{n-1} + F_{n-2}$ with $F_0=0, F_1=1 \implies F_n = \frac{1}{\sqrt{5}} \left( \left(\frac{1+\sqrt{5}}{2}\right)^n - \left(\frac{1-\sqrt{5}}{2}\right)^n \right)$.

### Solving Recurrences of Degree $k$
CE: $r^k - c_1 r^{k-1} - \dots - c_k = 0$.
If $k$ distinct roots $r_1, \dots, r_k$, solution is $a_n = \alpha_1 r_1^n + \dots + \alpha_k r_k^n$.

### Degenerate Roots (Multiplicity)
If $r_0$ is a root with multiplicity 2: $a_n = (\alpha_1 + \alpha_2 \cdot n) r_0^n$.
*Example:* $a_n = 8a_{n-1} - 16a_{n-2}$ with $a_0=1, a_1=0 \implies a_n = (1-n)4^n$.
In general for root $r_i$ with multiplicity $m_i$: $\sum_{j=0}^{m_i - 1} \alpha_{i,j} n^j r_i^n$.

### Nonhomogeneous Recurrences
$$a_n = c_1 a_{n-1} + \dots + c_k a_{n-k} + F(n)$$
Solution is $a_n = p(n) + h(n)$, where $p(n)$ is a particular solution and $h(n)$ is the homogeneous solution.
*Example:* $a_n = 3a_{n-1} + 2n$ with $a_1=3 \implies a_n = \frac{11}{6} \cdot 3^n - n - \frac{3}{2}$.

---

## 8. Generating Functions (生成函数)

**Definition:** The generating function for the sequence $\{a_k\}$ is the infinite series:
$$G(x) = \sum_{k=0}^\infty a_k x^k$$
Operations: $f(x)+g(x) = \sum (a_k+b_k)x^k$, and $f(x)g(x) = \sum_{k=0}^\infty (\sum_{j=0}^k a_j b_{k-j}) x^k$.

### Useful Generating Functions
* $(1+x)^n = \sum_{k=0}^n \binom{n}{k} x^k$
* $\frac{1-x^{n+1}}{1-x} = \sum_{k=0}^n x^k$
* $\frac{1}{1-x} = \sum_{k=0}^\infty x^k = 1 + x + x^2 + \dots$
* $\frac{1}{(1-x)^2} = \sum_{k=0}^\infty (k+1)x^k$
* $\frac{1}{(1-x)^n} = \sum_{k=0}^\infty C(n+k-1, k) x^k$
* $e^x = \sum_{k=0}^\infty \frac{x^k}{k!}$

### Generating Functions for Counting
* *Integer Solutions:* Solutions to $x_1+x_2+x_3=17$ ($2 \le x_1 \le 5$, etc.) equals the coefficient of $x^{17}$ in $(x^2+x^3+x^4+x^5)\dots$
* *Multisets:* Size-17 multiset from $\{a,b,c\}$ is equivalent to $x_1+x_2+x_3=17$. Coefficient of $x^{17}$ in $1/(1-x)^3$.

### Solving Recurrences with $G(x)$
Step 1: Find closed-form expression of $G(x)$.
Step 2: Rewrite $G(x)$ as an infinite series.
*Example:* $a_k = 5a_{k-1} - 6a_{k-2}$ with $a_0=6, a_1=30$.
$G(x) - 5xG(x) + 6x^2 G(x) = \dots \implies G(x) = \frac{6}{(1-2x)(1-3x)} = \sum_{k=0}^\infty (18 \cdot 3^k - 12 \cdot 2^k)x^k$. Solution: $a_k = 18 \cdot 3^k - 12 \cdot 2^k$.